"""Attribution v4: separates the value of simulator replay (C2) from the value of the posterior (C1).

Uses the rollouts of results/attr/v3 (test, train, other) plus the generalisation suite written by
    python scripts/attribute.py --phase run --suites gen --out_dir results/attr/v3 ...

1. Replay-free baselines: a rule tree (depth-3 thresholds on six hand-picked log features) and an MLP
   on log-only features (statistics and subsampled trajectories of q, tau and actions), same training
   and test logs, no simulator replay. Compared with the replay-residual threshold rule and MLP.
2. Value of the posterior beyond accuracy: calibration (ECE, reliability diagram) of ours vs the MLPs;
   parameter estimates for the mass / friction logs (posterior mean over the class's candidates and a
   continuous parabola fit along the class's axis).
3. Generalisation without refitting: a new policy (obs-noise model_2999; faults injected only from
   initial states where model_2999 succeeds at nominal physics, so the label is clean), two numerics
   faults absent from training (2x physics substeps, solver iterations x0.25) and a compound fault
   (mass x2.5 + friction x0.4). Separately: what model_2999's own nominal failures are attributed to.
4. Torque signals under a per-joint sensor-noise model from the Panda datasheet (arm 0.055 N·m,
   wrist 0.008 N·m RMS at x1; fingers excluded: the Franka Hand has no joint torque sensor, and the
   PhysX finger joint forces are ill-posed, see the report).

Ours is version B of attribution_v3 (noise-aware normalisation; class-conditional H_other threshold).
Sections 1-3 use the v3 sensor-noise model (q 1e-3 rad, tau 0.5 on all joints) for comparability.

Usage (from ~/autorl-demo after ``source env.sh``):
    python scripts/attribution_v4.py
Writes results/attr/attribution_v4.md and results/attr/reliability.png.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attribution_eval as ev  # noqa: E402
from attr_common import CANDIDATES, CLASSES, new_rule  # noqa: E402
from traj_metrics import add_sensor_noise, joint_sd, residual_per_env  # noqa: E402

args = ev.args
LEVELS = args.noise_levels
SIGNALS = ev.SIGNALS
MAIN = ev.MAIN
TORQUE_KEYS = ev.TORQUE_KEYS
WRIST_LIMIT = 12.0  # N·m, effort limit of panda_joint5-7
ARM, WRIST, FINGERS = list(range(4)), [4, 5, 6], [7, 8]
NOISE_MODELS = {
    # name -> (q std [rad], per-joint tau std at x1, joints in the torque residual)
    "v3": (1e-3, np.full(9, 0.5), None),
    "spec": (1e-3, np.array([0.055] * 4 + [0.008] * 3 + [0.0, 0.0]), list(range(7))),
}
MASS_TRUE = {"mass_x1.5": 1.5, "mass_x2.5": 2.5, "mass_x4": 4.0, "op_mass_x1.5": 1.5, "op_mass_x2.5": 2.5, "op_mass_x4": 4.0}
FRIC_TRUE = {"fric_x0.7": 0.7, "fric_x0.4": 0.4, "fric_x0.25": 0.25, "op_fric_x0.7": 0.7, "op_fric_x0.4": 0.4, "op_fric_x0.25": 0.25}


def load(path):
    return ev.load(path)


# features


def log_features(log: dict, noisy: dict, e: int, failed: bool) -> dict[str, np.ndarray]:
    """Replay-free features of one episode: from the (noisy) log alone."""
    m = log["alive"][:, e]
    T = int(m.sum())
    q, tau = noisy["joint_pos"][m, e], noisy["applied_torque"][m, e]
    qd, act = log["joint_vel"][m, e], log["action"][m, e]
    hold = slice(max(T - 100, 0), T)
    hand = np.array([
        (np.abs(tau[:, WRIST]) > WRIST_LIMIT - 0.5).mean(),  # wrist torque saturation
        tau[hold, 1].mean(),  # joint2 torque while holding
        tau[hold][:, FINGERS].mean(),  # finger force while holding
        (act[:, 7] < 0).mean(),  # fraction of steps the gripper is commanded closed
        T / 250.0,
        float(failed),
    ])
    stats = np.concatenate([
        q.mean(0), q.std(0), q.min(0), q.max(0), tau.mean(0), tau.std(0), tau.min(0), tau.max(0),
        qd.std(0), np.abs(qd).max(0), act.mean(0), act.std(0), hand,
    ])
    idx = np.minimum(np.arange(0, 250, 10), T - 1)
    traj = np.concatenate([q[idx].ravel(), tau[idx].ravel()])
    return {"hand": hand, "all": np.concatenate([stats, traj])}


def build(noise_model: str) -> list[dict]:
    """Per-episode residuals (noise-aware normalisation), deviations, log-only features, at every level."""
    q_std, tau_std, tj = NOISE_MODELS[noise_model]
    with open(os.path.join(args.dir, "labels.json")) as f:
        labels = json.load(f)
    refs = [load(f"logs/ref_s{s}.npz") for s in args.seeds]
    pooled = {k: np.concatenate([r[k] for r in refs], axis=1) for k in ("joint_pos", "alive") + TORQUE_KEYS}
    sds = {sig: joint_sd(pooled, key or "applied_torque") for sig, key in SIGNALS.items()}

    def sd_at(sig, level):
        sd_tau, sd_q = sds[sig]
        return np.sqrt(sd_tau**2 + (level * tau_std) ** 2), np.sqrt(sd_q**2 + (level * q_std) ** 2)

    def noisy_copy(traj, level, key):
        return add_sensor_noise(traj, level * q_std, level * tau_std, np.random.default_rng(key), TORQUE_KEYS)

    floors, kept_floor = {}, []
    for level in LEVELS:
        vals = {sig: [] for sig in SIGNALS}
        for s in args.seeds:
            a, b = load(f"replay/ref_s{s}__nominal.npz"), load(f"replay/ref_s{s}__nominal_b.npz")
            noisy = noisy_copy(a, level, [int(level * 10), s, 7])
            for sig, key in SIGNALS.items():
                vals[sig].append(residual_per_env(noisy, b, *sd_at(sig, level), torque_key=key, torque_joints=tj)["residual"])
            if level == 1.0:  # kept set: always the v3 rule (plain sd, v3 noise, all joints) so every table uses the same logs
                v3 = add_sensor_noise(a, 1e-3, 0.5, np.random.default_rng([10, s, 7]), TORQUE_KEYS)
                kept_floor.append(residual_per_env(v3, b, *sds[MAIN], torque_key=SIGNALS[MAIN])["residual"])
        for sig in SIGNALS:
            v = np.concatenate(vals[sig])
            floors[(sig, level)] = {"mean": v.mean(), "sigma2": max(v.mean(), 1e-6), "thr": v.mean() + 4 * v.std() + 1e-12}
    kept_floor = np.concatenate(kept_floor)
    dev_thr = kept_floor.mean() + 4 * kept_floor.std() + 1e-12

    # model_2999's nominal outcome per (seed, env): faults are only evaluated from initial states it solves
    op_nominal_success = {}
    for bid, lab in labels.items():
        if lab["setting"] == "op_nominal":
            op_nominal_success[lab["seed"]] = load(f"logs/{bid}.npz")["success"]

    episodes = []
    for bid in sorted(labels):
        lab = labels[bid]
        log = load(f"logs/{bid}.npz")
        reps = {c: load(f"replay/{bid}__{c}.npz") for c in CANDIDATES}
        clean_nom = residual_per_env(log, reps["nominal"], *sds[MAIN], torque_key=SIGNALS[MAIN])["residual"]
        per_level = {}
        for level in LEVELS:
            noisy = noisy_copy(log, level, [int(level * 10), int(bid[1:]), ord(bid[0])])
            res = {sig: {c: residual_per_env(noisy, reps[c], *sd_at(sig, level), torque_key=key, torque_joints=tj)
                         for c in CANDIDATES} for sig, key in SIGNALS.items()}
            steps = min(len(noisy["alive"]), len(reps["nominal"]["alive"]))
            mask = (noisy["alive"][:steps] & reps["nominal"]["alive"][:steps])[..., None]
            n = np.maximum(mask.sum(axis=0), 1)
            sd_tau, sd_q = sd_at(MAIN, level)
            dev_tau = (np.abs(reps["nominal"]["applied_torque"][:steps] - noisy["applied_torque"][:steps]) * mask).sum(0) / n / sd_tau
            dev_q = (np.abs(reps["nominal"]["joint_pos"][:steps] - noisy["joint_pos"][:steps]) * mask).sum(0) / n / sd_q
            per_level[level] = (noisy, res, dev_tau, dev_q)
        for e in range(log["alive"].shape[1]):
            failed = not bool(log["success"][e])
            ep = {
                "batch": bid, "suite": {"b": "test", "t": "train", "o": "other", "g": "gen"}[bid[0]], "env": e,
                "seed": lab["seed"], "setting": lab["setting"], "true": lab["true_class"], "failed": failed,
                "final_dist_cm": float(log["final_dist"][e] * 100), "clean_nominal": float(clean_nom[e]), "levels": {},
            }
            if lab["setting"].startswith("op_"):  # new-policy suite: same initial states as op_nominal
                ep["op_nominal_success"] = bool(op_nominal_success[lab["seed"]][e])
            for level, (noisy, res, dev_tau, dev_q) in per_level.items():
                ep["levels"][level] = {
                    "res": {sig: {c: float(r[c]["residual"][e]) for c in CANDIDATES} for sig, r in res.items()},
                    "terms": {c: (float(res[MAIN][c]["torque_term"][e]), float(res[MAIN][c]["pos_term"][e])) for c in CANDIDATES},
                    "dev_tau": dev_tau[e], "dev_q": dev_q[e], "log": log_features(log, noisy, e, failed),
                }
            ep["deviating"] = ep["clean_nominal"] > dev_thr
            ep["kept"] = ep["failed"] or ep["deviating"]
            episodes.append(ep)
    return episodes, floors


# replay-free rule tree


def fit_tree(x, y, depth=3, min_leaf=3):
    """CART with class-balanced Gini; thresholds at feature quantiles."""
    w = (len(y) / (len(CLASSES) * np.maximum(np.bincount(y, minlength=len(CLASSES)), 1)))[y]

    def gini(idx):
        p = np.bincount(y[idx], weights=w[idx], minlength=len(CLASSES))
        return w[idx].sum() * (1 - ((p / max(p.sum(), 1e-12)) ** 2).sum())

    def grow(idx, d):
        counts = np.bincount(y[idx], weights=w[idx], minlength=len(CLASSES))
        if d == 0 or len(idx) < 2 * min_leaf or (counts > 0).sum() <= 1:
            return int(counts.argmax())
        best = None
        for f in range(x.shape[1]):
            for t in np.unique(np.quantile(x[idx, f], np.linspace(0.05, 0.95, 19))):
                left, right = idx[x[idx, f] <= t], idx[x[idx, f] > t]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                g = gini(left) + gini(right)
                if best is None or g < best[0]:
                    best = (g, f, t, left, right)
        if best is None:
            return int(counts.argmax())
        _, f, t, left, right = best
        return (f, t, grow(left, d - 1), grow(right, d - 1))

    tree = grow(np.arange(len(y)), depth)

    def predict(v):
        node = tree
        while isinstance(node, tuple):
            node = node[2] if v[node[0]] <= node[1] else node[3]
        return CLASSES[node]

    predict.tree = tree
    return predict


HAND_NAMES = ["wrist_saturation", "tau2_hold", "finger_force_hold", "grip_closed_frac", "episode_len", "failed"]


def describe_tree(node, depth=0) -> list[str]:
    if not isinstance(node, tuple):
        return [f"{'  ' * depth}-> {CLASSES[node]}"]
    f, t, left, right = node
    return ([f"{'  ' * depth}if {HAND_NAMES[f]} <= {t:.3g}:"] + describe_tree(left, depth + 1)
            + [f"{'  ' * depth}else:"] + describe_tree(right, depth + 1))


# posterior quantities


def ours_post(ep, level, floors, signal=MAIN):
    f = floors[(signal, level)]
    return new_rule(ep["levels"][level]["res"][signal], ep["failed"], f["sigma2"], f["thr"])


def axis_fit(res, axis):
    """Continuous multiplier along one axis: parabola in log-multiplier through the best grid point and neighbours."""
    x = np.log([m for m, _ in axis])
    r = np.array([res[c] for _, c in axis])
    order = np.argsort(x)
    x, r = x[order], r[order]
    k = int(np.argmin(r))
    lo = min(max(k - 1, 0), len(x) - 3)
    a, b, c = np.polyfit(x[lo: lo + 3], r[lo: lo + 3], 2)
    if a > 0:
        xs = float(np.clip(-b / (2 * a), x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]))
        if np.polyval([a, b, c], xs) <= r[k]:
            return float(np.exp(xs))
    return float(np.exp(x[k]))


def posterior_mean(res, cls, sigma2):
    cands = [c for c in CANDIDATES if CANDIDATES[c][3] == cls]
    ll = np.array([-res[c] / (2 * sigma2) for c in cands])
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return float(np.exp((w * np.log([CANDIDATES[c][1] for c in cands])).sum()))


def ece(confs, correct, bins=10):
    confs, correct = np.asarray(confs), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confs[m].mean())
            rows.append((confs[m].mean(), correct[m].mean(), int(m.sum())))
    return total, rows


def main():
    episodes, floors = build("v3")
    kept = [ep for ep in episodes if ep["kept"]]
    test = [ep for ep in kept if ep["suite"] == "test"]
    train = [ep for ep in kept if ep["suite"] == "train"]
    gen = [ep for ep in episodes if ep["suite"] == "gen"]
    gen_op = [ep for ep in gen if ep["kept"] and ep.get("op_nominal_success") and ep["setting"] != "op_nominal"]
    gen_nu = [ep for ep in gen if ep["kept"] and ep["setting"].startswith("nu_")]
    gen_compound = [ep for ep in gen if ep["kept"] and ep["setting"].startswith("compound")]
    op_nominal_failed = [ep for ep in gen if ep["setting"] == "op_nominal" and ep["failed"]]
    rng = np.random.default_rng(0)
    out = {"lines": []}
    L = out["lines"]

    tau_other = {(sig, level, cls): float(np.percentile(
        [min(ep["levels"][level]["res"][sig].values()) for ep in train if ep["true"] == cls], args.other_quantile))
        for sig in SIGNALS for level in LEVELS for cls in CLASSES}

    preds, probs, models = {}, {}, {}
    evaluated = test + gen_op + gen_nu + gen_compound + op_nominal_failed + [ep for ep in kept if ep["suite"] == "other"]
    for level in LEVELS:
        mlp_replay = ev.train_supervised(train, level, floors)
        mlp_log = ev.train_supervised(train, level, floors, feat=lambda ep, lv=level: ep["levels"][lv]["log"]["all"])
        tree = fit_tree(np.stack([ep["levels"][level]["log"]["hand"] for ep in train]), np.array([CLASSES.index(ep["true"]) for ep in train]))
        thr = ev.fit_threshold(train, level, floors)
        models[level] = {"tree": tree, "thr": thr}
        methods = {
            "ours": lambda ep: ev.predict_ours(ep, level, floors),
            "ours+other": lambda ep: ev.predict_ours_other(ep, level, floors, tau_other),
            "threshold (replay)": lambda ep: ev.threshold_predict(ep, level, floors, *thr),
            "MLP (replay)": mlp_replay,
            "rule tree (log only)": lambda ep: tree(ep["levels"][level]["log"]["hand"]),
            "MLP (log only)": mlp_log,
            "SysID-only": lambda ep: ev.predict_sysid(ep, level),
        }
        for name, fn in methods.items():
            preds[(name, level)] = {id(ep): fn(ep) for ep in evaluated}
        probs[("ours", level)] = {id(ep): ours_post(ep, level, floors) for ep in evaluated}
        probs[("MLP (replay)", level)] = {id(ep): dict(zip(CLASSES, mlp_replay.proba(ep))) for ep in evaluated}
        probs[("MLP (log only)", level)] = {id(ep): dict(zip(CLASSES, mlp_log.proba(ep))) for ep in evaluated}

    def P(name, level, eps):
        return [preds[(name, level)][id(ep)] for ep in eps]

    BA = ev.balanced_acc
    fmt = ev.fmt_ci

    def boot(eps, name, level, metric=BA):
        return ev.bootstrap(eps, P(name, level, eps), metric, rng)

    L += ["# Inverse failure attribution v4: what the replay buys (C2) and what the posterior buys (C1)", ""]
    L += [
        "Same logs as attribution_v3 (test: seeds 0-2, causes off the candidate grid; train: seeds 3-5, other intensities),",
        "plus a generalisation suite. Ours = attribution_v3 version B. Sections 1-3 use the v3 sensor-noise model",
        "(q 1e-3 rad, tau 0.5 on all joints at x1); section 4 uses a per-joint Panda datasheet model. 95% intervals:",
        f"cluster bootstrap over batches ({args.bootstrap} resamples). Implicit actuators; `applied_torque` is the PD estimate.", "",
    ]

    # 1. replay vs log-only
    L += ["## 1. Does attribution need the replay? (test logs, class-balanced accuracy %)", "",
          "| method | uses replay | " + " | ".join(f"x{lv:g}" for lv in LEVELS) + " |", "|---|---|" + "---|" * len(LEVELS)]
    for name, rep in [("ours", "yes"), ("threshold (replay)", "yes"), ("MLP (replay)", "yes"),
                      ("rule tree (log only)", "no"), ("MLP (log only)", "no")]:
        L.append(f"| {name} | {rep} | " + " | ".join(fmt(boot(test, name, lv)) for lv in LEVELS) + " |")
    L += ["", "Per class at x1:", "", "| method | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name in ["ours", "threshold (replay)", "MLP (replay)", "rule tree (log only)", "MLP (log only)"]:
        pc = ev.per_class_acc(test, P(name, 1.0, test))
        L.append(f"| {name} | " + " | ".join(f"{pc[c] * 100:.0f}" for c in CLASSES) + " |")
    L += ["", "Log-only rule tree fitted at x1 (features: " + ", ".join(HAND_NAMES) + "):", "", "```"]
    L += describe_tree(models[1.0]["tree"].tree) + ["```", ""]
    L += ["Log-only MLP input: per-joint mean/std/min/max of q and tau, std and max |qd|, action mean/std, the six rule-tree",
          "features, and q, tau subsampled every 10 steps (560 features).", ""]

    # 2. calibration and parameter estimates
    L += ["## 2. What the posterior adds beyond accuracy", "", "### Calibration (test logs; ECE over 10 confidence bins, %)", "",
          "| method | " + " | ".join(f"x{lv:g}" for lv in LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    rel = {}
    for name in ("ours", "MLP (replay)", "MLP (log only)"):
        cells = []
        for lv in LEVELS:
            ps = [probs[(name, lv)][id(ep)] for ep in test]
            conf = [max(p.values()) for p in ps]
            corr = [max(p, key=p.get) == ep["true"] for p, ep in zip(ps, test)]
            e, rows = ece(conf, corr)
            rel[(name, lv)] = rows
            cells.append(f"{e * 100:.1f} (acc {np.mean(corr) * 100:.0f}, mean conf {np.mean(conf) * 100:.0f})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "`ours` at x0 uses sigma^2 = 1e-6 (the replay floor is exactly 0), so its posterior is essentially one-hot there.",
          "Reliability diagram: `reliability.png`.", ""]
    plot_reliability(rel)

    L += ["### Parameter estimates (test logs of the mass and friction classes, noise x1)", "",
          "Estimated multiplier along the true class's axis. `posterior mean`: likelihood-weighted geometric mean over the",
          "class's grid candidates (mass 2/3/5, friction 0.5/0.3/0.2). `continuous fit`: parabola in log-multiplier through",
          "the residuals of nominal and the grid (can leave the grid). The threshold rule and the MLPs give no estimate.", "",
          "| true setting | n | posterior mean: median (median abs rel. error) | continuous fit: median (median abs rel. error) |",
          "|---|---|---|---|"]
    lvl = 1.0
    for table, axis, cls in ((MASS_TRUE, ev.MASS_AXIS, "H_theta-mass"), (FRIC_TRUE, ev.FRIC_AXIS, "H_theta-friction")):
        for setting, true in table.items():
            eps = [ep for ep in test if ep["setting"] == setting]
            if not eps:
                continue
            pm = np.array([posterior_mean(ep["levels"][lvl]["res"][MAIN], cls, floors[(MAIN, lvl)]["sigma2"]) for ep in eps])
            cf = np.array([axis_fit(ep["levels"][lvl]["res"][MAIN], axis) for ep in eps])
            L.append(f"| {setting} | {len(eps)} | {np.median(pm):.2f} ({np.median(np.abs(pm / true - 1)) * 100:.0f}%) |"
                     f" {np.median(cf):.2f} ({np.median(np.abs(cf / true - 1)) * 100:.0f}%) |")

    # 3. generalisation
    L += ["", "## 3. Generalisation without refitting (class-balanced accuracy %, noise x1 unless stated)", "",
          f"New policy: obs-noise `model_2999` (nominal success {np.mean([not ep['failed'] for ep in gen if ep['setting'] == 'op_nominal']) * 100:.0f}%"
          " over these initial states); faults injected only from initial states it solves at nominal physics; its H_pi fault is"
          " obs-noise `model_1499`. New numerics: physics dt 0.005 with decimation 4 (same control rate), solver position",
          "iterations x0.25 (robot 8 -> 2, cube 16 -> 4); baseline policy. Compound: cube mass x2.5 and friction x0.4.", ""]
    L += ["Kept logs: new policy " + ", ".join(f"{c} {sum(ep['true'] == c for ep in gen_op)}" for c in CLASSES)
          + f"; new numerics {len(gen_nu)} ("
          + ", ".join(f"{s} {sum(ep['setting'] == s for ep in gen_nu)}/{sum(ep['setting'] == s for ep in gen)}" for s in ("nu_substeps_x2", "nu_solver_iters_x0.25"))
          + f"); compound {len(gen_compound)}/{sum(ep['setting'].startswith('compound') for ep in gen)}.", ""]
    names = ["ours", "threshold (replay)", "MLP (replay)", "rule tree (log only)", "MLP (log only)"]
    L += ["| method | Lift test (ref) | new policy | drop | new numerics -> H_nu | compound -> mass or friction |",
          "|---|---|---|---|---|---|"]
    for name in names:
        a = boot(test, name, lvl)
        b = boot(gen_op, name, lvl)
        nu = ev.bootstrap(gen_nu, P(name, lvl, gen_nu), lambda e, q: np.mean([x == "H_nu" for x in q]), rng) if gen_nu else (np.nan,) * 3
        cp = ev.bootstrap(gen_compound, P(name, lvl, gen_compound),
                          lambda e, q: np.mean([x in ("H_theta-mass", "H_theta-friction") for x in q]), rng) if gen_compound else (np.nan,) * 3
        L.append(f"| {name} | {fmt(a)} | {fmt(b)} | {(a[0] - b[0]) * 100:+.0f} | {fmt(nu)} | {fmt(cp)} |")
    L += ["", "Per class on the new policy (x1):", "", "| method | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name in names:
        pc = ev.per_class_acc(gen_op, P(name, lvl, gen_op))
        L.append(f"| {name} | " + " | ".join("-" if np.isnan(pc[c]) else f"{pc[c] * 100:.0f}" for c in CLASSES) + " |")
    cols = CLASSES + ["H_other"]
    L += ["", "New numerics, predictions per setting (x1; `ours+other` adds the class-conditional H_other criterion):", "",
          "| setting | method | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for s in ("nu_substeps_x2", "nu_solver_iters_x0.25"):
        eps = [ep for ep in gen_nu if ep["setting"] == s]
        for name in names + ["ours+other"]:
            p = P(name, lvl, eps)
            L.append(f"| {s} | {name} | " + " | ".join(str(p.count(c)) for c in cols) + " |")
        r = np.median([min(ep["levels"][lvl]["res"][MAIN].values()) for ep in eps])
        best = max(set(b := [min(ep["levels"][lvl]["res"][MAIN], key=ep["levels"][lvl]["res"][MAIN].get) for ep in eps]), key=b.count)
        L.append(f"| {s} | (median min residual {r:.3g}, most common best candidate {best}; floor {floors[(MAIN, lvl)]['mean']:.3g}) | | | | | |")
    if gen_compound:
        L += ["", "Compound fault (x1): predictions, and ours' continuous estimates along each axis (true mass 2.5, friction 0.4):", "",
              "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name in names + ["ours+other"]:
            p = P(name, lvl, gen_compound)
            L.append(f"| {name} | " + " | ".join(str(p.count(c)) for c in cols) + " |")
        m_est = [axis_fit(ep["levels"][lvl]["res"][MAIN], ev.MASS_AXIS) for ep in gen_compound]
        f_est = [axis_fit(ep["levels"][lvl]["res"][MAIN], ev.FRIC_AXIS) for ep in gen_compound]
        L += ["", f"Median continuous estimates: mass x{np.median(m_est):.2f}, friction x{np.median(f_est):.2f}"
              " (each axis fitted with the other at nominal, since the grid has no joint candidates)."]
    L += ["", f"model_2999's own nominal failures ({len(op_nominal_failed)} episodes; ideal answer H_pi), x1:", "",
          "| method | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name in names:
        p = P(name, lvl, op_nominal_failed)
        L.append(f"| {name} | " + " | ".join(str(p.count(c)) for c in CLASSES) + " |")
    L += ["", "Generalisation at every noise level (class-balanced accuracy on the new-policy logs):", "",
          "| method | " + " | ".join(f"x{lv:g}" for lv in LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for name in names:
        L.append(f"| {name} | " + " | ".join(f"{BA(gen_op, P(name, lv, gen_op)) * 100:.0f}" for lv in LEVELS) + " |")

    # 4. torque signals with a per-joint datasheet noise model
    spec_eps, spec_floors = build("spec")
    spec_kept = [ep for ep in spec_eps if ep["kept"]]
    s_test = [ep for ep in spec_kept if ep["suite"] == "test"]
    s_train = [ep for ep in spec_kept if ep["suite"] == "train"]
    s_other = [ep for ep in spec_kept if ep["suite"] == "other"]
    s_tau_other = {(sig, level, cls): float(np.percentile(
        [min(ep["levels"][level]["res"][sig].values()) for ep in s_train if ep["true"] == cls], args.other_quantile))
        for sig in SIGNALS for level in LEVELS for cls in CLASSES}
    L += ["", "## 4. Torque signals under a per-joint sensor-noise model", "",
          "x1 = Panda datasheet torque noise (RMS): 0.055 N·m on joints 1-4 (the larger of the two published figures, 0.005 and",
          "0.055), scaled by sensor range (12/87) to 0.008 N·m on the wrist joints 5-7; q noise 1e-3 rad. The finger joints are left",
          "out of the torque residual: the Franka Hand has no joint torque sensors, and PhysX's projected finger joint force is",
          "ill-posed (below). Ours, version B, test logs, class-balanced accuracy %; then H_other detection / 4-class accuracy.", "",
          "| signal | " + " | ".join(f"x{lv:g}" for lv in LEVELS) + " | H_other detected / 4-class, x1 |", "|---|" + "---|" * (len(LEVELS) + 1)]
    for sig in SIGNALS:
        accs = [BA(s_test, [ev.predict_ours(ep, lv, spec_floors, sig) for ep in s_test]) for lv in LEVELS]
        det = np.mean([ev.predict_ours_other(ep, 1.0, spec_floors, s_tau_other, sig) == "H_other" for ep in s_other])
        four = BA(s_test, [ev.predict_ours_other(ep, 1.0, spec_floors, s_tau_other, sig) for ep in s_test])
        L.append(f"| {sig} | " + " | ".join(f"{a * 100:.0f}" for a in accs) + f" | {det * 100:.0f} / {four * 100:.0f} |")
    L += ["", "Per class at x1:", "", "| signal | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for sig in SIGNALS:
        pc = ev.per_class_acc(s_test, [ev.predict_ours(ep, 1.0, spec_floors, sig) for ep in s_test])
        L.append(f"| {sig} | " + " | ".join(f"{pc[c] * 100:.0f}" for c in CLASSES) + " |")

    L += [""] + finger_section() + [""] + factory_section()
    text = "\n".join(L) + "\n"
    return text


def finger_section() -> list[str]:
    """Which finger's projected joint force carries the grip force, per reference episode."""
    counts = {"finger1 only": 0, "finger2 only": 0, "neither": 0, "both": 0}
    for s in args.seeds:
        d = load(f"logs/ref_s{s}.npz")
        for e in range(d["alive"].shape[1]):
            a = d["alive"][:, e]
            q, ts = d["joint_pos"][a, e, 7:], d["tau_solver"][a, e, 7:]
            grasp = q.max(axis=1) < 0.03
            big = (np.abs(ts[grasp]) > 5).mean(axis=0) > 0.5
            counts["both" if big.all() else "finger1 only" if big[0] else "finger2 only" if big[1] else "neither"] += 1
    return [
        "### Why the two fingers' tau_solver differ", "",
        "The two finger joints are defined identically in the USD (prismatic, no mimic joint, same drive, limits 0-0.04 m), and",
        "Isaac Lab's and PhysX's DOF orders match. During a grasp the PD estimate gives both fingers ~45-55 N, but PhysX's",
        "projected joint force reports the full grip force (~-44 N) on at most ONE finger for the whole grasp and ~0.1 N on the",
        "other. Reference episodes (" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + "). Finger 1 carries it more",
        "often, hence its larger sd. Joint limits are not the cause (<2% of the large readings occur at a limit). Interpretation",
        "(not verified): the two fingers and the cube close a kinematic loop; the squeeze is an internal force of that loop,",
        "which is statically indeterminate, and the solver attributes it to one joint arbitrarily. So the finger channels of",
        "tau_solver are not a physical measurement, and the real Franka Hand has no joint torque sensor either: section 4",
        "leaves them out. The arm channels track the PD estimate at steady state (e.g. joint2 -25.8 vs -25.7 N·m) but differ",
        "in transients and on joint7, where the PD estimate sits at its 12 N·m limit while the solver force is ~0.",
        "",
        "Torque-noise sources: Panda datasheet (torque noise RMS listed as 0.005 N·m in one revision and 0.055 N·m in",
        "another; repeatability < 0.05 N·m; relative accuracy 0.15 N·m, a bias-like error not modelled here; 13-bit",
        "resolution). The datasheet gives one figure for all joints; the wrist value is scaled by the sensor range (±12 vs",
        "±87 N·m). [Panda datasheet](https://www.wiredworkers.io/wp-content/uploads/2019/12/Panda_FrankaEmika_ENG.pdf),",
        "[instruction handbook](https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf),",
        "[datasheet (alternate revision)](https://centers.ulbsibiu.ro/incon/wp-content/uploads/Datasheet-Franka-Emika-Panda-Research-collaborative-robot.pdf).",
    ]


def factory_section() -> list[str]:
    return [
        "## 5. Second task: Factory peg insertion (survey only, nothing trained)", "",
        "- Local stack: Isaac Lab 2.0.2 on Isaac Sim 4.5; task `Isaac-Factory-PegInsert-Direct-v0` (rl_games, LSTM policy:",
        "  MLP 512-128-64 + LSTM 1024; obs 21, state 72, actions 6; sim dt 1/120, decimation 8, 10 s episodes = 150 steps).",
        "- Published pretrained checkpoint (`--use_pretrained_checkpoint` path on NVIDIA's asset server): **404 for the",
        "  Isaac Sim 4.5 asset root** that this install uses; **present for the 5.0 and 5.1 roots** (`checkpoint.pth`,",
        "  213 MB, dated Aug and Oct 2025), i.e. released for Isaac Lab 2.2/2.3. Whether it loads on 2.0.2 (same obs/state/",
        "  action sizes and network) is not verified: downloading and unpickling it was blocked in this session, and the",
        "  file needs `weights_only=False`, so it should be inspected deliberately.",
        "- Training from scratch: `scripts/factory_throughput.py` measures 552 env-steps/s with 128 envs (random actions,",
        "  one RTX 3090). The shipped config (200 epochs x 128 envs x 128-step horizon = 3.3M env steps) is then ~1.7 h of",
        "  simulation plus PPO/LSTM updates: roughly 2-2.5 h per seed on one GPU. More envs may raise throughput (untested).",
    ]


def plot_reliability(rel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ours": "#2a78d6", "MLP (replay)": "#eb6834", "MLP (log only)": "#1baf7a"}
    shown = [lv for lv in (1.0, 5.0) if lv in LEVELS]
    fig, axes = plt.subplots(1, len(shown), figsize=(5 * len(shown), 4.4), sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], shown):
        ax.plot([0, 1], [0, 1], "--", color="#9a9a96", linewidth=1)
        for name, color in colors.items():
            rows = rel[(name, lv)]
            ax.plot([r[0] for r in rows], [r[1] for r in rows], color=color, linewidth=2, label=name)
            ax.scatter([r[0] for r in rows], [r[1] for r in rows], s=[20 + 200 * r[2] / sum(x[2] for x in rows) for r in rows],
                       color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(f"sensor noise x{lv:g}", fontsize=11)
        ax.set_xlabel("confidence (max posterior)")
        ax.set_xlim(0.2, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0][0].set_ylabel("accuracy in bin")
    fig.suptitle("Reliability on the test logs (marker area = share of logs in the bin)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(os.path.dirname(args.md), "reliability.png")
    fig.savefig(path, dpi=130)
    print(f"[attribution_v4] wrote {path}")


if __name__ == "__main__":
    text = main()
    path = os.path.join(os.path.dirname(args.md), "attribution_v4.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
