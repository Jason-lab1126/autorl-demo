"""Attribution v3 evaluation (no simulator): our method against baselines, an out-of-hypothesis
criterion, and three residual signals, on the rollouts written by
    python scripts/attribute.py --phase run --suites test train other --out_dir results/attr/v3 ...

Logs: test (b*, seeds 0-2, causes off the candidate grid, same settings as attribution_v2), train
(t*, seeds 3-5, other intensities) and other (o*, seeds 0-2, faults outside the 4 classes). One log
per episode. Kept = failed, or noise-free nominal-replay residual (q + applied_torque) above the
real-robot noise floor; the kept set is fixed across methods, signals and noise levels.

Methods (all on q + applied_torque residuals unless stated):
    ours          Gaussian likelihood with the replay noise floor, class-mean likelihood (attribution_v2)
    ours+other    ours, but H_other when every candidate's residual exceeds tau_other (see the two versions)
    old rule      attribution v1 posterior
    SysID-only    continuous fit of the cube mass or friction multiplier (parabola through the grid residuals
                  in log-scale) and nothing else: predicts mass, friction, or none (fitted multiplier ~ 1)
    supervised    MLP on residual features, trained on the training logs at the same noise level
    threshold     hand rules on per-joint deviation from the nominal replay, constants fitted on training logs
    LLM-only      interface only (llm_attribution.py), not run
Signals for ours: q only, q + applied_torque (PD estimate), q + tau_solver (PhysX projected joint force).

Two versions are evaluated and both reported:
    A  residuals normalised by the per-joint sd of the nominal reference; tau_other = 95th percentile of the
       minimum candidate residual over all kept training logs (pooled)
    B  residuals normalised by sqrt(sd^2 + sigma_noise^2) per joint at each noise level, so a low-variance
       channel is not dominated by sensor noise; tau_other conditional on the class ours predicts: the 95th
       percentile of the minimum candidate residual over the kept training logs of that class
The kept set is the same in both (decided with version A's normalisation).

95% intervals: cluster bootstrap over log batches (one batch = one setting x seed, 8 episodes).

Usage (from ~/autorl-demo after ``source env.sh``):
    python scripts/attribution_eval.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attr_common import CANDIDATES, CLASSES, new_rule, old_rule  # noqa: E402
from llm_attribution import LLMAttributor  # noqa: E402
from traj_metrics import add_sensor_noise, joint_sd, residual_per_env  # noqa: E402

parser = argparse.ArgumentParser(description="Attribution v3 evaluation.")
parser.add_argument("--dir", type=str, default="results/attr/v3")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Reference seeds (noise floor, sd).")
parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0])
parser.add_argument("--q_noise", type=float, default=1e-3)
parser.add_argument("--tau_noise", type=float, default=0.5)
parser.add_argument("--other_quantile", type=float, default=95.0)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--md", type=str, default="results/attr/attribution_v3.md")
args = parser.parse_args()

SIGNALS = {"q": None, "q+applied": "applied_torque", "q+solver": "tau_solver"}
MAIN = "q+applied"
TORQUE_KEYS = ("applied_torque", "tau_solver")
MASS_AXIS = [(1.0, "nominal"), (2.0, "mass_x2"), (3.0, "mass_x3"), (5.0, "mass_x5")]
FRIC_AXIS = [(1.0, "nominal"), (0.5, "fric_x0.5"), (0.3, "fric_x0.3"), (0.2, "fric_x0.2")]
NONE_BAND = np.log(1.15)  # SysID: fitted multiplier within +-15% of nominal counts as "no parameter change"
ARM_WRIST, FINGERS, JOINT2 = [4, 5, 6], [7, 8], 1


def load(path: str) -> dict:
    with np.load(os.path.join(args.dir, path)) as f:
        return {key: f[key] for key in f.files}


# per-episode features


def build_features(noise_aware: bool):
    with open(os.path.join(args.dir, "labels.json")) as f:
        labels = json.load(f)
    refs = [load(f"logs/ref_s{s}.npz") for s in args.seeds]
    pooled = {k: np.concatenate([r[k] for r in refs], axis=1) for k in ("joint_pos", "alive") + TORQUE_KEYS}
    sds = {sig: joint_sd(pooled, key or "applied_torque") for sig, key in SIGNALS.items()}
    joint_names = [str(n) for n in refs[0]["joint_names"]]

    def sd_at(sig: str, level: float) -> tuple[np.ndarray, np.ndarray]:
        sd_tau, sd_q = sds[sig]
        if not noise_aware:
            return sd_tau, sd_q
        return np.sqrt(sd_tau**2 + (level * args.tau_noise) ** 2), np.sqrt(sd_q**2 + (level * args.q_noise) ** 2)

    # noise floor per (signal, level): two nominal replays of each reference, one with sensor noise
    floors = {}
    kept_floor = []  # x1 floor of the main signal with plain-sd normalisation: decides the kept set in every version
    for level in args.noise_levels:
        vals = {sig: [] for sig in SIGNALS}
        for s in args.seeds:
            a, b = load(f"replay/ref_s{s}__nominal.npz"), load(f"replay/ref_s{s}__nominal_b.npz")
            noisy = add_sensor_noise(a, level * args.q_noise, level * args.tau_noise,
                                     np.random.default_rng([int(level * 10), s, 7]), TORQUE_KEYS)
            for sig, key in SIGNALS.items():
                vals[sig].append(residual_per_env(noisy, b, *sd_at(sig, level), torque_key=key)["residual"])
            if level == 1.0:
                kept_floor.append(residual_per_env(noisy, b, *sds[MAIN], torque_key=SIGNALS[MAIN])["residual"])
        for sig in SIGNALS:
            v = np.concatenate(vals[sig])
            floors[(sig, level)] = {"mean": v.mean(), "sigma2": max(v.mean(), 1e-6), "thr": v.mean() + 4 * v.std() + 1e-12}

    exact = {}  # reproducibility inside the run: reference closed loop vs its nominal replays
    for s, ref in zip(args.seeds, refs):
        a, b = load(f"replay/ref_s{s}__nominal.npz"), load(f"replay/ref_s{s}__nominal_b.npz")
        exact[f"ref_s{s}"] = max(float(np.abs(ref[k] - a[k]).max()) for k in ("joint_pos",) + TORQUE_KEYS) + \
            max(float(np.abs(a[k] - b[k]).max()) for k in ("joint_pos",) + TORQUE_KEYS)

    episodes = []
    for bid in sorted(labels):
        log = load(f"logs/{bid}.npz")
        reps = {c: load(f"replay/{bid}__{c}.npz") for c in CANDIDATES}
        clean_nom = residual_per_env(log, reps["nominal"], *sds[MAIN], torque_key=SIGNALS[MAIN])["residual"]
        per_level = {}
        for level in args.noise_levels:
            noisy = add_sensor_noise(log, level * args.q_noise, level * args.tau_noise,
                                     np.random.default_rng([int(level * 10), int(bid[1:]), ord(bid[0])]), TORQUE_KEYS)
            res = {sig: {c: residual_per_env(noisy, reps[c], *sd_at(sig, level), torque_key=key) for c in CANDIDATES}
                   for sig, key in SIGNALS.items()}
            # per-joint deviation from the nominal replay (main signal), for the baselines
            steps = min(len(noisy["alive"]), len(reps["nominal"]["alive"]))
            mask = (noisy["alive"][:steps] & reps["nominal"]["alive"][:steps])[..., None]
            n = np.maximum(mask.sum(axis=0), 1)
            dev_tau = (np.abs(reps["nominal"]["applied_torque"][:steps] - noisy["applied_torque"][:steps]) * mask).sum(0) / n / sd_at(MAIN, level)[0]
            dev_q = (np.abs(reps["nominal"]["joint_pos"][:steps] - noisy["joint_pos"][:steps]) * mask).sum(0) / n / sd_at(MAIN, level)[1]
            per_level[level] = {"res": res, "dev_tau": dev_tau, "dev_q": dev_q}
        for e in range(log["alive"].shape[1]):
            failed = not bool(log["success"][e])
            episodes.append({
                "batch": bid, "suite": {"b": "test", "t": "train", "o": "other"}[bid[0]], "env": e,
                "setting": labels[bid]["setting"], "true": labels[bid]["true_class"],
                "failed": failed, "final_dist_cm": float(log["final_dist"][e] * 100),
                "clean_nominal": float(clean_nom[e]),
                "levels": {
                    level: {
                        "res": {sig: {c: float(r[c]["residual"][e]) for c in CANDIDATES} for sig, r in lv["res"].items()},
                        "terms": {c: (float(lv["res"][MAIN][c]["torque_term"][e]), float(lv["res"][MAIN][c]["pos_term"][e])) for c in CANDIDATES},
                        "dev_tau": lv["dev_tau"][e], "dev_q": lv["dev_q"][e],
                    }
                    for level, lv in per_level.items()
                },
            })
    kept_floor = np.concatenate(kept_floor)
    dev_thr = kept_floor.mean() + 4 * kept_floor.std() + 1e-12
    for ep in episodes:
        ep["deviating"] = ep["clean_nominal"] > dev_thr
        ep["kept"] = ep["failed"] or ep["deviating"]
    return episodes, floors, exact, joint_names, dev_thr


# methods: each returns a predicted label for an episode at a noise level


def sysid_fit(res: dict) -> tuple[str, float, float]:
    """Best axis, fitted multiplier and fitted residual from a parabola in log-multiplier space."""
    best = None
    for axis_name, axis in (("H_theta-mass", MASS_AXIS), ("H_theta-friction", FRIC_AXIS)):
        x = np.log([m for m, _ in axis])
        r = np.array([res[c] for _, c in axis])
        order = np.argsort(x)
        x, r = x[order], r[order]
        k = int(np.argmin(r))
        lo = min(max(k - 1, 0), len(x) - 3)
        xs, rs = x[lo: lo + 3], r[lo: lo + 3]
        a, b, c = np.polyfit(xs, rs, 2)
        if a > 0:
            x_star = float(np.clip(-b / (2 * a), x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]))
            r_star = max(float(np.polyval([a, b, c], x_star)), 0.0)
        else:
            x_star, r_star = float(x[k]), float(r[k])
        if r_star > r[k]:
            x_star, r_star = float(x[k]), float(r[k])
        if best is None or r_star < best[2]:
            best = (axis_name, float(np.exp(x_star)), r_star)
    return best


def predict_sysid(ep, level):
    cls, mult, _ = sysid_fit(ep["levels"][level]["res"][MAIN])
    return "none" if abs(np.log(mult)) < NONE_BAND else cls


def predict_ours(ep, level, floors, signal=MAIN):
    f = floors[(signal, level)]
    post = new_rule(ep["levels"][level]["res"][signal], ep["failed"], f["sigma2"], f["thr"])
    return max(post, key=post.get)


def predict_ours_other(ep, level, floors, tau_other, signal=MAIN):
    """tau_other is keyed (signal, level) when pooled, (signal, level, class) when class-conditional."""
    pred = predict_ours(ep, level, floors, signal)
    tau = tau_other.get((signal, level, pred), tau_other.get((signal, level)))
    return "H_other" if min(ep["levels"][level]["res"][signal].values()) > tau else pred


def predict_old(ep, level):
    post = old_rule(ep["levels"][level]["res"][MAIN], ep["failed"])
    return max(post, key=post.get)


def features(ep, level, floors):
    lv = ep["levels"][level]
    terms = np.log10(1e-6 + np.array([t for c in CANDIDATES for t in lv["terms"][c]]))
    floor = np.log10(1e-6 + floors[(MAIN, level)]["mean"])
    return np.concatenate([terms - floor, np.log1p(lv["dev_tau"]), np.log1p(lv["dev_q"]), [float(ep["failed"])]])


def train_supervised(train_eps, level, floors, n_models=5):
    import torch

    x = np.stack([features(ep, level, floors) for ep in train_eps]).astype(np.float32)
    y = np.array([CLASSES.index(ep["true"]) for ep in train_eps])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.from_numpy((x - mu) / sd), torch.from_numpy(y)
    weights = torch.tensor(len(y) / (len(CLASSES) * np.maximum(np.bincount(y, minlength=len(CLASSES)), 1)), dtype=torch.float32)
    models = []
    for seed in range(n_models):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, len(CLASSES)),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        for _ in range(400):
            opt.zero_grad()
            loss_fn(model(xt), yt).backward()
            opt.step()
        models.append(model.eval())

    def predict(ep):
        with torch.no_grad():
            z = torch.from_numpy(((features(ep, level, floors) - mu) / sd).astype(np.float32))[None]
            p = torch.stack([torch.softmax(m(z), dim=1) for m in models]).mean(0)[0]
        return CLASSES[int(p.argmax())]

    return predict


def threshold_scores(ep, level):
    lv = ep["levels"][level]
    return lv["dev_tau"][ARM_WRIST].mean(), lv["dev_tau"][FINGERS].mean() / max(lv["dev_tau"][JOINT2], 1e-6)


def threshold_predict(ep, level, floors, wrist_thr, finger_ratio):
    f = floors[(MAIN, level)]
    if ep["failed"] and ep["levels"][level]["res"][MAIN]["nominal"] <= f["thr"]:
        return "H_pi"
    wrist, ratio = threshold_scores(ep, level)
    if wrist > wrist_thr:
        return "H_nu"
    return "H_theta-friction" if ratio > finger_ratio else "H_theta-mass"


def fit_threshold(train_eps, level, floors):
    """Rules: nominal replay reproduces a failed episode -> H_pi; large wrist torque deviation -> H_nu;
    finger deviation large relative to joint2 -> friction; else mass. Constants: best class-balanced
    accuracy on the training logs."""
    wrists = np.array([threshold_scores(ep, level)[0] for ep in train_eps])
    ratios = np.array([threshold_scores(ep, level)[1] for ep in train_eps])
    best = (-1, None)
    for w in np.quantile(wrists, np.linspace(0.02, 0.98, 49)):
        for r in np.quantile(ratios, np.linspace(0.02, 0.98, 49)):
            preds = [threshold_predict(ep, level, floors, w, r) for ep in train_eps]
            acc = balanced_acc(train_eps, preds)
            if acc > best[0]:
                best = (acc, (w, r))
    return best[1]


def llm_predict(llm, ep, level, floors, joint_names) -> str:
    lv = ep["levels"][level]
    probs = llm.predict({
        "failed": ep["failed"], "final_dist_cm": ep["final_dist_cm"], "residual": lv["res"][MAIN],
        "floor": floors[(MAIN, level)]["mean"], "joint_names": joint_names, "dev_tau": lv["dev_tau"], "dev_q": lv["dev_q"],
    })
    return max(probs, key=probs.get)


# metrics


def balanced_acc(eps, preds) -> float:
    accs = [np.mean([p == ep["true"] for ep, p in zip(eps, preds) if ep["true"] == cls])
            for cls in CLASSES if any(ep["true"] == cls for ep in eps)]
    return float(np.mean(accs))


def per_class_acc(eps, preds) -> dict:
    return {cls: (np.mean([p == cls for ep, p in zip(eps, preds) if ep["true"] == cls])
                  if any(ep["true"] == cls for ep in eps) else np.nan) for cls in CLASSES}


def bootstrap(eps, preds, metric, rng) -> tuple[float, float, float]:
    """Point estimate and cluster-bootstrap 95% interval over batches."""
    batches = sorted({ep["batch"] for ep in eps})
    idx = {b: [i for i, ep in enumerate(eps) if ep["batch"] == b] for b in batches}
    point = metric(eps, preds)
    samples = []
    for _ in range(args.bootstrap):
        draw = [i for b in rng.choice(batches, len(batches)) for i in idx[b]]
        v = metric([eps[i] for i in draw], [preds[i] for i in draw])
        if not np.isnan(v):
            samples.append(v)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, lo, hi


def fmt_ci(t) -> str:
    p, lo, hi = t
    return "-" if np.isnan(p) else f"{p * 100:.0f} [{lo * 100:.0f}, {hi * 100:.0f}]"


def evaluate(noise_aware: bool, class_other: bool) -> tuple[list[str], list[str], dict]:
    """Returns (shared intro lines, this version's section lines, headline numbers)."""
    episodes, floors, exact, joint_names, dev_thr = build_features(noise_aware)
    kept = [ep for ep in episodes if ep["kept"]]
    test = [ep for ep in kept if ep["suite"] == "test"]
    train = [ep for ep in kept if ep["suite"] == "train"]
    other = [ep for ep in kept if ep["suite"] == "other"]
    levels = args.noise_levels
    rng = np.random.default_rng(0)

    def min_res(eps, sig, level):
        return [min(ep["levels"][level]["res"][sig].values()) for ep in eps]

    if class_other:
        tau_other = {
            (sig, level, cls): float(np.percentile(min_res([ep for ep in train if ep["true"] == cls], sig, level), args.other_quantile))
            for sig in SIGNALS for level in levels for cls in CLASSES
        }
    else:
        tau_other = {(sig, level): float(np.percentile(min_res(train, sig, level), args.other_quantile))
                     for sig in SIGNALS for level in levels}
    preds = {}  # (method, level) -> {episode index in kept: label}
    thresholds = {}
    llm = LLMAttributor()
    for level in levels:
        supervised = train_supervised(train, level, floors)
        thresholds[level] = fit_threshold(train, level, floors)
        methods = {
            "ours": lambda ep: predict_ours(ep, level, floors),
            "ours+other": lambda ep: predict_ours_other(ep, level, floors, tau_other),
            "old rule (v1)": lambda ep: predict_old(ep, level),
            "SysID-only": lambda ep: predict_sysid(ep, level),
            "supervised MLP": supervised,
            "threshold": lambda ep: threshold_predict(ep, level, floors, *thresholds[level]),
        }
        for sig in SIGNALS:
            methods[f"ours [{sig}]"] = lambda ep, sig=sig: predict_ours(ep, level, floors, sig)
            methods[f"ours+other [{sig}]"] = lambda ep, sig=sig: predict_ours_other(ep, level, floors, tau_other, sig)
        if llm.available:
            methods["LLM-only"] = lambda ep: llm_predict(llm, ep, level, floors, joint_names)
        for name, fn in methods.items():
            preds[(name, level)] = {id(ep): fn(ep) for ep in test + other}

    def P(name, level, eps):
        return [preds[(name, level)][id(ep)] for ep in eps]

    main_methods = ["ours", "ours+other", "old rule (v1)", "SysID-only", "supervised MLP", "threshold"]
    intro = [
        "Test logs are the attribution_v2 settings (causes off the candidate grid, seeds 0-2, one log per episode),",
        "re-recorded with PhysX joint forces. Training logs: seeds 3-5, mass x1.75/3.5/6, friction x0.55/0.35/0.15,",
        "model_100/700, arm-only explicit actuators. Implicit actuators throughout unless stated; `applied_torque` is",
        f"Isaac Lab's PD estimate. Sensor noise x1 = q {args.q_noise:g} rad, tau {args.tau_noise:g} N·m (N on the fingers),",
        "applied to the logs' q and both torque signals. Accuracy is on kept logs; brackets are 95% cluster-bootstrap",
        f"intervals over batches ({args.bootstrap} resamples). Methods are described in `scripts/attribution_eval.py`.",
        "",
        "Kept logs (failed, or noise-free nominal residual above the x1 floor threshold "
        f"{dev_thr:.3g}):", "",
        "| suite | class | kept / total |", "|---|---|---|",
    ]
    for suite in ("test", "train", "other"):
        for cls in CLASSES + ["H_other"]:
            all_eps = [ep for ep in episodes if ep["suite"] == suite and ep["true"] == cls]
            if all_eps:
                intro.append(f"| {suite} | {cls} | {sum(ep['kept'] for ep in all_eps)} / {len(all_eps)} |")
    intro += ["", "Reproducibility in this run (reference closed loop vs its two nominal replays, max |diff| of q and both torques): "
              + ", ".join(f"{k} {v:.3g}" for k, v in exact.items()), ""]
    lines = []

    # 1. main comparison
    lines += ["### 1. Methods (test logs, 4 classes), class-balanced accuracy %", "",
              "| method | " + " | ".join(f"noise x{lv:g}" for lv in levels) + " |", "|---|" + "---|" * len(levels)]
    for name in main_methods:
        lines.append(f"| {name} | " + " | ".join(fmt_ci(bootstrap(test, P(name, lv, test), balanced_acc, rng)) for lv in levels) + " |")
    lines.append("| LLM-only | " + " | ".join("not run (interface in `llm_attribution.py`)" if i == 0 else "" for i in range(len(levels))) + " |")
    for level in sorted({1.0} & set(levels)):
        lines += ["", f"Per class, noise x{level:g}:", "", "| method | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for name in main_methods:
            p = P(name, level, test)
            cells = [fmt_ci(bootstrap(test, p, lambda e, q, c=cls: per_class_acc(e, q)[c], rng)) for cls in CLASSES]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", f"Threshold constants (fitted on training logs): wrist |dtau| > {thresholds[level][0]:.3g} sd -> H_nu,"
                  f" finger/joint2 deviation ratio > {thresholds[level][1]:.3g} -> friction.", ""]
        lines += ["How SysID-only labels the policy and engine logs (noise x1):", "",
                  "| true | n | " + " | ".join(["H_theta-mass", "H_theta-friction", "none"]) + " |", "|---|---|---|---|---|"]
        for cls in ("H_pi", "H_nu"):
            eps = [ep for ep in test if ep["true"] == cls]
            p = P("SysID-only", level, eps)
            lines.append(f"| {cls} | {len(eps)} | " + " | ".join(f"{p.count(k)}" for k in ["H_theta-mass", "H_theta-friction", "none"]) + " |")
        fits = [sysid_fit(ep["levels"][level]["res"][MAIN]) for ep in test if ep["true"] == "H_nu"]
        lines += ["", f"(engine logs: median fitted multiplier {np.median([f[1] for f in fits]):.2f}, median fitted residual"
                  f" {np.median([f[2] for f in fits]):.3g} vs noise floor {floors[(MAIN, level)]['mean']:.3g})"]

    # 2. out-of-hypothesis faults
    out_cols = CLASSES + ["H_other"]
    lines += ["", "### 2. Faults outside the hypothesis space (H_other)", "",
              f"Criterion: H_other if every candidate's residual exceeds tau_other = the {args.other_quantile:g}th percentile of"
              " the minimum candidate residual over the kept training logs"
              + (" of the class ours predicts." if class_other else " (all classes pooled)."), ""]
    if class_other:
        lines += ["| noise | " + " | ".join(f"tau_other {c}" for c in CLASSES) + " | floor mean |", "|---|" + "---|" * (len(CLASSES) + 1)]
        lines += [f"| x{lv:g} | " + " | ".join(f"{tau_other[(MAIN, lv, c)]:.3g}" for c in CLASSES)
                  + f" | {floors[(MAIN, lv)]['mean']:.3g} |" for lv in levels]
    else:
        lines += ["| noise | tau_other | floor mean |", "|---|---|---|"]
        lines += [f"| x{lv:g} | {tau_other[(MAIN, lv)]:.3g} | {floors[(MAIN, lv)]['mean']:.3g} |" for lv in levels]
    for level in sorted({0.0, 1.0} & set(levels)):
        lines += ["", f"Noise x{level:g}: predictions for the H_other logs with the criterion (without it in brackets):", "",
                  "| setting | kept / total | " + " | ".join(out_cols) + " |", "|---|---|" + "---|" * len(out_cols)]
        for setting in sorted({ep["setting"] for ep in episodes if ep["suite"] == "other"}):
            all_eps = [ep for ep in episodes if ep["setting"] == setting]
            eps = [ep for ep in other if ep["setting"] == setting]
            with_c, without = P("ours+other", level, eps), P("ours", level, eps)
            lines.append(f"| {setting} | {len(eps)} / {len(all_eps)} | "
                         + " | ".join(f"{with_c.count(k)} ({without.count(k)})" for k in out_cols) + " |")
    lines += ["", "H_other detection rate on the H_other logs and accuracy on the 4-class test logs:", "",
              "| noise | H_other logs -> H_other | 4-class balanced acc, ours | with criterion | drop (points) |",
              "|---|---|---|---|---|"]
    for level in levels:
        det = bootstrap(other, P("ours+other", level, other), lambda e, q: np.mean([x == "H_other" for x in q]), rng)
        a = bootstrap(test, P("ours", level, test), balanced_acc, rng)
        b = bootstrap(test, P("ours+other", level, test), balanced_acc, rng)
        false_other = np.mean([x == "H_other" for x in P("ours+other", level, test)])
        lines.append(f"| x{level:g} | {fmt_ci(det)} | {fmt_ci(a)} | {fmt_ci(b)} | {(a[0] - b[0]) * 100:.1f}"
                     f" ({false_other * 100:.0f}% of test logs called H_other) |")

    # 3. torque signals
    lines += ["", "### 3. Residual signal (ours), class-balanced accuracy %", "",
              "`tau_solver` = PhysX `get_dof_projected_joint_forces` (the solver's joint force along each DOF);"
              " `get_dof_actuation_forces` reads all zeros for implicit drives.", "",
              "| signal | " + " | ".join(f"noise x{lv:g}" for lv in levels) + " |", "|---|" + "---|" * len(levels)]
    for sig in SIGNALS:
        lines.append(f"| {sig} | " + " | ".join(fmt_ci(bootstrap(test, P(f'ours [{sig}]', lv, test), balanced_acc, rng)) for lv in levels) + " |")
    lines += ["", "With the H_other criterion, H_other detection rate / 4-class balanced accuracy:", "",
              "| signal | " + " | ".join(f"noise x{lv:g}" for lv in levels) + " |", "|---|" + "---|" * len(levels)]
    for sig in SIGNALS:
        cells = []
        for lv in levels:
            det = np.mean([x == "H_other" for x in P(f"ours+other [{sig}]", lv, other)])
            cells.append(f"{det * 100:.0f} / {balanced_acc(test, P(f'ours+other [{sig}]', lv, test)) * 100:.0f}")
        lines.append(f"| {sig} | " + " | ".join(cells) + " |")
    if 1.0 in levels:
        lines += ["", "Per class, noise x1:", "", "| signal | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for sig in SIGNALS:
            pc = per_class_acc(test, P(f"ours [{sig}]", 1.0, test))
            lines.append(f"| {sig} | " + " | ".join(f"{pc[c] * 100:.0f}" for c in CLASSES) + " |")

    headline = {}
    for lv in levels:
        headline[("ours", lv)] = balanced_acc(test, P("ours", lv, test))
        headline[("detect", lv)] = np.mean([x == "H_other" for x in P("ours+other", lv, other)])
        headline[("ours+other", lv)] = balanced_acc(test, P("ours+other", lv, test))
        for sig in SIGNALS:
            headline[(sig, lv)] = balanced_acc(test, P(f"ours [{sig}]", lv, test))
            headline[(f"detect {sig}", lv)] = np.mean([x == "H_other" for x in P(f"ours+other [{sig}]", lv, other)])
    tag = ("B" if noise_aware else "A")
    with open(os.path.join(args.dir, f"predictions_{tag}.csv"), "w") as f:
        names = sorted({name for name, _ in preds})
        f.write("noise,batch,env,suite,setting,true_class,failed,kept," + ",".join(names) + "\n")
        for level in levels:
            for ep in test + other:
                f.write(f"{level},{ep['batch']},{ep['env']},{ep['suite']},{ep['setting']},{ep['true']},{ep['failed']},True,"
                        + ",".join(preds[(n, level)][id(ep)] for n in names) + "\n")
    return intro, lines, headline


def main():
    title = ["# Inverse failure attribution v3: baselines, out-of-hypothesis faults, torque signals", ""]
    intro, lines_a, head_a = evaluate(noise_aware=False, class_other=False)
    _, lines_b, head_b = evaluate(noise_aware=True, class_other=True)
    levels = args.noise_levels
    summary = [
        "## Summary: version A (as first run) vs B (fixed)", "",
        "A: per-joint sd normalisation, pooled tau_other. B: sqrt(sd^2 + sigma_noise^2) normalisation, tau_other conditional"
        " on the predicted class. Class-balanced accuracy on the 4-class test logs and H_other detection rate, %.", "",
        "| metric | version | " + " | ".join(f"x{lv:g}" for lv in levels) + " |", "|---|---|" + "---|" * len(levels),
    ]
    rows = [("ours, q+applied", "ours"), ("ours+other, 4-class", "ours+other"), ("H_other detected", "detect"),
            ("ours, q only", "q"), ("ours, q+solver", "q+solver"), ("H_other detected, q+solver", "detect q+solver")]
    for label, key in rows:
        for tag, head in (("A", head_a), ("B", head_b)):
            summary.append(f"| {label} | {tag} | " + " | ".join(f"{head[(key, lv)] * 100:.0f}" for lv in levels) + " |")
    text = "\n".join(title + intro + summary + ["", "## Version A: as first run", ""] + lines_a
                     + ["", "## Version B: noise-aware normalisation, class-conditional H_other threshold", ""] + lines_b) + "\n"
    with open(args.md, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
