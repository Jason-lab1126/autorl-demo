"""Attribution v5: a calibrated posterior (C1), a parameter-vector H_theta, and LLM-proposed candidates.

Re-analysis of the results/attr/v3 rollouts (test, train, other, gen suites); no new closed-loop logs.

1. Likelihood. v4 used log L_c = -r_c / (2 sigma^2) with r_c the per-step mean residual and sigma^2 the
   replay noise floor. v5 sums the residual over time steps and corrects for temporal correlation with an
   effective sample size n_eff = n (1 - rho) / (1 + rho), rho the lag-1 autocorrelation of the per-step
   residual, one n_eff per episode (median over its candidates, so a badly matched candidate's strongly
   correlated residual does not buy it a smaller penalty): log L_c = -n_eff r_c / (2 T). One temperature T
   per noise level is fitted on the training logs only (minimum NLL of the true class), and only on those the
   hypothesis space can explain (best candidate of the true class within 10x the x1 noise floor); training
   logs no candidate matches would otherwise force T up and flatten every posterior. Ablations reported.
2. H_theta as a parameter vector (cube mass, friction) on [0.05, 20]^2 in log scale. The replays only
   cover the two axes (nominal, mass x2/3/5, friction x0.5/0.3/0.2), so the surrogate residual is
   separable: r(m, f) = r_mass(m) + r_fric(f) - r_nominal, each axis piecewise-linear in log-multiplier
   and growing linearly away from the data outside it. Posterior over the grid with the v5 likelihood;
   reported: MAP, 95% marginal intervals, which parameters deviate from nominal (interval excludes 1).
3. LLM-proposed candidates (llm_attribution.propose, claude-sonnet-5-5, 3 samples per log) for logs
   called H_other or whose residuals all exceed the class-conditional H_other threshold; the proposals
   are replayed (attribute.py --jobs_file) and the posterior recomputed with them. Needs the anthropic
   SDK and ANTHROPIC_API_KEY; otherwise the stage is reported as not run. An oracle run, clearly
   labelled, inserts the true solver configuration instead, to check the pipeline and bound the result.

Usage (from ~/autorl-demo after ``source env.sh``):
    python scripts/attribution_v5.py [--llm] [--oracle] [--workers 8]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attribution_eval as ev  # noqa: E402
import attribution_v4 as v4  # noqa: E402
import llm_attribution as llm  # noqa: E402
from attr_common import CANDIDATES, CLASSES, logsumexp  # noqa: E402
from traj_metrics import add_sensor_noise, joint_sd, residual_per_env  # noqa: E402

parser = argparse.ArgumentParser(description="Attribution v5.")
parser.add_argument("--llm", action="store_true", help="Run the LLM proposer stage (needs ANTHROPIC_API_KEY).")
parser.add_argument("--oracle", action="store_true", help="Run the oracle check of the proposer pipeline.")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--samples", type=int, default=3, help="Proposer samples per log.")
args5, _ = parser.parse_known_args()
args = ev.args
LEVELS = args.noise_levels
MAIN = ev.MAIN
GRID = np.linspace(np.log(0.05), np.log(20.0), 241)
LLM_DIR = os.path.join(args.dir, "replay_llm")
RAW_LOG = os.path.join(os.path.dirname(args.md), "v5_llm_raw.jsonl")


# effective sample sizes


def neff_table(episodes) -> dict:
    """(batch, env, level, candidate) -> (n, n_eff) for the main signal, same noise draws as v4.build('v3')."""
    q_std, tau_std, tj = v4.NOISE_MODELS["v3"]
    refs = [v4.load(f"logs/ref_s{s}.npz") for s in args.seeds]
    pooled = {k: np.concatenate([r[k] for r in refs], axis=1) for k in ("joint_pos", "alive", "applied_torque")}
    sd_tau0, sd_q0 = joint_sd(pooled, "applied_torque")
    out = {}
    for bid in sorted({ep["batch"] for ep in episodes}):
        log = v4.load(f"logs/{bid}.npz")
        reps = {c: v4.load(f"replay/{bid}__{c}.npz") for c in CANDIDATES}
        for level in LEVELS:
            noisy = add_sensor_noise(log, level * q_std, level * tau_std, np.random.default_rng([int(level * 10), int(bid[1:]), ord(bid[0])]), ev.TORQUE_KEYS)
            sd_tau = np.sqrt(sd_tau0**2 + (level * tau_std) ** 2)
            sd_q = np.sqrt(sd_q0**2 + (level * q_std) ** 2)
            for c, rep in reps.items():
                for e, (n, ne) in enumerate(series_neff(noisy, rep, sd_tau, sd_q)):
                    out[(bid, e, level, c)] = (n, ne)
    return out


def series_neff(log, rep, sd_tau, sd_q):
    steps = min(len(log["alive"]), len(rep["alive"]))
    mask = log["alive"][:steps] & rep["alive"][:steps]
    s = (((rep["applied_torque"][:steps] - log["applied_torque"][:steps]) / sd_tau) ** 2).mean(axis=2) \
        + (((rep["joint_pos"][:steps] - log["joint_pos"][:steps]) / sd_q) ** 2).mean(axis=2)
    res = []
    for e in range(mask.shape[1]):
        x = s[mask[:, e], e]
        n = len(x)
        if n < 3 or x.std() == 0:
            res.append((max(n, 1), max(n, 1)))
            continue
        xc = x - x.mean()
        rho = float(np.clip((xc[1:] * xc[:-1]).sum() / (xc**2).sum(), -0.99, 0.99))
        res.append((n, float(np.clip(n * (1 - rho) / (1 + rho), 1.0, n))))
    return res


# posteriors


def class_posterior(log_l: dict, cand_class: dict, failed: bool, r_nom: float, thr: float) -> dict:
    """Mean likelihood within each class (uniform prior within a class), normalised over classes. H_pi takes
    the nominal candidate only for a failed episode whose nominal residual is inside the noise floor."""
    classes = sorted({cand_class[c] for c in log_l} | set(CLASSES))
    ll = {}
    for cls in classes:
        if cls == "H_pi":
            ll[cls] = log_l["nominal"] if failed and r_nom <= thr else -np.inf
        else:
            members = np.array([log_l[c] for c in log_l if cand_class[c] == cls])
            ll[cls] = logsumexp(members) - np.log(len(members)) if len(members) else -np.inf
    z = logsumexp(np.array(list(ll.values())))
    return {cls: float(np.exp(v - z)) for cls, v in ll.items()}


CAND_CLASS = {c: spec[3] for c, spec in CANDIDATES.items()}


def loglik(ep, level, variant, T, neff, extra=None):
    res = dict(ep["levels"][level]["res"][MAIN])
    if extra:
        res.update(extra)
    out = {}
    for c, r in res.items():
        if variant == "sum":
            ne = neff.get((ep["batch"], ep["env"], level, c), neff.get((ep["batch"], ep["env"], level, "nominal")))[1]
            out[c] = -ne * r / (2 * T)
        else:
            out[c] = -r / (2 * T)
    return out, res


def posterior(ep, level, variant, T, neff, floors, extra=None, extra_class=None):
    log_l, res = loglik(ep, level, variant, T, neff, extra)
    f = floors[(MAIN, level)]
    return class_posterior(log_l, {**CAND_CLASS, **(extra_class or {})}, ep["failed"], res["nominal"], f["thr"])


def common_neff(neff: dict) -> dict:
    """One n_eff per (episode, level): the median over its candidates, used for every candidate."""
    per = {}
    for (b, e, lv, c), (n, ne) in neff.items():
        per.setdefault((b, e, lv), []).append(ne)
    return {(b, e, lv, c): (neff[(b, e, lv, c)][0], float(np.median(per[(b, e, lv)]))) for (b, e, lv, c) in neff}


def in_hypothesis(ep, level, floors) -> bool:
    """Whether the true class has a candidate within 10x the noise floor (x1 floor when the level's is 0)."""
    res = ep["levels"][level]["res"][MAIN]
    best = min(r for c, r in res.items() if CANDIDATES[c][3] == ep["true"])
    return best <= 10 * max(floors[(MAIN, level)]["mean"], floors[(MAIN, 1.0)]["mean"])


def fit_temperature(train, level, variant, neff, floors) -> float:
    best = None
    for logT in np.linspace(-12, 8, 201):
        T = float(np.exp(logT))
        nll = 0.0
        for ep in train:
            p = posterior(ep, level, variant, T, neff, floors).get(ep["true"], 0.0)
            nll -= np.log(max(p, 1e-12))
        if best is None or nll < best[0]:
            best = (nll, T)
    return best[1]


# parameter vector


def axis_surrogate(res, axis):
    """Piecewise-linear residual in log-multiplier over the replayed axis points; outside them the residual
    grows linearly away from the data with the edge segment's absolute slope."""
    x = np.log([m for m, _ in axis])
    r = np.array([res[c] for _, c in axis])
    o = np.argsort(x)
    x, r = x[o], r[o]
    inside = np.interp(GRID, x, r)
    lo_slope = abs((r[1] - r[0]) / (x[1] - x[0]))
    hi_slope = abs((r[-1] - r[-2]) / (x[-1] - x[-2]))
    out = np.where(GRID < x[0], r[0] + lo_slope * (x[0] - GRID), inside)
    return np.where(GRID > x[-1], r[-1] + hi_slope * (GRID - x[-1]), out)


def theta_posterior(ep, level, T, neff):
    res = ep["levels"][level]["res"][MAIN]
    rm, rf = axis_surrogate(res, ev.MASS_AXIS), axis_surrogate(res, ev.FRIC_AXIS)
    r2 = np.maximum(rm[:, None] + rf[None, :] - res["nominal"], 0.0)
    ne = np.median([neff[(ep["batch"], ep["env"], level, c)][1] for c in CANDIDATES if c != "numerics"])
    ll = -ne * r2 / (2 * T)
    p = np.exp(ll - ll.max())
    p /= p.sum()
    i, j = np.unravel_index(np.argmax(p), p.shape)

    def interval(marg):
        cdf = np.cumsum(marg)
        return float(np.exp(GRID[np.searchsorted(cdf, 0.025)])), float(np.exp(GRID[min(np.searchsorted(cdf, 0.975), len(GRID) - 1)]))

    def median(marg):
        return float(np.exp(GRID[np.searchsorted(np.cumsum(marg), 0.5)]))

    # point estimate: marginal posterior median (the piecewise-linear surrogate puts the MAP on a grid knot)
    return {"mass": median(p.sum(axis=1)), "fric": median(p.sum(axis=0)),
            "mass_map": float(np.exp(GRID[i])), "fric_map": float(np.exp(GRID[j])),
            "mass_ci": interval(p.sum(axis=1)), "fric_ci": interval(p.sum(axis=0))}


# LLM proposer stage


def proposer_summary(ep, level, floors, joint_names) -> dict:
    """What the proposer sees: outcome, residuals of the existing candidates, where (joints, phases) the
    best candidate still misses, and the simulator configuration. Never the setting or its class."""
    lv = ep["levels"][level]
    res = lv["res"][MAIN]
    best = min(res, key=res.get)
    log = v4.load(f"logs/{ep['batch']}.npz")
    rep = v4.load(f"replay/{ep['batch']}__{best}.npz")
    e = ep["env"]
    steps = min(len(log["alive"]), len(rep["alive"]))
    m = log["alive"][:steps, e] & rep["alive"][:steps, e]
    sd_tau, sd_q = joint_sd(log, "applied_torque")
    dtau = np.abs(rep["applied_torque"][:steps, e] - log["applied_torque"][:steps, e]) / sd_tau
    dq = np.abs(rep["joint_pos"][:steps, e] - log["joint_pos"][:steps, e]) / sd_q
    close = np.nonzero(log["action"][:steps, e, 7] < 0)[0]
    t_close = int(close[0]) if len(close) else steps
    z = log["cube_pos"][:steps, e, 2]
    lifted = np.nonzero(z > z[0] + 0.02)[0]
    t_lift = int(lifted[0]) if len(lifted) else steps
    phases = {"approach": (0, t_close), "grasp_and_lift": (t_close, max(t_lift, t_close)), "transport_and_hold": (max(t_lift, t_close), steps)}
    per_phase = {}
    for name, (a, b) in phases.items():
        idx = np.arange(a, b)
        idx = idx[m[idx]] if len(idx) else idx
        per_phase[name] = {"steps": f"{a}-{b}", "torque_dev_sd": round(float(dtau[idx].mean()), 3) if len(idx) else None,
                           "position_dev_sd": round(float(dq[idx].mean()), 3) if len(idx) else None}
    joint_dev = {n: {"torque_dev_sd": round(float(dtau[m, j].mean()), 3), "position_dev_sd": round(float(dq[m, j].mean()), 3)}
                 for j, n in enumerate(joint_names)}
    return {
        "task_outcome": {"success": not ep["failed"], "final_cube_to_goal_cm": round(ep["final_dist_cm"], 2)},
        "noise_floor_residual": round(floors[(MAIN, level)]["mean"], 4),
        "existing_candidates": {c: {"change": d, "residual": round(res[c], 4)} for c, d in CANDIDATE_TEXT.items()},
        "best_existing_candidate": best, "best_residual_over_noise_floor": round(res[best] / floors[(MAIN, level)]["mean"], 1),
        "deviation_from_best_candidate_by_joint": joint_dev,
        "deviation_from_best_candidate_by_phase": per_phase,
        "simulator_configuration": llm.SIM_CONFIG,
    }


CANDIDATE_TEXT = {
    "nominal": "no change", "mass_x2": "cube mass x2", "mass_x3": "cube mass x3", "mass_x5": "cube mass x5",
    "fric_x0.5": "friction x0.5", "fric_x0.3": "friction x0.3", "fric_x0.2": "friction x0.2",
    "numerics": "actuator_model explicit_all (IdealPD torques on all joints)",
}


def run_jobs(jobs, name):
    if not jobs:
        return
    path = os.path.join(LLM_DIR, f"jobs_{name}.json")
    os.makedirs(LLM_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(jobs, f, indent=1)
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "attribute.py"), "--phase", "run",
           "--jobs_file", path, "--out_dir", args.dir, "--job_log_dir", "logs/attr_v5", "--workers", str(args5.workers),
           "--headless", "--device", "cuda:0"]
    subprocess.run(cmd, check=True)


def replay_job(bid, key, cfg):
    return {"kind": "replay_cfg", "cfg": cfg, "log_path": os.path.join(args.dir, "logs", f"{bid}.npz"),
            "out": os.path.join(LLM_DIR, f"{bid}__{key}.npz")}


def extra_residuals(ep, level, keys) -> dict:
    """Residuals of extra replays for this episode (same noise model and normalisation as v4.build('v3'))."""
    q_std, tau_std, _ = v4.NOISE_MODELS["v3"]
    refs = [v4.load(f"logs/ref_s{s}.npz") for s in args.seeds]
    pooled = {k: np.concatenate([r[k] for r in refs], axis=1) for k in ("joint_pos", "alive", "applied_torque")}
    sd_tau0, sd_q0 = joint_sd(pooled, "applied_torque")
    sd_tau, sd_q = np.sqrt(sd_tau0**2 + (level * tau_std) ** 2), np.sqrt(sd_q0**2 + (level * q_std) ** 2)
    log = v4.load(f"logs/{ep['batch']}.npz")
    noisy = add_sensor_noise(log, level * q_std, level * tau_std, np.random.default_rng([int(level * 10), int(ep["batch"][1:]), ord(ep["batch"][0])]), ev.TORQUE_KEYS)
    out = {}
    for key in keys:
        path = os.path.join(LLM_DIR, f"{ep['batch']}__{key}.npz")
        if os.path.isfile(path):
            rep = dict(np.load(path))
            out[key] = float(residual_per_env(noisy, rep, sd_tau, sd_q, torque_key="applied_torque")["residual"][ep["env"]])
    return out


# main


def main():
    episodes, floors = v4.build("v3")
    kept = [ep for ep in episodes if ep["kept"]]
    train = [ep for ep in kept if ep["suite"] == "train"]
    test = [ep for ep in kept if ep["suite"] == "test"]
    gen = [ep for ep in kept if ep["suite"] == "gen"]
    other = [ep for ep in kept if ep["suite"] == "other"]
    neff = neff_table(kept)
    L = ["# Inverse failure attribution v5: calibrated posterior, parameter-vector H_theta, LLM-proposed candidates", ""]
    L += ["Re-analysis of the attribution_v3/v4 rollouts (no new closed-loop logs). Sensor-noise model as in v3/v4 sections",
          "1-3 (q 1e-3 rad, tau 0.5 at x1, all joints), noise-aware normalisation. Implicit actuators; `applied_torque` is",
          "the PD estimate. Kept logs as before. Method details: `scripts/attribution_v5.py`.", ""]

    # 1. likelihood and calibration
    cneff = common_neff(neff)
    variants = {  # name -> (likelihood, n_eff table, training subset for T)
        "v4: -r/(2 sigma_floor^2)": ("mean", None, None),
        "mean residual, T fitted on all training logs": ("mean", neff, "all"),
        "mean residual, T fitted on in-hypothesis training logs": ("mean", neff, "in"),
        "summed, per-candidate n_eff, T in-hypothesis": ("sum", neff, "in"),
        "summed, common n_eff, T in-hypothesis (v5)": ("sum", cneff, "in"),
    }
    FINAL = "summed, common n_eff, T in-hypothesis (v5)"
    temps, rel, rows, excluded = {}, {}, {}, {}
    for name, (var, table, subset) in variants.items():
        for level in LEVELS:
            if subset is None:
                T = floors[(MAIN, level)]["sigma2"]
            else:
                tr = train if subset == "all" else [ep for ep in train if in_hypothesis(ep, level, floors)]
                if subset == "in":
                    excluded[level] = sorted({ep["setting"] for ep in train} - set()), len(train) - len(tr), \
                        {st: sum(1 for ep in train if ep["setting"] == st and not in_hypothesis(ep, level, floors)) for st in sorted({ep["setting"] for ep in train})}
                T = fit_temperature(tr, level, var, table, floors)
            temps[(name, level)] = T
            ps = [posterior(ep, level, var, T, table or neff, floors) for ep in test]
            pred = [max(p, key=p.get) for p in ps]
            conf = [max(p.values()) for p in ps]
            corr = [a == ep["true"] for a, ep in zip(pred, test)]
            e, rr = v4.ece(conf, corr)
            rel[(name, level)] = rr
            rows[(name, level)] = (ev.balanced_acc(test, pred), e, np.mean(conf), np.mean(corr))
    mlp_rows = {}
    for level in LEVELS:
        mlp = ev.train_supervised(train, level, floors)
        ps = [dict(zip(CLASSES, mlp.proba(ep))) for ep in test]
        pred = [max(p, key=p.get) for p in ps]
        conf = [max(p.values()) for p in ps]
        corr = [a == ep["true"] for a, ep in zip(pred, test)]
        e, rr = v4.ece(conf, corr)
        rel[("MLP (replay)", level)] = rr
        mlp_rows[level] = (ev.balanced_acc(test, pred), e, np.mean(conf), np.mean(corr))
    L += ["## 1. Calibrated likelihood (test logs)", "",
          "T is fitted per noise level on training logs only. n_eff from the lag-1 autocorrelation of the per-step residual",
          f"(median n_eff / n over test logs at x1: {np.median([cneff[(ep['batch'], ep['env'], 1.0, 'nominal')][1] / neff[(ep['batch'], ep['env'], 1.0, 'nominal')][0] for ep in test]):.3f}).",
          "In-hypothesis training logs: the true class has a candidate within 10x the x1 noise floor. Excluded at x1: "
          + ", ".join(f"{k} {v}" for k, v in excluded[1.0][2].items() if v) + f" ({excluded[1.0][1]} of {len(train)}).", "",
          "| likelihood | noise | balanced acc % | ECE % | mean confidence % | accuracy % | T |", "|---|---|---|---|---|---|---|"]
    for name in variants:
        for level in LEVELS:
            a, e, c, acc = rows[(name, level)]
            L.append(f"| {name} | x{level:g} | {a * 100:.0f} | {e * 100:.1f} | {c * 100:.0f} | {acc * 100:.0f} | {temps[(name, level)]:.3g} |")
    for level in LEVELS:
        a, e, c, acc = mlp_rows[level]
        L.append(f"| MLP (replay), for reference | x{level:g} | {a * 100:.0f} | {e * 100:.1f} | {c * 100:.0f} | {acc * 100:.0f} | - |")
    L += ["", "Reliability diagram: `reliability_v5.png` (v4 likelihood, v5 likelihood, replay MLP).", ""]
    plot_reliability(rel, ["v4: -r/(2 sigma_floor^2)", FINAL, "MLP (replay)"])
    neff = cneff  # the final likelihood uses the common n_eff from here on

    # 2. parameter vector
    lvl = 1.0
    T = temps[(FINAL, lvl)]
    L += ["## 2. H_theta as a parameter vector (cube mass, friction), noise x1", "",
          "Search range [0.05, 20] per parameter, log scale. Replays exist only on the two axes, so the surrogate is separable",
          "and the mass-friction interaction is not identifiable from them; values beyond the replayed range (mass > 5,",
          "friction < 0.2, either < or > nominal on the other side) are extrapolated. A parameter 'deviates' when its 95%",
          "marginal interval excludes 1.", "",
          "Point estimate = marginal posterior median (the surrogate is piecewise linear, so its MAP always sits on a replayed",
          "grid value; the MAP is shown for comparison).", "",
          "| true setting | n | estimate of the true parameter: median over logs (median abs rel. error) | MAP: median (error) | true value in 95% interval | other parameter flagged as deviating | median 95% interval |",
          "|---|---|---|---|---|---|---|"]
    settings = [(s, "mass", v) for s, v in v4.MASS_TRUE.items() if not s.startswith("op_")] + \
               [(s, "fric", v) for s, v in v4.FRIC_TRUE.items() if not s.startswith("op_")]
    for setting, param, true in settings:
        eps = [ep for ep in test if ep["setting"] == setting]
        if not eps:
            continue
        tp = [theta_posterior(ep, lvl, T, neff) for ep in eps]
        est = np.array([t[param] for t in tp])
        cov = np.mean([t[f"{param}_ci"][0] <= true <= t[f"{param}_ci"][1] for t in tp])
        other_p = "fric" if param == "mass" else "mass"
        false_dev = np.mean([not (t[f"{other_p}_ci"][0] <= 1.0 <= t[f"{other_p}_ci"][1]) for t in tp])
        lo, hi = np.median([t[f"{param}_ci"][0] for t in tp]), np.median([t[f"{param}_ci"][1] for t in tp])
        mp = np.array([t[f"{param}_map"] for t in tp])
        L.append(f"| {setting} | {len(eps)} | {np.median(est):.2f} ({np.median(np.abs(est / true - 1)) * 100:.0f}%) |"
                 f" {np.median(mp):.2f} ({np.median(np.abs(mp / true - 1)) * 100:.0f}%) | {cov * 100:.0f}% |"
                 f" {false_dev * 100:.0f}% | [{lo:.2f}, {hi:.2f}] |")
    comp = [ep for ep in gen if ep["setting"].startswith("compound")]
    if comp:
        tp = [theta_posterior(ep, lvl, T, neff) for ep in comp]
        both = np.mean([not (t["mass_ci"][0] <= 1 <= t["mass_ci"][1]) and not (t["fric_ci"][0] <= 1 <= t["fric_ci"][1]) for t in tp])
        L += ["", f"Compound fault (mass x2.5 + friction x0.4, {len(comp)} logs): median estimate mass x{np.median([t['mass'] for t in tp]):.2f},"
              f" friction x{np.median([t['fric'] for t in tp]):.2f}; both parameters flagged as deviating in {both * 100:.0f}% of logs;"
              f" true mass in its interval {np.mean([t['mass_ci'][0] <= 2.5 <= t['mass_ci'][1] for t in tp]) * 100:.0f}%, true friction"
              f" {np.mean([t['fric_ci'][0] <= 0.4 <= t['fric_ci'][1] for t in tp]) * 100:.0f}%."]
    L += ["", "Compared with attribution_v4's per-axis parabola fit (mass x2.5 / x4 within 10-12%, friction x0.4 within 3%,",
          "compound 2.54 / 0.38), the point estimates here are not better: with replays only on the two axes, re-analysis",
          "cannot locate the minimum between grid points or capture the mass-friction interaction. What the posterior adds",
          "is an interval with stated coverage. Better point estimates need a few adaptive replays near the current",
          "estimate (e.g. 2-3 rounds of one off-axis replay per log batch).", "",
          "Friction x0.25: residual profiles along the friction axis are flat while the cube does not slip (0.5, 0.3 and",
          "0.2 all sit near the noise floor) and jump when it does (0.2 often 100x the floor), so friction is only",
          "identifiable from episodes where slip occurs; the intervals above show how wide that leaves it.", ""]

    # 3. LLM-proposed candidates
    L += ["## 3. LLM-proposed candidates", ""]
    tau_other = {(level, cls): float(np.percentile([min(ep["levels"][level]["res"][MAIN].values()) for ep in train if ep["true"] == cls], args.other_quantile))
                 for level in LEVELS for cls in CLASSES}
    T1 = temps[(FINAL, lvl)]

    def base_pred(ep):
        p = posterior(ep, lvl, "sum", T1, neff, floors)
        pred = max(p, key=p.get)
        if min(ep["levels"][lvl]["res"][MAIN].values()) > tau_other.get((lvl, pred), np.inf):
            return "H_other"
        return pred

    flagged = [ep for ep in gen + other + test if base_pred(ep) == "H_other"]
    solver = [ep for ep in gen if ep["setting"] == "nu_solver_iters_x0.25"]
    flagged_settings = [ep["setting"] for ep in flagged]
    flagged_counts = ", ".join("%s %d" % (s, flagged_settings.count(s)) for s in sorted(set(flagged_settings)))
    solver_preds = [base_pred(ep) for ep in solver]
    solver_rest = ", ".join("%s %d" % (c, solver_preds.count(c)) for c in CLASSES if solver_preds.count(c))
    L += [f"Logs called H_other at x1 with the v5 likelihood and the class-conditional threshold: {len(flagged)} ({flagged_counts}).",
          f"nu_solver_iters_x0.25: {len(solver)} kept logs, {solver_preds.count('H_other')} called H_other before any proposal;"
          f" the rest: {solver_rest or '-'}.", ""]
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic  # noqa: F401
        have_sdk = True
    except ImportError:
        have_sdk = False
    if args5.llm and have_key and have_sdk:
        L += run_llm_stage(flagged, solver, floors, neff, T1, tau_other, lvl)
    else:
        reason = [] if args5.llm else ["not requested (--llm)"]
        reason += [] if have_key else ["ANTHROPIC_API_KEY is not set"]
        reason += [] if have_sdk else ["the anthropic Python SDK is not installed in this environment"]
        L += ["**Proposer not run: " + "; ".join(reason) + ".** The proposer (`llm_attribution.propose`) and the replay /",
              "re-scoring pipeline are implemented; run `python scripts/attribution_v5.py --llm` once both are available.", ""]
    if args5.oracle:
        L += oracle_stage(solver, floors, neff, T1, tau_other, lvl)

    # 4. generalisation summary with the undetectable fault marked
    L += ["## 4. New numerics faults (generalisation), v5 likelihood, x1", "",
          "| setting | kept / total | min residual / noise floor (median) | ours (v5) | status |", "|---|---|---|---|---|"]
    allgen = [ep for ep in episodes if ep["suite"] == "gen"]
    for s in ("nu_substeps_x2", "nu_solver_iters_x0.25"):
        eps = [ep for ep in gen if ep["setting"] == s]
        ratio = np.median([min(ep["levels"][lvl]["res"][MAIN].values()) / floors[(MAIN, lvl)]["mean"] for ep in eps])
        preds = [base_pred(ep) for ep in eps]
        counts = ", ".join(f"{c} {preds.count(c)}" for c in CLASSES + ["H_other"] if preds.count(c))
        status = ("undetectable: indistinguishable from nominal within sensor noise (best candidate is nominal); not counted as an error"
                  if s == "nu_substeps_x2" else f"{preds.count('H_nu')} correct (H_nu), {preds.count('H_other')} flagged H_other")
        L.append(f"| {s} | {len(eps)} / {sum(ep['setting'] == s for ep in allgen)} | {ratio:.1f} | {counts} | {status} |")

    # 5. Factory
    L += ["", "## 5. Factory peg insertion", "",
          "NVIDIA's pretrained `Isaac-Factory-PegInsert-Direct-v0` checkpoint (published for the Isaac Sim 5.0 asset root,",
          "212,889,115 bytes, sha256 1a138840...081496) loads on the local Isaac Lab 2.0.2 with the standard rl_games loader",
          "(`scripts/factory_eval.py`) and succeeds in 98.8% of 2 x 128 nominal episodes (100.0% and 97.7%). No training needed.", ""]
    return "\n".join(L) + "\n"


def run_llm_stage(flagged, solver, floors, neff, T, tau_other, lvl):
    import glob
    joint_names = [str(n) for n in v4.load(f"logs/{flagged[0]['batch']}.npz")["joint_names"]] if flagged else []
    proposals, jobs, extra_class = {}, {}, {}
    for ep in flagged:
        summary = proposer_summary(ep, lvl, floors, joint_names)
        tag = f"{ep['batch']}_env{ep['env']}"
        cands = llm.propose(summary, args5.samples, RAW_LOG, tag)
        proposals[tag] = cands
        for c in cands:
            key = llm.candidate_key(c)
            extra_class[key] = llm.candidate_class(c)
            jobs[(ep["batch"], key)] = replay_job(ep["batch"], key, llm.to_replay_cfg(c))
    run_jobs(list(jobs.values()), "llm")
    with open(os.path.join(os.path.dirname(args.md), "v5_llm_proposals.json"), "w") as f:
        json.dump(proposals, f, indent=1)
    return rescore(flagged, solver, proposals, extra_class, floors, neff, T, tau_other, lvl, "LLM (claude-sonnet-5-5)")


def oracle_stage(solver, floors, neff, T, tau_other, lvl):
    """Not an LLM result: inserts the true configuration (and a near miss) as candidates for the solver logs."""
    oracle = [{"kind": "do_nu", "rationale": "oracle: true configuration", **{n: llm.KNOBS[n][1] for n in llm.KNOBS},
               "robot_solver_iterations": 2, "cube_solver_iterations": 4},
              {"kind": "do_nu", "rationale": "oracle: near miss", **{n: llm.KNOBS[n][1] for n in llm.KNOBS},
               "robot_solver_iterations": 4, "cube_solver_iterations": 8}]
    proposals, jobs, extra_class = {}, {}, {}
    for ep in solver:
        tag = f"{ep['batch']}_env{ep['env']}"
        proposals[tag] = oracle
        for c in oracle:
            key = "oracle_" + llm.candidate_key(c)
            extra_class[key] = llm.candidate_class(c)
            jobs[(ep["batch"], key)] = replay_job(ep["batch"], key, llm.to_replay_cfg(c))
    run_jobs(list(jobs.values()), "oracle")
    lines = ["### Oracle check of the pipeline (NOT an LLM result)", "",
             "The true configuration (robot 2 / cube 4 solver iterations) and a near miss (4 / 8) are inserted by hand as",
             "proposals for the nu_solver_iters_x0.25 logs, replayed and re-scored like LLM proposals would be. This checks",
             "the executable-config path and bounds what a proposer could achieve.", ""]
    return lines + rescore(solver, solver, {k: v for k, v in proposals.items()}, extra_class, floors, neff, T, tau_other, lvl,
                           "oracle", key_prefix="oracle_")


def rescore(flagged, solver, proposals, extra_class, floors, neff, T, tau_other, lvl, who, key_prefix=""):
    lines = []
    results = {}
    for ep in flagged:
        tag = f"{ep['batch']}_env{ep['env']}"
        keys = sorted({key_prefix + llm.candidate_key(c) for c in proposals.get(tag, [])})
        extra = extra_residuals(ep, lvl, keys)
        p = posterior(ep, lvl, "sum", T, neff, floors, extra=extra, extra_class={k: extra_class[k] for k in extra})
        pred = max(p, key=p.get)
        allres = {**ep["levels"][lvl]["res"][MAIN], **extra}
        if min(allres.values()) > tau_other.get((lvl, pred), np.inf):
            pred = "H_other"
        best = min(allres, key=allres.get)
        results[tag] = (ep, pred, best, allres[best], len(keys))
    sol = [results[f"{ep['batch']}_env{ep['env']}"] for ep in solver if f"{ep['batch']}_env{ep['env']}" in results]
    lines += [f"Re-scored with the proposals ({who}), noise x1:", "",
              "| setting | logs re-scored | H_nu (correct for solver logs) | H_other | other classes | best candidate is a proposal | median best residual / floor |",
              "|---|---|---|---|---|---|---|"]
    for s in sorted({r[0]["setting"] for r in results.values()}):
        rs = [r for r in results.values() if r[0]["setting"] == s]
        preds = [r[1] for r in rs]
        others = ", ".join(f"{c} {preds.count(c)}" for c in CLASSES + ["H_theta-gripper"] if c != "H_nu" and preds.count(c))
        lines.append(f"| {s} | {len(rs)} | {preds.count('H_nu')} | {preds.count('H_other')} | {others or '-'} |"
                     f" {sum(r[2] not in CANDIDATES for r in rs)} | {np.median([r[3] for r in rs]) / floors[(MAIN, lvl)]['mean']:.1f} |")
    if sol:
        lines += ["", f"nu_solver_iters_x0.25: {sum(r[1] == 'H_nu' for r in sol)} of the {len(solver)} kept logs attributed to H_nu"
                  f" after re-scoring ({len(sol)} of them were re-scored)."]
    return lines + [""]


def plot_reliability(rel, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(names, ["#9a9a96", "#2a78d6", "#eb6834"]))
    labels = {names[0]: "ours, v4 likelihood", names[1]: "ours, v5 (summed, n_eff, fitted T)", names[2]: "MLP (replay)"}
    shown = [lv for lv in (1.0, 5.0) if lv in LEVELS]
    fig, axes = plt.subplots(1, len(shown), figsize=(5 * len(shown), 4.4), sharey=True, squeeze=False)
    for ax, lv in zip(axes[0], shown):
        ax.plot([0, 1], [0, 1], ":", color="#c3c2b7", linewidth=1)
        for name in names:
            rows = rel[(name, lv)]
            ls = "--" if name == names[0] else "-"
            ax.plot([r[0] for r in rows], [r[1] for r in rows], ls, color=colors[name], linewidth=2, label=labels[name])
            ax.scatter([r[0] for r in rows], [r[1] for r in rows], s=[20 + 200 * r[2] / sum(x[2] for x in rows) for r in rows],
                       color=colors[name], edgecolor="white", linewidth=1.5, zorder=3)
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
    path = os.path.join(os.path.dirname(args.md), "reliability_v5.png")
    fig.savefig(path, dpi=130)
    print(f"[attribution_v5] wrote {path}")


if __name__ == "__main__":
    text = main()
    path = os.path.join(os.path.dirname(args.md), "attribution_v5.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
