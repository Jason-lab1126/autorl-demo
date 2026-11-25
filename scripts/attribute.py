"""Inverse failure attribution on the Franka lift-cube task (v2): off-grid causes, per-episode logs,
Gaussian likelihood with a measured replay noise floor, and sensor-noise robustness.

"Real" logs: closed-loop rollouts whose cause lies OFF the candidate grid,
    mass      x1.5, x2.5, x4                        (H_theta-mass)
    friction  x0.7, x0.4, x0.25 (cube and fingers)  (H_theta-friction)
    policy    model_300, model_500 at nominal       (H_pi)
    engine    all actuators explicit IdealPD        (H_nu)
for seeds 0, 1, 2 with 8 envs each; every env's episode is one log. Batches are stored under
anonymous ids (logs/bNN.npz). Their causes are only in labels.json, read after attribution.

Candidates (unchanged grid): nominal, do(mass x2, x3, x5), do(friction x0.5, x0.3, x0.2),
do(numerics) = explicit IdealPD actuators. Each log batch is replayed open loop under every candidate.

Determinism: GPU PhysX reproduces a rollout bit for bit only when the simulator history is identical
(see check_replay_determinism.py). So every rollout here (log, reference or replay) runs as the first
and only rollout of its own process; --phase run schedules them over --workers parallel processes.

Residual of a candidate on one episode, over the steps where log and replay are both alive:
    r = mean_t mean_j (dtau_j / sd_tau_j)^2 + mean_t mean_j (dq_j / sd_q_j)^2
sd from nominal closed-loop references (ref_sN, model_1499). Torques are Isaac Lab's applied_torque:
a PD estimate under the implicit actuators, the applied torque only under explicit actuators.

New rule: log L_c = -r_c / (2 sigma^2), sigma^2 = mean replay noise floor at that sensor-noise level
(residual between two replays of the same actions in the same setting, one of them with the sensor
noise added). Classes get the mean likelihood of their candidates (uniform prior within a class),
then a posterior over the 4 classes. H_pi takes the nominal candidate's likelihood only if the episode
failed and its nominal residual is within the noise floor (mean + 4 std); otherwise H_pi gets 0.
Old rule (v1, kept for comparison): p_c ∝ exp(-r_c / T), T = median residual, summed per class,
nominal counts as H_pi if the episode failed and as "none" otherwise.

Kept logs: failed, or deviating from nominal, i.e. the noise-free nominal-replay residual is above the
noise floor at real-robot sensor noise (the deviation would be visible on hardware).

Sensor noise on the logs' q and tau: levels x0, x1, x2, x5 of (--q_noise rad, --tau_noise N·m;
the same std in N for the finger joints).

Usage (from ~/autorl-demo after ``source env.sh``):
    python scripts/attribute.py --phase run --workers 4 --headless --device cuda:0
    python scripts/attribute.py --phase analyze
v3 (adds training and out-of-hypothesis logs, and PhysX joint forces; evaluated by attribution_eval.py):
    python scripts/attribute.py --phase run --suites test train other --out_dir results/attr/v3 --workers 8 --headless --device cuda:0
Outputs in --out_dir (default results/attr/v2) and results/attr/attribution_v2.md, acc_vs_noise.png.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip
from attr_common import (  # noqa: E402  isort: skip
    ALL_SETTINGS, CANDIDATES, CLASSES, ENGINES, GEN_SETTINGS, OTHER_SETTINGS, REAL_SETTINGS, REF_SETTING, TRAIN_SETTINGS,
    new_rule, old_rule,
)


parser = argparse.ArgumentParser(description="Inverse failure attribution by open-loop replay (v2).")
parser.add_argument("--phase", type=str, required=True, choices=["run", "job", "analyze"])
parser.add_argument("--job", type=str, default=None, help="(internal) JSON job spec for --phase job.")
parser.add_argument("--jobs_file", type=str, default=None,
                    help="--phase run: run the replay_cfg jobs listed in this JSON file instead of the suites.")
parser.add_argument("--workers", type=int, default=4, help="Parallel simulator processes for --phase run.")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds of the test and H_other logs.")
parser.add_argument("--train_seeds", type=int, nargs="+", default=[3, 4, 5], help="Seeds of the training logs.")
parser.add_argument("--suites", type=str, nargs="+", default=["test"], choices=["test", "train", "other", "gen"],
                    help="Log suites to record and replay with --phase run.")
parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0])
parser.add_argument("--q_noise", type=float, default=1e-3, help="Real-robot joint position noise std [rad].")
parser.add_argument("--tau_noise", type=float, default=0.5, help="Real-robot torque noise std [N·m].")
parser.add_argument("--run_dir", type=str, default="logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="results/attr/v2")
parser.add_argument("--job_log_dir", type=str, default="logs/attr_v2")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402

from traj_metrics import add_sensor_noise, joint_sd, residual_per_env  # noqa: E402


def batch_ids(suite: str = "test") -> dict[str, tuple[str, int]]:
    """Anonymous batch id -> (setting, seed), in a fixed shuffled order (b: test, t: train, o: other, g: gen)."""
    settings, seeds, prefix = {
        "test": (REAL_SETTINGS, args_cli.seeds, "b"),
        "train": (TRAIN_SETTINGS, args_cli.train_seeds, "t"),
        "other": (OTHER_SETTINGS, args_cli.seeds, "o"),
        "gen": (GEN_SETTINGS, args_cli.seeds, "g"),
    }[suite]
    pairs = [(setting, seed) for setting in settings for seed in seeds]
    order = np.random.default_rng(12345).permutation(len(pairs))
    return {f"{prefix}{i:02d}": pairs[j] for i, j in enumerate(order)}


def out_path(*parts) -> str:
    return os.path.join(args_cli.out_dir, *parts)


# one rollout per process


def run_job(spec: dict):
    simulation_app = AppLauncher(args_cli).app
    from lift_sim import LiftSim, load_traj, save_traj

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    faults = {}
    if spec["kind"] == "replay_cfg":
        # an arbitrary executable configuration (e.g. proposed by an LLM): LiftSim keyword arguments plus
        # cube mass / friction multipliers
        cfg = dict(spec["cfg"])
        mass, fric = cfg.pop("cube_mass_scale", 1.0), cfg.pop("friction_scale", 1.0)
        sim = LiftSim(args_cli.task, args_cli.num_envs, args_cli.device, agent_cfg, **cfg)
        log = load_traj(spec["log_path"])
        sim.apply_perturbation("mass_friction", (mass, fric))
        traj = sim.rollout(int(log["seed"]), replay=log)
        tmp = spec["out"][: -len(".npz")] + ".tmp.npz"
        save_traj(tmp, traj, **sim.meta(int(log["seed"])), cfg_json=json.dumps(spec["cfg"], sort_keys=True))
        os.replace(tmp, spec["out"])
        print(f"[attribute] {spec['out']}", flush=True)
        sim.env.close()
        simulation_app.close()
        return
    if spec["kind"] == "record":
        setting = REF_SETTING if spec["setting"] == "ref" else ALL_SETTINGS[spec["setting"]]
        name, value, engine, it = setting[:4]
        faults = setting[5] if len(setting) > 5 else {}
    else:
        name, value, engine, _ = CANDIDATES[spec["cand"]]
    sim = LiftSim(args_cli.task, args_cli.num_envs, args_cli.device, agent_cfg, **ENGINES[engine], **faults)
    if spec["kind"] == "record":
        ckpt = os.path.join(args_cli.run_dir, f"model_{it}.pt") if isinstance(it, int) else \
            os.path.join(os.path.dirname(os.path.abspath(args_cli.run_dir)), it)
        policy = sim.load_policy(ckpt)
        sim.apply_perturbation(name, value)
        traj = sim.rollout(spec["seed"], policy=policy)
    else:
        log = load_traj(spec["log_path"])
        sim.apply_perturbation(name, value)
        traj = sim.rollout(int(log["seed"]), replay=log)
    tmp = spec["out"][: -len(".npz")] + ".tmp.npz"
    save_traj(tmp, traj, **sim.meta(spec.get("seed", 0) if spec["kind"] == "record" else int(log["seed"])))
    os.replace(tmp, spec["out"])
    print(f"[attribute] {spec['out']}: success {traj['success'].mean() * 100:.0f}%", flush=True)
    sim.env.close()
    simulation_app.close()


def run_all():
    os.makedirs(args_cli.job_log_dir, exist_ok=True)
    if args_cli.jobs_file:
        with open(args_cli.jobs_file) as f:
            stages = (("replay_cfg", json.load(f)),)
        return run_stages(stages)
    ids = {bid: pair for suite in args_cli.suites for bid, pair in batch_ids(suite).items()}
    labels = {bid: {"setting": s, "seed": seed, "true_class": ALL_SETTINGS[s][4]} for bid, (s, seed) in ids.items()}
    os.makedirs(args_cli.out_dir, exist_ok=True)
    if os.path.isfile(out_path("labels.json")):
        with open(out_path("labels.json")) as f:
            labels = {**json.load(f), **labels}
    with open(out_path("labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    records = [{"kind": "record", "setting": "ref", "seed": s, "out": out_path("logs", f"ref_s{s}.npz")} for s in args_cli.seeds]
    records += [
        {"kind": "record", "setting": setting, "seed": seed, "out": out_path("logs", f"{bid}.npz")}
        for bid, (setting, seed) in ids.items()
    ]
    log_names = [f"ref_s{s}" for s in args_cli.seeds] + list(ids)
    replays = [
        {"kind": "replay", "cand": cand, "log_path": out_path("logs", f"{log}.npz"), "out": out_path("replay", f"{log}__{cand}.npz")}
        for log in log_names for cand in CANDIDATES
    ]
    # a second nominal replay of each reference: the replay noise floor
    replays += [
        {"kind": "replay", "cand": "nominal", "log_path": out_path("logs", f"ref_s{s}.npz"), "out": out_path("replay", f"ref_s{s}__nominal_b.npz")}
        for s in args_cli.seeds
    ]
    for d in ("logs", "replay"):
        os.makedirs(out_path(d), exist_ok=True)
    run_stages((("record", records), ("replay", replays)))


def run_stages(stages):
    """Runs each stage's jobs (skipping those whose output exists) on --workers single-rollout processes."""
    forward = ["--task", args_cli.task, "--num_envs", str(args_cli.num_envs), "--run_dir", args_cli.run_dir,
               "--out_dir", args_cli.out_dir, "--device", args_cli.device, "--headless"]

    def launch(spec):
        name = os.path.basename(spec["out"])[: -len(".npz")]
        with open(os.path.join(args_cli.job_log_dir, f"{name}.log"), "w") as log_file:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--phase", "job", "--job", json.dumps(spec)] + forward,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
        return name, proc.returncode == 0 and os.path.isfile(spec["out"])

    for stage, jobs in stages:
        for job in jobs:
            os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        todo = [job for job in jobs if not os.path.isfile(job["out"])]
        start = time.time()
        print(f"[attribute] {stage}: {len(todo)} of {len(jobs)} jobs to run on {args_cli.workers} workers", flush=True)
        with ThreadPoolExecutor(args_cli.workers) as pool:
            failed = [name for name, ok in pool.map(launch, todo) if not ok]
        print(f"[attribute] {stage}: done in {time.time() - start:.0f}s, {len(failed)} failed", flush=True)
        if failed:
            raise RuntimeError(f"Failed jobs (see {args_cli.job_log_dir}): {failed}")


# attribution (no labels used until evaluate())


def load(name: str) -> dict:
    with np.load(out_path(name)) as f:
        return {key: f[key] for key in f.files}


def noise_floor(level: float, sd_tau, sd_q, rng) -> tuple[np.ndarray, float, float]:
    """Per-episode residuals between two nominal replays of each reference (one with sensor noise)."""
    floors = []
    for s in args_cli.seeds:
        a, b = load(f"replay/ref_s{s}__nominal.npz"), load(f"replay/ref_s{s}__nominal_b.npz")
        noisy = add_sensor_noise(a, level * args_cli.q_noise, level * args_cli.tau_noise, rng)
        floors.append(residual_per_env(noisy, b, sd_tau, sd_q)["residual"])
    floors = np.concatenate(floors)
    sigma2 = max(floors.mean(), 1e-6)  # zero when noise-free and bit-exact: keep the likelihood finite
    threshold = floors.mean() + 4.0 * floors.std() + 1e-12
    return floors, sigma2, threshold








def attribute() -> dict:
    """Per-episode residuals, the kept set and both rules' posteriors at every noise level (label-free)."""
    refs = [load(f"logs/ref_s{s}.npz") for s in args_cli.seeds]
    pooled = {key: np.concatenate([r[key] for r in refs], axis=1) for key in ("applied_torque", "joint_pos", "alive")}
    sd_tau, sd_q = joint_sd(pooled)
    rng = np.random.default_rng(0)

    # determinism inside this run: reference closed loop vs both nominal replays, all noise-free
    exact = {}
    for s in args_cli.seeds:
        ref, a, b = refs[args_cli.seeds.index(s)], load(f"replay/ref_s{s}__nominal.npz"), load(f"replay/ref_s{s}__nominal_b.npz")
        exact[f"ref_s{s}"] = (residual_per_env(ref, a, sd_tau, sd_q)["residual"].max(),
                              residual_per_env(a, b, sd_tau, sd_q)["residual"].max())

    floors = {level: noise_floor(level, sd_tau, sd_q, rng) for level in args_cli.noise_levels}
    _, _, deviation_threshold = floors[1.0] if 1.0 in floors else noise_floor(1.0, sd_tau, sd_q, rng)

    episodes = []
    for bid in sorted(batch_ids()):
        log = load(f"logs/{bid}.npz")
        reps = {c: load(f"replay/{bid}__{c}.npz") for c in CANDIDATES}
        clean = {c: residual_per_env(log, reps[c], sd_tau, sd_q) for c in CANDIDATES}
        noisy_res = {}
        for level in args_cli.noise_levels:
            noisy = add_sensor_noise(log, level * args_cli.q_noise, level * args_cli.tau_noise, rng)
            noisy_res[level] = {c: residual_per_env(noisy, reps[c], sd_tau, sd_q)["residual"] for c in CANDIDATES}
        for e in range(log["alive"].shape[1]):
            failed = not bool(log["success"][e])
            deviating = clean["nominal"]["residual"][e] > deviation_threshold
            ep = {
                "batch": bid, "env": e, "failed": failed, "deviating": bool(deviating), "kept": failed or bool(deviating),
                "final_dist_cm": float(log["final_dist"][e] * 100),
                "coverage": {c: float(clean[c]["coverage"][e]) for c in CANDIDATES},
                "levels": {},
            }
            for level in args_cli.noise_levels:
                res = {c: float(noisy_res[level][c][e]) for c in CANDIDATES}
                _, sigma2, threshold = floors[level]
                new, old = new_rule(res, failed, sigma2, threshold), old_rule(res, failed)
                ep["levels"][level] = {
                    "residual": res, "new": new, "old": old,
                    "new_pred": max(new, key=new.get), "old_pred": max(old, key=old.get),
                }
            episodes.append(ep)
    floor_stats = {level: (f.mean(), f.std(), s2, thr) for level, (f, s2, thr) in floors.items()}
    return {"episodes": episodes, "exact": exact, "floors": floor_stats, "deviation_threshold": deviation_threshold,
            "sd_tau": sd_tau, "sd_q": sd_q}


# evaluation against the hidden labels


def evaluate(result: dict):
    with open(out_path("labels.json")) as f:
        labels = json.load(f)
    episodes = result["episodes"]
    for ep in episodes:
        ep["setting"] = labels[ep["batch"]]["setting"]
        ep["true_class"] = labels[ep["batch"]]["true_class"]
    kept = [ep for ep in episodes if ep["kept"]]
    levels = args_cli.noise_levels
    pred_cols = CLASSES + ["none"]

    def accuracy(rule: str, level: float) -> dict:
        acc = {}
        for cls in CLASSES:
            eps = [ep for ep in kept if ep["true_class"] == cls]
            acc[cls] = np.mean([ep["levels"][level][f"{rule}_pred"] == cls for ep in eps]) if eps else np.nan
        acc["overall"] = np.mean([ep["levels"][level][f"{rule}_pred"] == ep["true_class"] for ep in kept])
        acc["class-balanced"] = np.nanmean([acc[c] for c in CLASSES])
        return acc

    acc = {(rule, level): accuracy(rule, level) for rule in ("new", "old") for level in levels}

    def confusion(rule: str, level: float) -> list[str]:
        rows = ["| true \\ predicted | " + " | ".join(pred_cols) + " | n |", "|---|" + "---|" * (len(pred_cols) + 1)]
        for cls in CLASSES:
            eps = [ep for ep in kept if ep["true_class"] == cls]
            counts = [sum(ep["levels"][level][f"{rule}_pred"] == p for ep in eps) for p in pred_cols]
            rows.append(f"| {cls} | " + " | ".join(str(c) for c in counts) + f" | {len(eps)} |")
        return rows

    def by_setting(rule: str, level: float) -> list[str]:
        rows = ["| setting | true class | kept / total | " + " | ".join(pred_cols) + " |", "|---|---|---|" + "---|" * len(pred_cols)]
        for setting, spec in REAL_SETTINGS.items():
            all_eps = [ep for ep in episodes if ep["setting"] == setting]
            eps = [ep for ep in all_eps if ep["kept"]]
            counts = [sum(ep["levels"][level][f"{rule}_pred"] == p for ep in eps) for p in pred_cols]
            rows.append(f"| {setting} | {spec[4]} | {len(eps)} / {len(all_eps)} | " + " | ".join(str(c) for c in counts) + " |")
        return rows

    fmt = lambda x: "-" if np.isnan(x) else f"{x * 100:.0f}%"  # noqa: E731
    lines = ["# Inverse failure attribution v2: off-grid causes, Gaussian likelihood, sensor noise", ""]
    lines += [
        f"Checkpoint model_1499 unless noted, implicit actuators unless noted, seeds {args_cli.seeds} x"
        f" {args_cli.num_envs} envs per setting, one log per episode. Torques are Isaac Lab's `applied_torque`:"
        " a PD **estimate** under implicit actuators, the applied torque only under explicit IdealPD.",
        "",
        "## 1. Replay determinism (check_replay_determinism.py)",
        "",
    ]
    lines += determinism_section()
    lines += [
        "",
        "In this run every rollout is the only rollout of its own process. Reference closed loop vs its nominal"
        " replays (max per-episode residual, noise-free):",
        "",
        "| reference | log vs replay | replay vs replay |", "|---|---|---|",
    ]
    lines += [f"| {k} | {a:.3g} | {b:.3g} |" for k, (a, b) in result["exact"].items()]
    lines += [
        "", "## 2. Noise floor and likelihood scale", "",
        f"Sensor noise x1 = q {args_cli.q_noise:g} rad, tau {args_cli.tau_noise:g} N·m (N on the fingers).",
        "sigma^2 = mean floor residual (clamped at 1e-6); H_pi / deviation threshold = mean + 4 std.", "",
        "| noise | floor mean | floor std | sigma^2 | threshold |", "|---|---|---|---|---|",
    ]
    lines += [f"| x{lv:g} | {m:.3g} | {s:.3g} | {s2:.3g} | {thr:.3g} |" for lv, (m, s, s2, thr) in result["floors"].items()]
    lines += [
        "", "## 3. Kept logs", "",
        f"Kept = failed, or noise-free nominal-replay residual above the x1 floor threshold"
        f" ({result['deviation_threshold']:.3g}).", "",
        "| true class | episodes | failed | deviating | kept |", "|---|---|---|---|---|",
    ]
    for cls in CLASSES:
        eps = [ep for ep in episodes if ep["true_class"] == cls]
        lines.append(f"| {cls} | {len(eps)} | {sum(ep['failed'] for ep in eps)} | {sum(ep['deviating'] for ep in eps)}"
                     f" | {sum(ep['kept'] for ep in eps)} |")
    lines += ["", "## 4. Accuracy per class (kept logs)", "",
              "| rule | noise | " + " | ".join(CLASSES) + " | overall | class-balanced |", "|---|---|" + "---|" * (len(CLASSES) + 2)]
    for rule in ("new", "old"):
        for level in levels:
            a = acc[(rule, level)]
            lines.append(f"| {rule} | x{level:g} | " + " | ".join(fmt(a[c]) for c in CLASSES + ["overall", "class-balanced"]) + " |")
    for level in sorted({levels[0], 1.0} & set(levels)):
        for rule in ("new", "old"):
            lines += ["", f"## Confusion, {rule} rule, noise x{level:g}", ""] + confusion(rule, level)
    if 1.0 in levels:
        lines += ["", "## Predictions per true setting, new rule, noise x1", ""] + by_setting("new", 1.0)
    lines += ["", "Per-episode residuals and posteriors for every noise level: `v2/episodes.csv`. Plot: `acc_vs_noise.png`."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(os.path.dirname(args_cli.out_dir.rstrip("/")), "attribution_v2.md"), "w") as f:
        f.write(text)
    print(text)

    write_csv(episodes)
    plot(acc)


def determinism_section() -> list[str]:
    root = os.path.join(os.path.dirname(args_cli.out_dir.rstrip("/")), "determinism")
    order = ["direct", "mass_set_reset", "fric_set_reset", "mass_rollout", "fric_rollout", "other_policy",
             "other_seed", "closed_loop_again", "fresh_replay", "fresh_closed_loop"]
    variants = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "log.npz"))) if os.path.isdir(root) else []
    if not variants:
        return ["(no determinism runs found)"]
    rows = ["Residual of a nominal replay against the nominal log after each history (mean over 8 envs)."
            " `fresh_*` run in a new process. First differing step in brackets.", "",
            "| case | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    data = {}
    for v in variants:
        log = dict(np.load(os.path.join(root, v, "log.npz")))
        sd_tau, sd_q = joint_sd(log)
        for name in order:
            path = os.path.join(root, v, f"{name}.npz")
            if os.path.isfile(path):
                rep = dict(np.load(path))
                r = residual_per_env(log, rep, sd_tau, sd_q)["residual"].mean()
                diff = np.abs(rep["joint_pos"] - log["joint_pos"]).max(axis=(1, 2))
                first = np.nonzero(diff > 0)[0]
                data[(name, v)] = f"{r:.2g}" + (f" ({first[0]})" if len(first) else "")
    for name in order:
        rows.append(f"| {name} | " + " | ".join(data.get((name, v), "") for v in variants) + " |")
    return rows


def write_csv(episodes: list[dict]):
    header = ["noise", "batch", "env", "setting", "true_class", "failed", "deviating", "kept", "final_dist_cm"]
    header += [f"r_{c}" for c in CANDIDATES] + [f"new_{c}" for c in CLASSES] + ["new_pred", "old_pred"]
    with open(out_path("episodes.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for ep in episodes:
            for level, lv in ep["levels"].items():
                row = [level, ep["batch"], ep["env"], ep["setting"], ep["true_class"], ep["failed"], ep["deviating"],
                       ep["kept"], round(ep["final_dist_cm"], 3)]
                row += [f"{lv['residual'][c]:.6g}" for c in CANDIDATES] + [f"{lv['new'][c]:.4g}" for c in CLASSES]
                row += [lv["new_pred"], lv["old_pred"]]
                f.write(",".join(str(v) for v in row) + "\n")


def plot(acc: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = args_cli.noise_levels
    colors = {"H_theta-mass": "#2a78d6", "H_theta-friction": "#eb6834", "H_nu": "#1baf7a", "H_pi": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax = axes[0]
    for rule, style in (("new", "-"), ("old", "--")):
        ys = [acc[(rule, lv)]["class-balanced"] * 100 for lv in levels]
        ax.plot(levels, ys, style, color="#3d3d3a", linewidth=2, marker="o", markersize=6, label=f"{rule} rule")
        ax.annotate(f"{rule} rule", (levels[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=9)
    ax.set_title("class-balanced accuracy", fontsize=11)
    ax = axes[1]
    end_labels = {}
    for i, (cls, color) in enumerate(colors.items()):
        ys = [acc[("new", lv)][cls] * 100 for lv in levels]
        # dashed later series stay visible where classes overlap (e.g. two at 100%)
        ax.plot(levels, ys, color=color, linewidth=2, linestyle="--" if i % 2 else "-", marker="o", markersize=6, label=cls)
        end_labels.setdefault(round(ys[-1], 1), []).append(cls)
    for y, names in end_labels.items():
        ax.annotate(", ".join(names), (levels[-1], y), textcoords="offset points", xytext=(6, 0), va="center", fontsize=9, color="#3d3d3a")
    ax.set_title("per class, new rule", fontsize=11)
    for ax in axes:
        ax.set_xticks(levels)
        ax.set_xticklabels([f"x{lv:g}" for lv in levels])
        ax.set_xlabel(f"sensor noise (x1 = q {args_cli.q_noise:g} rad, τ {args_cli.tau_noise:g} N·m)")
        ax.set_ylim(-5, 105)
        ax.set_xlim(levels[0] - 0.3, levels[-1] + 2.2)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].set_ylabel("attribution accuracy [%] (kept logs)")
    fig.tight_layout()
    path = os.path.join(os.path.dirname(args_cli.out_dir.rstrip("/")), "acc_vs_noise.png")
    fig.savefig(path, dpi=130)
    print(f"[attribute] wrote {path}")


if __name__ == "__main__":
    if args_cli.phase == "job":
        run_job(json.loads(args_cli.job))
    elif args_cli.phase == "run":
        run_all()
    else:
        evaluate(attribute())
