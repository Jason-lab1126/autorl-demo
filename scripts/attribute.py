"""Inverse failure attribution on the Franka lift-cube task: which hypothesis explains a "real" log?

Four closed-loop rollouts stand in for real-robot logs, each with one hidden cause:
    real_mass    cube mass x3                                   (H_theta-mass)
    real_fric    cube and finger friction x0.3                  (H_theta-friction)
    real_policy  nominal physics, flawed policy model_300.pt    (H_pi)
    real_engine  nominal physics, explicit IdealPD actuators    (H_nu)
They are stored under anonymous ids (logs/log_A.npz ...); the cause of each id is written only to
labels.json, which is read after the posteriors are computed, to fill the confusion table.

Attribution replays each log's actions open loop, from the same seeded initial state, under every
candidate: nominal, do(mass in {x2, x3, x5}), do(friction in {x0.5, x0.3, x0.2}) and do(numerics)
(explicit IdealPD actuators, as in real_engine). The residual of a candidate is, over the steps where
both the log and the replay are alive,
    mean over steps/envs of [ mean_j (dtau_j / sd_tau_j)^2 + mean_j (dq_j / sd_q_j)^2 ]
with per-joint sd from a nominal closed-loop rollout (ref_nominal). Posterior ∝ exp(-residual / T),
T = median residual of that log over the candidates. Candidates are summed into classes:
H_theta-mass, H_theta-friction, H_nu and, for the nominal candidate, H_pi if the log failed its task
(success rate below ref_nominal's), otherwise "none" (nothing to explain).

Torques are Isaac Lab's applied_torque. With the default implicit actuators that is a PD estimate,
not the solver's torque; only under explicit actuators (real_engine, do(numerics)) is it the torque
actually applied.

Implicit and explicit actuators need separate simulator processes. Run, from ~/autorl-demo after
``source env.sh``:
    A="--headless --num_envs 8 --device cuda:0"
    python scripts/attribute.py --phase record --engine explicit $A
    python scripts/attribute.py --phase record --engine implicit $A
    python scripts/attribute.py --phase replay --engine implicit $A
    python scripts/attribute.py --phase replay --engine explicit $A
    python scripts/attribute.py --phase analyze

Outputs in --out_dir (default results/attr): logs/, replay/, ref_nominal.npz, labels.json,
residuals.csv and attribution.md.
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

# real cell -> (perturbation, multiplier, engine, checkpoint iteration, true class)
REAL_CELLS = {
    "real_mass": ("cube_mass", 3.0, "implicit", 1499, "H_theta-mass"),
    "real_fric": ("friction", 0.3, "implicit", 1499, "H_theta-friction"),
    "real_policy": ("none", 1.0, "implicit", 300, "H_pi"),
    "real_engine": ("none", 1.0, "explicit", 1499, "H_nu"),
}
LOG_IDS = dict(zip(REAL_CELLS, ["log_C", "log_A", "log_D", "log_B"]))  # anonymous ids, fixed shuffle
# candidate -> (perturbation, multiplier, engine, class)
CANDIDATES = {
    "nominal": ("none", 1.0, "implicit", "H_pi"),
    "mass_x2": ("cube_mass", 2.0, "implicit", "H_theta-mass"),
    "mass_x3": ("cube_mass", 3.0, "implicit", "H_theta-mass"),
    "mass_x5": ("cube_mass", 5.0, "implicit", "H_theta-mass"),
    "fric_x0.5": ("friction", 0.5, "implicit", "H_theta-friction"),
    "fric_x0.3": ("friction", 0.3, "implicit", "H_theta-friction"),
    "fric_x0.2": ("friction", 0.2, "implicit", "H_theta-friction"),
    "numerics": ("none", 1.0, "explicit", "H_nu"),
}
CLASSES = ["H_theta-mass", "H_theta-friction", "H_nu", "H_pi"]
FINGER_BODIES = "panda_.*finger"
LOG_KEYS = ["joint_pos", "joint_vel", "applied_torque", "computed_torque", "cube_pos", "action", "alive"]

parser = argparse.ArgumentParser(description="Inverse failure attribution by open-loop replay.")
parser.add_argument("--phase", type=str, required=True, choices=["record", "replay", "analyze"])
parser.add_argument("--engine", type=str, default="implicit", choices=["implicit", "explicit"])
parser.add_argument("--run_dir", type=str, default="logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--threshold", type=float, default=0.02, help="Success radius [m].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="results/attr")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402


def final_success(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cube-to-goal distance at each env's last alive step, and whether it is within the threshold."""
    alive = data["alive"]
    last = alive.shape[0] - 1 - np.argmax(alive[::-1], axis=0)
    final_pos = data["cube_pos"][last, np.arange(alive.shape[1])]
    final_dist = np.linalg.norm(final_pos - data["goal_pos"], axis=1)
    return final_dist, final_dist < args_cli.threshold


def simulate():
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from explicit_actuators import use_explicit_actuators

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    if args_cli.engine == "explicit":
        print(f"[attribute] explicit IdealPD actuators for {use_explicit_actuators(env_cfg)}")
    for sub in ("logs", "replay"):
        os.makedirs(os.path.join(args_cli.out_dir, sub), exist_ok=True)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base_env = env.unwrapped
    num_envs, device = base_env.num_envs, base_env.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    robot = base_env.scene["robot"]
    cube = base_env.scene["object"]
    cube_view = cube.root_physx_view
    all_ids_cpu = torch.arange(num_envs, dtype=torch.int32, device="cpu")

    # nominal physics, same handling as eval_lift.py
    default_masses = cube_view.get_masses().clone()
    default_inertias = cube_view.get_inertias().clone()
    default_materials = cube_view.get_material_properties().clone()
    default_robot_materials = robot.root_physx_view.get_material_properties().clone()
    shapes_per_body = [
        robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
        for link_path in robot.root_physx_view.link_paths[0]
    ]
    if sum(shapes_per_body) != robot.root_physx_view.max_shapes:
        raise RuntimeError("Could not map the robot's collision shapes to its links.")
    finger_ids, finger_names = robot.find_bodies(FINGER_BODIES)
    finger_shape_ids = [
        shape_id
        for body_id in finger_ids
        for shape_id in range(sum(shapes_per_body[:body_id]), sum(shapes_per_body[: body_id + 1]))
    ]
    if not finger_shape_ids:
        raise RuntimeError(f"No collision shapes found on the gripper fingers {finger_names}.")

    def apply_perturbation(name: str, value: float):
        mass_scale = value if name == "cube_mass" else 1.0
        friction_scale = value if name == "friction" else 1.0
        cube_view.set_masses(default_masses * mass_scale, all_ids_cpu)
        cube_view.set_inertias(default_inertias * mass_scale, all_ids_cpu)
        materials = default_materials.clone()
        materials[..., :2] *= friction_scale
        cube_view.set_material_properties(materials, all_ids_cpu)
        robot_materials = default_robot_materials.clone()
        robot_materials[:, finger_shape_ids, :2] *= friction_scale
        robot.root_physx_view.set_material_properties(robot_materials, all_ids_cpu)

    def goal_pos_w() -> torch.Tensor:
        command = base_env.command_manager.get_command("object_pose")
        pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
        return pos_w

    def snapshot() -> dict[str, torch.Tensor]:
        return {
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
            "applied_torque": robot.data.applied_torque.clone(),
            "computed_torque": robot.data.computed_torque.clone(),
            "cube_pos": (cube.data.root_pos_w - base_env.scene.env_origins).clone(),
        }

    # keep the last pre-reset state of envs that terminate inside step(), as in log_traj.py
    recording = False
    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    reset_now = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pre_reset = snapshot()
    original_reset_idx = base_env._reset_idx

    def recording_reset_idx(env_ids):
        if recording:
            ids = torch.arange(num_envs, device=device)[env_ids]
            ids = ids[~done[ids]]
            for key, value in snapshot().items():
                pre_reset[key][ids] = value[ids]
            reset_now[ids] = True
        original_reset_idx(env_ids)

    base_env._reset_idx = recording_reset_idx

    def rollout(policy=None, replay: dict | None = None) -> dict[str, np.ndarray]:
        """Closed loop with ``policy``, or open loop replaying ``replay['action']``."""
        nonlocal recording
        base_env.reset(seed=args_cli.seed)
        obs, _ = env.get_observations()
        done[:] = False
        init = {f"init_{key}": value.cpu().numpy() for key, value in snapshot().items()}
        goal = goal_pos_w() - base_env.scene.env_origins
        actions_in = None if replay is None else torch.from_numpy(replay["action"]).to(device)
        max_steps = base_env.max_episode_length + 5 if replay is None else len(actions_in)
        log = {key: [] for key in LOG_KEYS}
        recording = True
        for t in range(max_steps):
            actions = policy(obs) if replay is None else actions_in[t]
            alive = ~done
            reset_now[:] = False
            obs, _, _, _ = env.step(actions)
            state = snapshot()
            for key in state:
                state[key][reset_now] = pre_reset[key][reset_now]
                log[key].append(state[key])
            log["action"].append(actions.clone())
            log["alive"].append(alive.clone())
            done[reset_now] = True
            if replay is None and bool(done.all()):
                break
        recording = False
        if replay is None and not bool(done.all()):
            raise RuntimeError("Some environments never finished their episode.")
        out = {key: torch.stack(values).cpu().numpy() for key, values in log.items()}
        if replay is not None:
            out["alive"] &= replay["alive"]  # recorded actions past the log's episode belong to the next one
        out["goal_pos"] = goal.cpu().numpy()
        out.update(init)
        out["final_dist"], out["success"] = final_success(out)
        return out

    def report(tag: str, data: dict):
        print(
            f"[attribute] {tag:<28} success {data['success'].mean() * 100:5.1f}% ({data['success'].sum()}/{num_envs})"
            f"  mean final dist {data['final_dist'].mean() * 100:6.2f} cm  alive steps {data['alive'].sum()}",
            flush=True,
        )

    joint_names = np.array(robot.joint_names)
    meta = {"joint_names": joint_names, "engine": args_cli.engine, "seed": args_cli.seed}
    with torch.inference_mode():
        if args_cli.phase == "record":
            labels_path = os.path.join(args_cli.out_dir, "labels.json")
            labels = json.load(open(labels_path)) if os.path.isfile(labels_path) else {}
            todo = [(cell, spec) for cell, spec in REAL_CELLS.items() if spec[2] == args_cli.engine]
            if args_cli.engine == "implicit":
                todo = [("ref_nominal", ("none", 1.0, "implicit", 1499, None))] + todo
            for cell, (name, value, _, it, true_class) in sorted(todo, key=lambda c: -c[1][3]):  # 1499 first
                runner.load(os.path.join(args_cli.run_dir, f"model_{it}.pt"))
                policy = runner.get_inference_policy(device=device)
                apply_perturbation(name, value)
                data = rollout(policy=policy)
                if cell == "ref_nominal":
                    np.savez(os.path.join(args_cli.out_dir, "ref_nominal.npz"), **data, **meta)
                    report(cell, data)
                else:
                    np.savez(os.path.join(args_cli.out_dir, "logs", f"{LOG_IDS[cell]}.npz"), **data, **meta)
                    labels[LOG_IDS[cell]] = {"cell": cell, "true_class": true_class, "checkpoint_iter": it}
                    report(f"{LOG_IDS[cell]} ({cell})", data)
            json.dump(labels, open(labels_path, "w"), indent=2)
        else:
            log_paths = sorted(glob.glob(os.path.join(args_cli.out_dir, "logs", "log_*.npz")))
            if len(log_paths) != len(REAL_CELLS):
                raise FileNotFoundError(f"Expected {len(REAL_CELLS)} logs, found {log_paths}; run --phase record first.")
            for path in log_paths:
                log_id = os.path.basename(path)[: -len(".npz")]
                with np.load(path) as f:
                    log = {"action": f["action"], "alive": f["alive"]}
                for cand, (name, value, engine, _) in CANDIDATES.items():
                    if engine != args_cli.engine:
                        continue
                    apply_perturbation(name, value)
                    data = rollout(replay=log)
                    np.savez(os.path.join(args_cli.out_dir, "replay", f"{log_id}__{cand}.npz"), **data, **meta)
                    report(f"{log_id} <- {cand}", data)
        apply_perturbation("none", 1.0)
    env.close()
    simulation_app.close()


def analyze():
    out_dir = args_cli.out_dir
    load = lambda path: dict(np.load(path))  # noqa: E731
    ref = load(os.path.join(out_dir, "ref_nominal.npz"))
    ref_success = ref["success"].mean()
    alive = ref["alive"]
    sd_tau = np.maximum(ref["applied_torque"][alive].std(axis=0), 1e-6)
    sd_q = np.maximum(ref["joint_pos"][alive].std(axis=0), 1e-6)

    rows, results = [], {}
    log_ids = sorted(os.path.basename(p)[: -len(".npz")] for p in glob.glob(os.path.join(out_dir, "logs", "log_*.npz")))
    for log_id in log_ids:
        log = load(os.path.join(out_dir, "logs", f"{log_id}.npz"))
        log_failed = log["success"].mean() < ref_success
        res = {}
        for cand in CANDIDATES:
            rep = load(os.path.join(out_dir, "replay", f"{log_id}__{cand}.npz"))
            for key in ("init_joint_pos", "init_cube_pos"):
                if not np.allclose(rep[key], log[key], atol=1e-5):
                    raise RuntimeError(f"{log_id} <- {cand}: replay starts from a different {key}.")
            steps = min(len(rep["alive"]), len(log["alive"]))
            mask = rep["alive"][:steps] & log["alive"][:steps]
            d_tau = ((rep["applied_torque"][:steps] - log["applied_torque"][:steps]) / sd_tau)[mask]
            d_q = ((rep["joint_pos"][:steps] - log["joint_pos"][:steps]) / sd_q)[mask]
            tau_term, q_term = (d_tau**2).mean(), (d_q**2).mean()
            res[cand] = tau_term + q_term
            rows.append({
                "log": log_id, "candidate": cand, "residual": res[cand], "torque_term": tau_term, "pos_term": q_term,
                "coverage": mask.sum() / log["alive"].sum(), "replay_success": rep["success"].mean(),
            })
        r = np.array([res[c] for c in CANDIDATES])
        temp = max(np.median(r), 1e-12)
        post = np.exp(-(r - r.min()) / temp)
        post /= post.sum()
        cand_post = dict(zip(CANDIDATES, post))
        class_post = {c: 0.0 for c in CLASSES + ["none"]}
        for cand, p in cand_post.items():
            cls = CANDIDATES[cand][3]
            class_post[cls if cls != "H_pi" or log_failed else "none"] += p
        for row in rows[-len(CANDIDATES):]:
            row["posterior"] = cand_post[row["candidate"]]
        pred = max(class_post, key=class_post.get)
        map_cand = min(res, key=res.get)
        map_class = CANDIDATES[map_cand][3]
        results[log_id] = {
            "map_cand": map_cand, "map_class": map_class if map_class != "H_pi" or log_failed else "none",
            "cand_post": cand_post, "class_post": class_post, "pred": pred, "temp": temp,
            "success": log["success"].mean(), "failed": log_failed,
        }

    # only now read the hidden causes
    labels = json.load(open(os.path.join(out_dir, "labels.json")))
    pred_cols = CLASSES + (["none"] if any(r["pred"] == "none" for r in results.values()) else [])
    confusion = {t: {p: 0 for p in pred_cols} for t in CLASSES}
    for log_id, r in results.items():
        confusion[labels[log_id]["true_class"]][r["pred"]] += 1

    with open(os.path.join(out_dir, "residuals.csv"), "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")

    lines = ["# Inverse failure attribution by open-loop replay", ""]
    lines += [
        f"Seed {args_cli.seed}, {len(ref['success'])} envs per log. ref_nominal success {ref_success * 100:.0f}%.",
        "Torques are Isaac Lab's `applied_torque`: a PD **estimate** under the default implicit actuators,",
        "the torque actually applied only under explicit IdealPD (real_engine, do(numerics)).",
        "",
        "## Confusion (rows: true cause, columns: class with the highest posterior)",
        "",
        "| true \\ predicted | " + " | ".join(pred_cols) + " |",
        "|---|" + "---|" * len(pred_cols),
    ]
    lines += [f"| {t} | " + " | ".join(str(confusion[t][p]) for p in pred_cols) + " |" for t in CLASSES]
    lines += ["", "## Class posteriors per log", "", "| log | true cause | task success | predicted | "
              + " | ".join(CLASSES + ["none"]) + " |", "|---|---|---|---|" + "---|" * (len(CLASSES) + 1)]
    for log_id, r in results.items():
        lines.append(
            f"| {log_id} | {labels[log_id]['cell']} | {r['success'] * 100:.0f}% | {r['pred']} | "
            + " | ".join(f"{r['class_post'][c]:.3f}" for c in CLASSES + ["none"]) + " |"
        )
    lines += ["", "## Residual / posterior per candidate", "",
              "| log | T | " + " | ".join(CANDIDATES) + " |", "|---|---|" + "---|" * len(CANDIDATES)]
    for log_id, r in results.items():
        log_rows = {row["candidate"]: row for row in rows if row["log"] == log_id}
        lines.append(
            f"| {log_id} | {r['temp']:.3g} | "
            + " | ".join(f"{log_rows[c]['residual']:.3g} / {r['cand_post'][c]:.3f}" for c in CANDIDATES) + " |"
        )
    lines += [
        "", "## Diagnostic: class of the single lowest-residual candidate (not the specified posterior rule)", "",
        "Summing posteriors over classes favours classes with more candidates (mass and friction have 3 each,",
        "numerics and nominal 1), and T = median residual flattens the posterior when one candidate is far ahead.",
        "", "| log | true cause | min-residual candidate | class | second-lowest / lowest residual |", "|---|---|---|---|---|",
    ]
    for log_id, r in results.items():
        res_sorted = sorted(row["residual"] for row in rows if row["log"] == log_id)
        lines.append(
            f"| {log_id} | {labels[log_id]['cell']} | {r['map_cand']} | {r['map_class']} |"
            f" {res_sorted[1] / max(res_sorted[0], 1e-12):.3g} |"
        )
    lines += ["", "Coverage (both-alive steps / log alive steps) is in residuals.csv."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "attribution.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    if args_cli.phase == "analyze":
        analyze()
    else:
        simulate()
