"""Robustness eval matrix for rsl_rl PPO checkpoints on the Franka lift-cube task.

For every (checkpoint, perturbation, level) cell this rolls out one episode in each environment
and measures the cube-to-goal distance at the END of the episode (the last simulated state before
the environment resets), not the per-step average that training logs. A cell's success rate is the
fraction of episodes ending within ``--threshold`` of the goal.

Everything runs in a single simulator instance: perturbations are applied at runtime between
cells, and every cell is reset with the same seed so all cells see the same goals and the same
underlying spawn draws.

Usage (from ~/autorl-demo, after ``source env.sh``):
    python scripts/eval_lift.py --run_dir logs/rsl_rl/franka_lift/<run> --headless --device cuda:0

Outputs (in --out_dir, default ./results):
    eval_matrix_cells.csv   one row per cell with all statistics
    eval_matrix.csv         success-rate table (rows: perturbation/level, columns: checkpoint)
    eval_matrix.md          the same table plus mean final distance, in markdown

Plot the sweeps afterwards with ``python scripts/plot_eval.py``.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

# Perturbation levels as {label: value}. "none" is the shared unperturbed baseline (every multiplier at
# x1, no noise), evaluated once per checkpoint.
PERTURBATIONS = {
    # multiplier on the cube's mass (inertia scaled with it)
    "cube_mass": {"x2": 2.0, "x3": 3.0, "x5": 5.0, "x10": 10.0},
    # multiplier on the static and dynamic friction of the cube and of the gripper fingers
    "friction": {"x0.5": 0.5, "x0.2": 0.2, "x0.1": 0.1},
    # multiplier on OBS_NOISE_STD, Gaussian noise added to the policy's observations every step
    "obs_noise": {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x5": 5.0},
    # std of Gaussian noise added to the policy's raw actions every step (arm and gripper)
    "action_noise": {"medium": 0.1, "strong": 0.5},
    # multiplier on the cube's x/y spawn range (default x: +-0.10 m, y: +-0.25 m)
    "spawn_range": {"medium": 1.5, "strong": 2.0},
}
# Base noise std per observation term; terms not listed (goal command, last action) stay noise-free.
OBS_NOISE_STD = {"joint_pos": 0.01, "joint_vel": 0.1, "object_position": 0.01}  # rad, rad/s, m
FINGER_BODIES = "panda_.*finger"

parser = argparse.ArgumentParser(description="Robustness eval matrix for the lift-cube task.")
parser.add_argument("--run_dir", type=str, required=True, help="Training run directory holding model_<iter>.pt.")
parser.add_argument("--iters", type=int, nargs="+", default=[100, 300, 500, 1499], help="Checkpoint iterations.")
parser.add_argument(
    "--perturbations", type=str, nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS),
    help="Perturbations to evaluate (the unperturbed baseline is always included).",
)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=100, help="Episodes per cell (one per environment).")
parser.add_argument("--threshold", type=float, default=0.02, help="Success radius [m].")
parser.add_argument("--seed", type=int, default=0, help="Seed every cell is reset with.")
parser.add_argument("--out_dir", type=str, default="results")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

COMMAND_NAME = "object_pose"
SPAWN_EVENT = "reset_object_position"
LIFT_HEIGHT = 0.04  # same threshold the task's rewards use for "lifted"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed

    run_dir = os.path.abspath(args_cli.run_dir)
    checkpoints = {it: os.path.join(run_dir, f"model_{it}.pt") for it in args_cli.iters}
    missing = [path for path in checkpoints.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base_env = env.unwrapped
    num_envs, device = base_env.num_envs, base_env.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    robot = base_env.scene["robot"]
    cube = base_env.scene["object"]
    cube_view = cube.root_physx_view
    all_ids_cpu = torch.arange(num_envs, dtype=torch.int32, device="cpu")

    # nominal values the perturbations are applied relative to
    default_masses = cube_view.get_masses().clone()
    default_inertias = cube_view.get_inertias().clone()
    default_materials = cube_view.get_material_properties().clone()  # (env, shape, [static, dynamic, restitution])
    default_robot_materials = robot.root_physx_view.get_material_properties().clone()
    # The articulation view exposes materials per collision shape, flattened over links, so work out
    # which shapes belong to the gripper fingers.
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
    print(
        f"[eval_lift] nominal friction (static, dynamic): cube {default_materials[0, 0, :2].tolist()},"
        f" fingers {finger_names} {default_robot_materials[0, finger_shape_ids[0], :2].tolist()}"
    )
    default_spawn_range = dict(base_env.event_manager.get_term_cfg(SPAWN_EVENT).params["pose_range"])

    # per-dimension observation noise std, laid out like the flattened policy observation
    obs_terms = base_env.observation_manager.active_terms["policy"]
    obs_dims = [int(torch.tensor(d).prod()) for d in base_env.observation_manager.group_obs_term_dim["policy"]]
    unknown = set(OBS_NOISE_STD) - set(obs_terms)
    if unknown:
        raise KeyError(f"OBS_NOISE_STD names terms the policy does not observe: {unknown}")
    obs_noise_std = torch.cat(
        [torch.full((dim,), OBS_NOISE_STD.get(name, 0.0), device=device) for name, dim in zip(obs_terms, obs_dims)]
    )

    def apply_perturbation(name: str, value: float) -> tuple[float, float]:
        """Sets the physical perturbations for a cell and returns its (obs, action) noise scales."""
        mass_scale = value if name == "cube_mass" else 1.0
        friction_scale = value if name == "friction" else 1.0
        spawn_scale = value if name == "spawn_range" else 1.0

        cube_view.set_masses(default_masses * mass_scale, all_ids_cpu)
        cube_view.set_inertias(default_inertias * mass_scale, all_ids_cpu)
        materials = default_materials.clone()
        materials[..., :2] *= friction_scale
        cube_view.set_material_properties(materials, all_ids_cpu)
        robot_materials = default_robot_materials.clone()
        robot_materials[:, finger_shape_ids, :2] *= friction_scale
        robot.root_physx_view.set_material_properties(robot_materials, all_ids_cpu)

        spawn_cfg = base_env.event_manager.get_term_cfg(SPAWN_EVENT)
        spawn_cfg.params["pose_range"] = {
            axis: (lo * spawn_scale, hi * spawn_scale) if axis in ("x", "y") else (lo, hi)
            for axis, (lo, hi) in default_spawn_range.items()
        }
        base_env.event_manager.set_term_cfg(SPAWN_EVENT, spawn_cfg)

        return (value if name == "obs_noise" else 0.0), (value if name == "action_noise" else 0.0)

    def goal_pos_w() -> torch.Tensor:
        """Commanded cube position in the world frame (commands are given in the robot base frame)."""
        command = base_env.command_manager.get_command(COMMAND_NAME)
        pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
        return pos_w

    # per-cell episode buffers, filled by the reset hook below
    recording = False
    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    goal_start_w = torch.zeros(num_envs, 3, device=device)
    final_dist = torch.zeros(num_envs, device=device)
    final_height = torch.zeros(num_envs, device=device)
    dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # The env resets terminated sub-environments inside step(), so the final state is gone by the
    # time step() returns. Capture it right before the reset instead.
    original_reset_idx = base_env._reset_idx

    def recording_reset_idx(env_ids):
        if recording:
            ids = torch.arange(num_envs, device=device)[env_ids]
            ids = ids[~done[ids]]
            cube_pos_w = cube.data.root_pos_w
            final_dist[ids] = torch.norm(goal_start_w[ids] - cube_pos_w[ids], dim=1)
            final_height[ids] = cube_pos_w[ids, 2] - base_env.scene.env_origins[ids, 2]
            dropped[ids] = ~base_env.termination_manager.time_outs[ids]
            done[ids] = True
        original_reset_idx(env_ids)

    base_env._reset_idx = recording_reset_idx

    def run_cell(policy, obs_noise_scale: float, action_noise_scale: float) -> dict:
        """Rolls out one episode per environment and returns the cell's statistics."""
        nonlocal recording
        base_env.reset(seed=args_cli.seed)
        obs, _ = env.get_observations()
        done[:] = False
        goal_start_w[:] = goal_pos_w()
        recording = True
        for _ in range(base_env.max_episode_length + 5):
            if obs_noise_scale > 0.0:
                obs = obs + torch.randn_like(obs) * obs_noise_std * obs_noise_scale
            actions = policy(obs)
            if action_noise_scale > 0.0:
                actions = actions + torch.randn_like(actions) * action_noise_scale
            obs, _, _, _ = env.step(actions)
            if bool(done.all()):
                break
        recording = False
        if not bool(done.all()):
            raise RuntimeError("Some environments never finished their episode.")
        return {
            "num_episodes": num_envs,
            "success_rate": (final_dist < args_cli.threshold).float().mean().item(),
            "final_dist_mean_m": final_dist.mean().item(),
            "final_dist_median_m": final_dist.median().item(),
            "final_dist_max_m": final_dist.max().item(),
            "lifted_rate": (final_height > LIFT_HEIGHT).float().mean().item(),
            "dropped_rate": dropped.float().mean().item(),
        }

    cells = [("none", "none", 1.0)]
    cells += [(name, level, value) for name in args_cli.perturbations for level, value in PERTURBATIONS[name].items()]

    rows = []
    start = time.time()
    with torch.inference_mode():
        for it, path in checkpoints.items():
            runner.load(path)
            policy = runner.get_inference_policy(device=device)
            for name, level, value in cells:
                cell_start = time.time()
                obs_noise_scale, action_noise_scale = apply_perturbation(name, value)
                stats = run_cell(policy, obs_noise_scale, action_noise_scale)
                rows.append({"iter": it, "perturbation": name, "level": level, "value": value, **stats})
                print(
                    f"[eval_lift] iter {it:>5} {name:>12}/{level:<6} success@{args_cli.threshold * 100:g}cm"
                    f" {stats['success_rate'] * 100:5.1f}%  mean dist {stats['final_dist_mean_m'] * 100:6.2f} cm"
                    f"  ({time.time() - cell_start:.1f}s)",
                    flush=True,
                )
        apply_perturbation("none", 1.0)
    print(f"[eval_lift] {len(rows)} cells in {time.time() - start:.0f}s")

    write_outputs(rows, cells, list(checkpoints))
    env.close()


def write_outputs(rows: list[dict], cells: list[tuple], iters: list[int]):
    out_dir = args_cli.out_dir
    with open(os.path.join(out_dir, "eval_matrix_cells.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_cell = {(r["iter"], r["perturbation"], r["level"]): r for r in rows}
    header = ["perturbation", "level", "value"] + [f"iter_{it}" for it in iters]

    def table(key: str, scale: float, fmt: str) -> list[list[str]]:
        return [
            [name, level, f"{value:g}"] + [format(by_cell[(it, name, level)][key] * scale, fmt) for it in iters]
            for name, level, value in cells
        ]

    success = table("success_rate", 100.0, ".0f")
    with open(os.path.join(out_dir, "eval_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(success)

    def markdown(body: list[list[str]]) -> str:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)

    threshold_cm = args_cli.threshold * 100
    with open(os.path.join(out_dir, "eval_matrix.md"), "w") as f:
        f.write(f"# Lift-cube eval matrix\n\n")
        f.write(f"Run: `{os.path.abspath(args_cli.run_dir)}`  \n")
        f.write(f"Task: `{args_cli.task}`, {args_cli.num_envs} episodes per cell, seed {args_cli.seed}, ")
        f.write("deterministic policy actions. Distances are cube-to-goal at the end of each episode.\n\n")
        f.write(f"## Success rate [%] (final distance < {threshold_cm:g} cm)\n\n{markdown(success)}\n\n")
        f.write(f"## Mean final distance [cm]\n\n{markdown(table('final_dist_mean_m', 100.0, '.2f'))}\n\n")
        f.write("## Perturbation values\n\n")
        f.write("- `cube_mass`: multiplier on the cube's mass and inertia\n")
        f.write("- `friction`: multiplier on the static and dynamic friction of the cube and the gripper fingers\n")
        f.write(f"- `obs_noise`: multiplier on per-step Gaussian observation noise with std {OBS_NOISE_STD}\n")
        f.write("- `action_noise`: std of per-step Gaussian noise on the raw policy actions (arm and gripper)\n")
        f.write("- `spawn_range`: multiplier on the cube's x/y spawn range\n")
    print(f"[eval_lift] wrote {out_dir}/eval_matrix.md, eval_matrix.csv and eval_matrix_cells.csv")


if __name__ == "__main__":
    main()
    simulation_app.close()
