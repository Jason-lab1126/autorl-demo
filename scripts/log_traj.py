"""Per-step trajectory logging for sim-to-real failure attribution on the Franka lift-cube task.

Rolls out one episode per environment for each cell and records, at every control step, the robot's
joint positions/velocities, the actuator torques (applied and computed), the policy action and the
cube position. All cells are reset with the same seed, so they see the same goals and cube spawns.

Cells:
    nominal          closed loop, nominal physics
    nominal_repeat   identical to nominal, run again (does the simulator reproduce itself?)
    friction_x0.5    closed loop, cube and finger friction x0.5
    mass_x5          closed loop, cube mass and inertia x5
    replay_fric0.5   open loop: nominal's recorded actions replayed with friction x0.5
    replay_mass5     open loop: nominal's recorded actions replayed with cube mass x5

Step t holds the state after action t was applied. On the step where an environment terminates, the
state is the last one before its reset (captured in a _reset_idx hook, as in eval_lift.py). ``alive``
is True from the first step up to and including that terminal step, False afterwards. For replay
cells it is also False once the nominal episode being replayed has ended, since the recorded actions
after that point belong to the next episode.

Usage (from ~/autorl-demo, after ``source env.sh``):
    python scripts/log_traj.py --ckpt logs/rsl_rl/franka_lift/<run>/model_1499.pt --headless --num_envs 8

Outputs: <out_dir>/<cell>.npz (default results/traj). Plot with ``python scripts/plot_traj.py``.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

# cell name -> (perturbation, multiplier, replay nominal's actions open loop)
CELLS = {
    "nominal": ("none", 1.0, False),
    "nominal_repeat": ("none", 1.0, False),
    "friction_x0.5": ("friction", 0.5, False),
    "mass_x5": ("cube_mass", 5.0, False),
    "replay_fric0.5": ("friction", 0.5, True),
    "replay_mass5": ("cube_mass", 5.0, True),
}
FINGER_BODIES = "panda_.*finger"

parser = argparse.ArgumentParser(description="Per-step trajectory logging for the lift-cube task.")
parser.add_argument("--ckpt", type=str, required=True, help="Path to model_<iter>.pt.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=8, help="Episodes per cell (one per environment).")
parser.add_argument("--threshold", type=float, default=0.02, help="Success radius [m].")
parser.add_argument("--seed", type=int, default=0, help="Seed every cell is reset with.")
parser.add_argument("--out_dir", type=str, default="results/traj")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

COMMAND_NAME = "object_pose"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    checkpoint = os.path.abspath(args_cli.ckpt)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base_env = env.unwrapped
    num_envs, device = base_env.num_envs, base_env.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

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
        command = base_env.command_manager.get_command(COMMAND_NAME)
        pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
        return pos_w

    def snapshot() -> dict[str, torch.Tensor]:
        """Current per-env state; cube position is relative to the env origin."""
        return {
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
            "applied_torque": robot.data.applied_torque.clone(),
            "computed_torque": robot.data.computed_torque.clone(),
            "cube_pos": (cube.data.root_pos_w - base_env.scene.env_origins).clone(),
        }

    # The env resets terminated sub-environments inside step(); keep their last pre-reset state.
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

    def run_cell(replay_actions: torch.Tensor | None, replay_alive: torch.Tensor | None) -> dict[str, np.ndarray]:
        nonlocal recording
        base_env.reset(seed=args_cli.seed)
        obs, _ = env.get_observations()
        done[:] = False
        goal = goal_pos_w() - base_env.scene.env_origins
        max_steps = len(replay_actions) if replay_actions is not None else base_env.max_episode_length + 5
        log = {key: [] for key in ["joint_pos", "joint_vel", "applied_torque", "computed_torque", "cube_pos", "action", "alive"]}
        recording = True
        for t in range(max_steps):
            actions = policy(obs) if replay_actions is None else replay_actions[t]
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
            if replay_actions is None and bool(done.all()):
                break
        recording = False
        if replay_actions is None and not bool(done.all()):
            raise RuntimeError("Some environments never finished their episode.")
        out = {key: torch.stack(values).cpu().numpy() for key, values in log.items()}
        if replay_alive is not None:
            out["alive"] &= replay_alive.cpu().numpy()
        out["goal_pos"] = goal.cpu().numpy()
        return out

    nominal = None
    joint_names = np.array(robot.joint_names)
    with torch.inference_mode():
        for cell, (name, value, replay) in CELLS.items():
            apply_perturbation(name, value)
            if replay:
                data = run_cell(torch.from_numpy(nominal["action"]).to(device), torch.from_numpy(nominal["alive"]))
            else:
                data = run_cell(None, None)
            if cell == "nominal":
                nominal = data
            # final state = last alive step of each env
            last = data["alive"].shape[0] - 1 - np.argmax(data["alive"][::-1], axis=0)
            final_pos = data["cube_pos"][last, np.arange(num_envs)]
            final_dist = np.linalg.norm(final_pos - data["goal_pos"], axis=1)
            success = final_dist < args_cli.threshold
            np.savez(
                os.path.join(args_cli.out_dir, f"{cell}.npz"),
                **data, joint_names=joint_names, final_dist=final_dist, success=success,
                perturbation=name, value=value, replay=replay, seed=args_cli.seed, checkpoint=checkpoint,
            )
            print(
                f"[log_traj] {cell:<15} success@{args_cli.threshold * 100:g}cm {success.mean() * 100:5.1f}%"
                f" ({success.sum()}/{num_envs})  mean final dist {final_dist.mean() * 100:6.2f} cm"
                f"  steps {data['alive'].shape[0]}  per-env [{', '.join('%.1f' % (d * 100) for d in final_dist)}]",
                flush=True,
            )
        apply_perturbation("none", 1.0)
    print(f"[log_traj] wrote {len(CELLS)} cells to {args_cli.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
