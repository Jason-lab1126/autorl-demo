"""Record a video of a trained lift-cube policy with a custom camera.

Isaac Lab's play.py always films the default wide overview. This does the same rollout but places
the viewport camera relative to one environment, for close-ups.

Usage (from ~/autorl-demo, after ``source env.sh``):
    python scripts/record_video.py --checkpoint <model.pt> --name closeup --headless --device cuda:0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Record a policy rollout with a custom camera.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=500, help="Length of the video in steps (50 steps = 1 s).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--name", type=str, default="closeup", help="Output is <run_dir>/videos/play/<name>.mp4.")
# camera pose in the frame of environment 0 (robot base at the origin, table along +x)
parser.add_argument("--eye", type=float, nargs=3, default=[1.2, 0.7, 0.75], help="Camera position [m].")
parser.add_argument("--lookat", type=float, nargs=3, default=[0.45, 0.0, 0.25], help="Camera target [m].")
parser.add_argument("--no_markers", action="store_true", help="Hide the goal and end-effector frame markers.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.eye)
    env_cfg.viewer.lookat = tuple(args_cli.lookat)
    if args_cli.no_markers:
        env_cfg.commands.object_pose.debug_vis = False
        env_cfg.scene.ee_frame.debug_vis = False

    checkpoint = os.path.abspath(args_cli.checkpoint)
    video_dir = os.path.join(os.path.dirname(checkpoint), "videos", "play")
    out_path = os.path.join(video_dir, f"{args_cli.name}.mp4")
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} already exists; pick another --name.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=args_cli.name,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.video_length):
            obs, _, _, _ = env.step(policy(obs))
    env.close()

    os.rename(os.path.join(video_dir, f"{args_cli.name}-step-0.mp4"), out_path)
    print(f"[record_video] wrote {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
