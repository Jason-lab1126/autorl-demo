"""Record a rollout of the vision student (or the teacher) in the camera environment.

Writes two videos of one environment: an external view, and the 84x84 image the student sees
(upscaled without smoothing). Also prints how each recorded episode ended.

    python scripts/record_student_video.py --policy student --student logs/student/bc_more_teacher/student.pt \\
        --out_prefix media/raw/student --headless --device cuda:0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Record a rollout in the camera environment.")
parser.add_argument("--policy", type=str, default="student", choices=["student", "teacher"])
parser.add_argument("--student", type=str, default=None, help="Student checkpoint (student.pt).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_index", type=int, default=0, help="Environment to film.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video_length", type=int, default=500, help="Length in steps (50 steps = 1 s).")
parser.add_argument("--out_prefix", type=str, required=True, help="Writes <prefix>_external.mp4 and <prefix>_student_view.mp4.")
parser.add_argument("--upscale", type=int, default=6, help="Integer upscaling of the student's image.")
# external camera pose in the filmed environment's frame (robot base at the origin, table along +x)
parser.add_argument("--eye", type=float, nargs=3, default=[1.45, 0.85, 0.9])
parser.add_argument("--lookat", type=float, nargs=3, default=[0.3, 0.0, 0.35])
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio.v2 as imageio
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import vision_env
from student_policy import load_student
from vision_env import cube_position, get_images, make_vision_env, student_state

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def main():
    env = make_vision_env(
        args_cli.num_envs, args_cli.device, args_cli.seed,
        viewer_eye=args_cli.eye, viewer_lookat=args_cli.lookat, viewer_env_index=args_cli.env_index,
    )
    base_env = env.unwrapped
    device, k = base_env.device, args_cli.env_index

    if args_cli.policy == "teacher":
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.abspath(args_cli.checkpoint))
        teacher = runner.get_inference_policy(device=device)
        act = lambda obs: teacher(obs)  # noqa: E731
    else:
        student = load_student(args_cli.student, device)
        act = lambda obs: student(get_images(base_env), student_state(obs))  # noqa: E731

    external, student_view, episode_ends = [], [], []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.video_length):
            external.append(base_env.render())
            student_view.append(get_images(base_env)[k].cpu().numpy())
            # cube-to-goal distance and cube height (robot base frame) one step before a possible reset
            dist = torch.norm(cube_position(obs)[k] - obs[k, 21:24]).item()
            height = cube_position(obs)[k, 2].item()
            obs, _, done, _ = env.step(act(obs))
            if bool(done[k]):
                episode_ends.append({"end_step": t + 1, "last_frame_dist_cm": round(dist * 100, 2), "last_frame_cube_height_m": round(height, 3)})
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_prefix)), exist_ok=True)
    big = [np.kron(frame, np.ones((args_cli.upscale, args_cli.upscale, 1), dtype=np.uint8)) for frame in student_view]
    imageio.mimwrite(f"{args_cli.out_prefix}_external.mp4", external, fps=vision_env.FPS, quality=9, macro_block_size=1)
    imageio.mimwrite(f"{args_cli.out_prefix}_student_view.mp4", big, fps=vision_env.FPS, quality=9, macro_block_size=1)
    print(f"[record_student_video] {args_cli.policy} env {k} of {args_cli.num_envs}, seed {args_cli.seed}: episodes {episode_ends}")
    print(f"[record_student_video] wrote {args_cli.out_prefix}_external.mp4 and {args_cli.out_prefix}_student_view.mp4")


if __name__ == "__main__":
    main()
    simulation_app.close()
