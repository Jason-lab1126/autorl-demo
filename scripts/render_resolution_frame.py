"""Renders one frame of a teacher episode from the student camera at a given resolution.

Saves the image, the cube position (robot base frame) and the camera pose to an .npz file. Run it
at 84 and at 512 with the same seed and step: the physics does not depend on the camera
resolution, so both show the same scene (the script records the joint positions to check that).
At 84 it also saves the 84x84 image upscaled to 512x512 the way SmolVLA resizes its inputs.
make_resolution_figure.py then draws the comparison.

    python scripts/render_resolution_frame.py --checkpoint <teacher.pt> --image_size 84 --step 30 \\
        --out results/resolution/frame_84.npz --headless --device cuda:0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Render one student-camera frame at a given resolution.")
parser.add_argument("--image_size", type=int, default=84)
parser.add_argument("--step", type=int, default=30, help="Control step of the episode to render.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--upscale_to", type=int, default=512, help="Also save the image resized to this size (SmolVLA's input).")
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import torch.nn.functional as F

from rsl_rl.runners import OnPolicyRunner

import vision_env
from vision_env import cube_position, get_images, make_vision_env

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def main():
    env = make_vision_env(1, args_cli.device, args_cli.seed, image_size=args_cli.image_size)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    teacher = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.step):
            obs, _, _, _ = env.step(teacher(obs))
        image = get_images(env.unwrapped)[0]
    result = {
        "image": image.cpu().numpy(),
        "cube_position": cube_position(obs)[0].cpu().numpy(),
        "joint_pos": obs[0, :9].cpu().numpy(),
        "eye": np.array(vision_env.CAMERA_EYE),
        "lookat": np.array(vision_env.CAMERA_LOOKAT),
        "focal_length": vision_env.CAMERA_FOCAL_LENGTH,
        "horizontal_aperture": 20.955,
        "step": args_cli.step,
    }
    if args_cli.upscale_to != args_cli.image_size:
        # same call as SmolVLA's resize_with_pad (square image, so no padding)
        pixels = image.permute(2, 0, 1)[None].float() / 255.0
        size = args_cli.upscale_to
        upscaled = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False)
        result["upscaled"] = (upscaled[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez(args_cli.out, **result)
    print(f"[render] wrote {args_cli.out}: cube at {result['cube_position']}, image {tuple(result['image'].shape)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
