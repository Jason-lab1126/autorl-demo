"""Closed-loop evaluation of the vision student in the camera environment.

Same metric as eval_lift.py: cube-to-goal distance at the END of each episode, one episode per
environment, success if within --threshold. Pass --policy teacher to run the RL teacher in the
same camera environment as a reference. Pass --policy smolvla to query a SmolVLA policy served by
smolvla_server.py (conda env lerobot-gpu) on a local socket; it replans every --replan steps.
With --crop_size C it also sends an oracle crop (C x C, from a --crop_render_size camera, centered
on the cube's true position) as a second camera, for SmolVLA trained on such crops. With
--oracle_state it appends the true cube position to the state, for SmolVLA trained that way.

    python scripts/eval_student.py --student logs/student/bc_500ep/student.pt --out_dir results/student_bc \\
        --headless --device cuda:0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Closed-loop evaluation of the vision student.")
parser.add_argument("--policy", type=str, default="student", choices=["student", "teacher", "smolvla"])
parser.add_argument("--student", type=str, default=None, help="Student checkpoint (student.pt).")
parser.add_argument("--server_port", type=int, default=47310, help="smolvla_server.py port (--policy smolvla).")
parser.add_argument("--replan", type=int, default=10, help="Steps of each SmolVLA action chunk to execute before replanning.")
parser.add_argument("--image_size", type=int, default=84, help="Camera resolution.")
parser.add_argument("--crop_size", type=int, default=0, help="Send an oracle crop of this size to SmolVLA (0 = off).")
parser.add_argument("--oracle_state", action="store_true", help="Append the true cube position to SmolVLA's state.")
parser.add_argument("--crop_render_size", type=int, default=512, help="Resolution of the camera the crop is taken from.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of episodes (one per environment).")
parser.add_argument("--threshold", type=float, default=0.02, help="Success radius [m].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="results/student_bc")
# visual perturbations (see vision_env.make_vision_env)
parser.add_argument("--light_scale", type=float, default=1.0)
parser.add_argument("--cube_color", type=float, nargs=3, default=None, help="Flat RGB color for the cube, 0 to 1.")
parser.add_argument("--table_color", type=float, nargs=3, default=None, help="Flat RGB color for the table, 0 to 1.")
parser.add_argument("--camera_shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Camera offset [m].")
parser.add_argument("--save_view", type=str, default=None, help="Save the student's image of env 0 at a few steps to this PNG.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import torch

from rsl_rl.runners import OnPolicyRunner

import vision_env
from student_policy import load_student
from vision_env import EpisodeEndRecorder, cube_position, get_images, make_vision_env, oracle_crops, student_state

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

INSTRUCTION = "pick up the cube and move it to the target"  # the task string in the datasets
LIFT_HEIGHT = 0.04  # same threshold the task's rewards use for "lifted"


def main():
    env = make_vision_env(
        args_cli.num_envs, args_cli.device, args_cli.seed, image_size=args_cli.image_size, light_scale=args_cli.light_scale,
        cube_color=args_cli.cube_color, table_color=args_cli.table_color, camera_shift=args_cli.camera_shift,
        crop_render_size=args_cli.crop_render_size if args_cli.crop_size > 0 else 0,
    )
    base_env = env.unwrapped
    device = base_env.device

    def images():
        return get_images(base_env)

    connection = None
    if args_cli.policy == "teacher":
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.abspath(args_cli.checkpoint))
        teacher = runner.get_inference_policy(device=device)
        act = lambda obs, t: teacher(obs)  # noqa: E731
        source = os.path.abspath(args_cli.checkpoint)
    elif args_cli.policy == "smolvla":
        from multiprocessing.connection import Client

        connection = Client(("127.0.0.1", args_cli.server_port), authkey=b"autorl-demo")
        chunk = None

        def act(obs, t):
            nonlocal chunk
            if t % args_cli.replan == 0:
                state = student_state(obs)
                if args_cli.oracle_state:
                    # privileged: the simulator's true cube position
                    state = torch.cat([state, cube_position(obs)], dim=1)
                request = {"image": images().cpu().numpy(), "state": state.cpu().numpy(), "task": INSTRUCTION}
                if args_cli.crop_size > 0:
                    # privileged: the crop window follows the simulator's true cube position
                    request["crop"] = oracle_crops(base_env, cube_position(obs), args_cli.crop_size)[0].cpu().numpy()
                connection.send(request)
                chunk = torch.from_numpy(connection.recv()).to(device)
            return chunk[:, t % args_cli.replan]

        source = f"smolvla server on port {args_cli.server_port}"
    else:
        student = load_student(args_cli.student, device)
        act = lambda obs, t: student(images(), student_state(obs))  # noqa: E731
        source = os.path.abspath(args_cli.student)

    recorder = EpisodeEndRecorder(base_env)
    obs, _ = env.get_observations()
    views = []
    with torch.inference_mode():
        for t in range(base_env.max_episode_length + 5):
            if args_cli.save_view and t in (0, 60, 240):
                views.append(images()[0].cpu())
            obs, _, _, _ = env.step(act(obs, t))
            if bool(recorder.done.all()):
                break
    if not bool(recorder.done.all()):
        raise RuntimeError("Some environments never finished their episode.")

    dist, height = recorder.final_dist.cpu(), recorder.final_height.cpu()
    summary = {
        "policy": args_cli.policy,
        "checkpoint": source,
        "task": vision_env.TASK,
        "num_episodes": len(dist),
        "threshold_m": args_cli.threshold,
        "success_rate": (dist < args_cli.threshold).float().mean().item(),
        "final_dist_mean_m": dist.mean().item(),
        "final_dist_median_m": dist.median().item(),
        "final_dist_max_m": dist.max().item(),
        "success_rate_5cm": (dist < 0.05).float().mean().item(),
        "lifted_rate": (height > LIFT_HEIGHT).float().mean().item(),
        "dropped_rate": (~recorder.timed_out).float().mean().item(),
        "seed": args_cli.seed,
        "light_scale": args_cli.light_scale,
        "cube_color": args_cli.cube_color,
        "table_color": args_cli.table_color,
        "camera_shift_m": args_cli.camera_shift,
        "image_size": args_cli.image_size,
    }
    if args_cli.policy == "smolvla":
        summary["replan_steps"] = args_cli.replan
        if args_cli.oracle_state:
            summary["oracle_state"] = "true cube position appended to the state, not available to a real robot"
        if args_cli.crop_size > 0:
            summary["oracle_crop"] = {"crop_size": args_cli.crop_size, "render_size": args_cli.crop_render_size,
                                      "note": "crop centered on the simulator's true cube position, not a real detector"}
        connection.close()
    if views:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.save_view)), exist_ok=True)
        imageio.imwrite(args_cli.save_view, torch.cat(views, dim=1).numpy())
    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args_cli.out_dir, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "steps", "timed_out", "final_dist_m", "final_cube_height_m"])
        for i in range(len(dist)):
            writer.writerow([i, int(recorder.steps[i]), bool(recorder.timed_out[i]), float(dist[i]), float(height[i])])
    print("[eval_student] SUMMARY " + json.dumps(summary, indent=2))
    print(f"[eval_student] wrote {args_cli.out_dir}/summary.json and episodes.csv")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
