"""Closed-loop evaluation of the vision student in the camera environment.

Same metric as eval_lift.py: cube-to-goal distance at the END of each episode, one episode per
environment, success if within --threshold. Pass --policy teacher to run the RL teacher in the
same camera environment as a reference.

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
parser.add_argument("--policy", type=str, default="student", choices=["student", "teacher"])
parser.add_argument("--student", type=str, default=None, help="Student checkpoint (student.pt).")
parser.add_argument("--num_envs", type=int, default=100, help="Number of episodes (one per environment).")
parser.add_argument("--threshold", type=float, default=0.02, help="Success radius [m].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="results/student_bc")
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
from student_policy import StudentPolicy
from vision_env import EpisodeEndRecorder, get_images, make_vision_env, student_state

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

LIFT_HEIGHT = 0.04  # same threshold the task's rewards use for "lifted"


def main():
    env = make_vision_env(args_cli.num_envs, args_cli.device, args_cli.seed)
    base_env = env.unwrapped
    device = base_env.device

    if args_cli.policy == "teacher":
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.abspath(args_cli.checkpoint))
        teacher = runner.get_inference_policy(device=device)
        act = lambda obs: teacher(obs)  # noqa: E731
        source = os.path.abspath(args_cli.checkpoint)
    else:
        checkpoint = torch.load(args_cli.student, map_location=device)
        student = StudentPolicy(checkpoint["state_dim"], checkpoint["action_dim"]).to(device)
        student.load_state_dict(checkpoint["model"])
        student.eval()
        act = lambda obs: student(get_images(base_env), student_state(obs))  # noqa: E731
        source = os.path.abspath(args_cli.student)

    recorder = EpisodeEndRecorder(base_env)
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(base_env.max_episode_length + 5):
            obs, _, _, _ = env.step(act(obs))
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
    }
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
