"""Collect camera demonstrations of the lift-cube task from the RL teacher.

The teacher acts on its usual state observation. Gaussian noise is added to the action that is
executed, for state diversity, but the teacher's clean action is what gets recorded as the label.

With --actor student (DAgger), the vision student drives the robot instead, and the teacher only
labels the states the student visits.
Every step stores the camera image, the student's state input, the clean action and (never used
by the student) the cube position. Output is one lossless HDF5 file; export_lerobot.py converts it
to a LeRobot dataset.

    python scripts/collect_lift_vision.py --checkpoint <model.pt> --out datasets/lift_vision_raw.h5 \\
        --num_envs 125 --episodes_per_env 4 --headless --device cuda:0

Use --probe_steps N to only time N steps and save sample images, without writing a dataset.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Collect camera demonstrations from the RL teacher.")
parser.add_argument("--out", type=str, default="datasets/lift_vision_raw.h5")
parser.add_argument("--num_envs", type=int, default=125)
parser.add_argument("--episodes_per_env", type=int, default=4)
parser.add_argument("--image_size", type=int, default=84)
parser.add_argument("--action_noise", type=float, default=0.1, help="Std of the noise on the executed action.")
parser.add_argument("--actor", type=str, default="teacher", choices=["teacher", "student"], help="Who drives the robot.")
parser.add_argument("--student", type=str, default=None, help="Student checkpoint, for --actor student.")
parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of episodes held out for validation.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--probe_steps", type=int, default=0, help="Only time this many steps and save sample images.")
parser.add_argument("--probe_dir", type=str, default="probe")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch

from rsl_rl.runners import OnPolicyRunner

import vision_env
from student_policy import StudentPolicy
from vision_env import ACTION_DIM, STATE_DIM, cube_position, get_images, make_vision_env, student_state

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
    env = make_vision_env(args_cli.num_envs, args_cli.device, args_cli.seed, image_size=args_cli.image_size)
    base_env = env.unwrapped
    num_envs, device = base_env.num_envs, base_env.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    teacher = runner.get_inference_policy(device=device)

    student = None
    if args_cli.actor == "student":
        checkpoint = torch.load(args_cli.student, map_location=device)
        student = StudentPolicy(checkpoint["state_dim"], checkpoint["action_dim"]).to(device)
        student.load_state_dict(checkpoint["model"])
        student.eval()

    probing = args_cli.probe_steps > 0
    num_steps = args_cli.probe_steps if probing else args_cli.episodes_per_env * base_env.max_episode_length
    size = args_cli.image_size
    images = torch.zeros(num_steps, num_envs, size, size, 3, dtype=torch.uint8)
    states = torch.zeros(num_steps, num_envs, STATE_DIM)
    actions = torch.zeros(num_steps, num_envs, ACTION_DIM)
    executed = torch.zeros(num_steps, num_envs, ACTION_DIM)
    cube_pos = torch.zeros(num_steps, num_envs, 3)
    goal_dist = torch.zeros(num_steps, num_envs)
    dones = torch.zeros(num_steps, num_envs, dtype=torch.bool)

    obs, _ = env.get_observations()
    start = time.time()
    with torch.inference_mode():
        for t in range(num_steps):
            clean_action = teacher(obs)
            image = get_images(base_env)
            driver_action = clean_action if student is None else student(image, student_state(obs))
            noisy_action = driver_action + torch.randn_like(driver_action) * args_cli.action_noise
            images[t] = image.cpu()
            states[t] = student_state(obs).cpu()
            actions[t] = clean_action.cpu()
            executed[t] = noisy_action.cpu()
            cube_pos[t] = cube_position(obs).cpu()
            goal_dist[t] = torch.norm(cube_position(obs) - obs[:, 21:24], dim=1).cpu()
            obs, _, done, _ = env.step(noisy_action)
            dones[t] = done.bool().cpu()
            if t == 9:
                start = time.time()  # the first steps include one-off renderer warm-up
    elapsed = time.time() - start
    steps_per_s = (num_steps - 10) / elapsed
    print(f"[collect] {num_envs} envs: {steps_per_s:.1f} steps/s = {steps_per_s * num_envs:.0f} frames/s")

    if probing:
        import imageio.v2 as imageio

        os.makedirs(args_cli.probe_dir, exist_ok=True)
        for t in sorted({0, num_steps // 2, num_steps - 1}):
            strip = torch.cat([images[t, i] for i in range(min(4, num_envs))], dim=1).numpy()
            imageio.imwrite(os.path.join(args_cli.probe_dir, f"step_{t:03d}.png"), strip)
        full_steps = args_cli.episodes_per_env * base_env.max_episode_length
        print(f"[collect] PROBE: full collection of {full_steps} steps would take about {full_steps / steps_per_s / 60:.1f} min")
        print(f"[collect] PROBE: image dtype/shape {tuple(images.shape[2:])}, mean pixel {images.float().mean():.1f}")
        env.close()
        return

    write_dataset(images, states, actions, executed, cube_pos, goal_dist, dones)
    env.close()


def write_dataset(images, states, actions, executed, cube_pos, goal_dist, dones):
    import h5py
    import numpy as np

    # cut every environment's stream into complete episodes; a trailing partial episode is dropped
    spans = []
    for env_id in range(dones.shape[1]):
        begin = 0
        for end in dones[:, env_id].nonzero().squeeze(-1).tolist():
            spans.append((env_id, begin, end + 1))
            begin = end + 1
    lengths = np.array([end - begin for _, begin, end in spans])
    total = int(lengths.sum())

    rng = np.random.default_rng(args_cli.seed)
    is_val = np.zeros(len(spans), dtype=bool)
    is_val[rng.permutation(len(spans))[: max(1, round(args_cli.val_fraction * len(spans)))]] = True

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    size = images.shape[2]
    with h5py.File(args_cli.out, "w") as f:
        image_ds = f.create_dataset(
            "image", (total, size, size, 3), dtype="uint8", chunks=(256, size, size, 3), compression="gzip", compression_opts=4
        )
        arrays = {"state": states, "action": actions, "executed_action": executed, "cube_position": cube_pos}
        float_ds = {k: f.create_dataset(k, (total, v.shape[-1]), dtype="float32") for k, v in arrays.items()}
        episode_index = np.zeros(total, dtype=np.int64)
        frame_index = np.zeros(total, dtype=np.int64)
        final_dist = np.zeros(len(spans), dtype=np.float32)
        cursor = 0
        for episode, (env_id, begin, end) in enumerate(spans):
            n = end - begin
            image_ds[cursor : cursor + n] = images[begin:end, env_id].numpy()
            for k, v in arrays.items():
                float_ds[k][cursor : cursor + n] = v[begin:end, env_id].numpy()
            episode_index[cursor : cursor + n] = episode
            frame_index[cursor : cursor + n] = np.arange(n)
            final_dist[episode] = goal_dist[end - 1, env_id]
            cursor += n
        f.create_dataset("episode_index", data=episode_index)
        f.create_dataset("frame_index", data=frame_index)
        f.create_dataset("episode_length", data=lengths)
        f.create_dataset("episode_is_val", data=is_val)
        # cube-to-goal distance at the last recorded frame of the episode (one step before it ends)
        f.create_dataset("episode_last_frame_dist", data=final_dist)
        f.attrs.update({
            "task": vision_env.TASK,
            "instruction": vision_env.INSTRUCTION,
            "fps": vision_env.FPS,
            "teacher_checkpoint": os.path.abspath(args_cli.checkpoint),
            "action_noise_std": args_cli.action_noise,
            "actor": args_cli.actor,
            "actor_student_checkpoint": os.path.abspath(args_cli.student) if args_cli.student else "",
            "state_names": ",".join(vision_env.STATE_NAMES),
            "camera_eye": vision_env.CAMERA_EYE,
            "camera_lookat": vision_env.CAMERA_LOOKAT,
            "camera_focal_length": vision_env.CAMERA_FOCAL_LENGTH,
            "seed": args_cli.seed,
        })
    print(
        f"[collect] wrote {args_cli.out}: {len(spans)} episodes ({int(is_val.sum())} val), {total} frames,"
        f" {os.path.getsize(args_cli.out) / 1e9:.2f} GB;"
        f" {args_cli.actor} (noise std {args_cli.action_noise:g}) within 2 cm at the last frame in {(final_dist < 0.02).mean() * 100:.0f}% of episodes"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
