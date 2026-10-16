"""Collect camera demonstrations of the lift-cube task from the RL teacher.

The teacher acts on its usual state observation. Gaussian noise is added to the action that is
executed, for state diversity, but the teacher's clean action is what gets recorded as the label.

With --actor student (DAgger), the vision student drives the robot instead, and the teacher only
labels the states the student visits.

With --randomize, every round of episodes gets new visuals: one light intensity for the round
(log-uniform in --light_range), and per environment a random cube color and table color from the
training palette and a camera offset in a random direction with length up to --max_camera_shift.
One environment in five keeps the original cube look, and another one in five the original table.
Every step stores the camera image, the student's state input, the clean action and (never used
by the student) the cube position. Output is one lossless HDF5 file; export_lerobot.py converts it
to a LeRobot dataset.

    python scripts/collect_lift_vision.py --checkpoint <model.pt> --out datasets/lift_vision_raw.h5 \\
        --num_envs 125 --episodes_per_env 4 --headless --device cuda:0

Use --probe_steps N to only time N steps and save sample images, without writing a dataset.

With --crop_size C, a second camera at the same pose renders at --crop_render_size, and every frame
also stores a C x C crop of it centered on the cube's TRUE position projected into the camera (an
oracle crop, standing in for a perfect detector). --no_compression writes the HDF5 file
uncompressed, which is faster for a temporary file in RAM (/dev/shm) that is exported right away.
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
parser.add_argument("--randomize", action="store_true", help="Randomize lighting, colors and camera per episode.")
parser.add_argument("--light_range", type=float, nargs=2, default=[0.5, 2.0])
parser.add_argument("--max_camera_shift", type=float, default=0.10, help="[m]")
parser.add_argument("--probe_dir", type=str, default="probe")
parser.add_argument("--crop_size", type=int, default=0, help="Also store an oracle crop of this size around the cube (0 = off).")
parser.add_argument("--crop_render_size", type=int, default=512, help="Resolution of the camera the crop is taken from.")
parser.add_argument("--no_compression", action="store_true", help="Write the HDF5 file uncompressed.")
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
from student_policy import load_student
from vision_env import ACTION_DIM, STATE_DIM, VisualRandomizer, cube_position, get_images, make_vision_env, oracle_crops, student_state

# Training palette for random colors: hue below this bound (red through blue). Hues from it up to 1
# (purple, magenta) are never used in training, so they can serve as held-out colors.
TRAIN_MAX_HUE = 0.7


def sample_color(rng) -> tuple:
    import colorsys

    return colorsys.hsv_to_rgb(rng.uniform(0.0, TRAIN_MAX_HUE), rng.uniform(0.0, 1.0), rng.uniform(0.15, 1.0))

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(vision_env.TASK, args_cli)
    env = make_vision_env(args_cli.num_envs, args_cli.device, args_cli.seed, image_size=args_cli.image_size, randomizable=args_cli.randomize,
                           crop_render_size=args_cli.crop_render_size if args_cli.crop_size > 0 else 0)
    base_env = env.unwrapped
    num_envs, device = base_env.num_envs, base_env.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    teacher = runner.get_inference_policy(device=device)

    student = None
    if args_cli.actor == "student":
        student = load_student(args_cli.student, device)

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
    crop = args_cli.crop_size
    crops = torch.zeros(num_steps, num_envs, crop, crop, 3, dtype=torch.uint8) if crop > 0 else None
    crop_centers = torch.zeros(num_steps, num_envs, 2) if crop > 0 else None

    import numpy as np

    rng = np.random.default_rng(args_cli.seed)
    randomizer = VisualRandomizer(base_env) if args_cli.randomize else None
    episode_length = base_env.max_episode_length
    # visuals used in each round, per environment (NaN color = original look)
    num_rounds = -(-num_steps // episode_length)
    round_light = np.ones(num_rounds, dtype=np.float32)
    round_cube = np.full((num_rounds, num_envs, 3), np.nan, dtype=np.float32)
    round_table = np.full((num_rounds, num_envs, 3), np.nan, dtype=np.float32)
    round_shift = np.zeros((num_rounds, num_envs, 3), dtype=np.float32)

    obs, _ = env.get_observations()
    start = time.time()
    with torch.inference_mode():
        for t in range(num_steps):
            if randomizer is not None and t % episode_length == 0:
                r = t // episode_length
                round_light[r] = np.exp(rng.uniform(*np.log(args_cli.light_range)))
                cubes = [sample_color(rng) for _ in range(num_envs)]
                tables = [sample_color(rng) for _ in range(num_envs)]
                directions = rng.normal(size=(num_envs, 3))
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                round_shift[r] = directions * rng.uniform(0.0, args_cli.max_camera_shift, size=(num_envs, 1))
                for i in range(num_envs):
                    if randomizer.recolorable("object", i):
                        round_cube[r, i] = cubes[i]
                    if randomizer.recolorable("table", i):
                        round_table[r, i] = tables[i]
                randomizer.apply(float(round_light[r]), cubes, tables, torch.from_numpy(round_shift[r]))
                if t > 0:
                    # every environment starts a fresh episode under the new visuals
                    dones[t - 1] = True
                base_env.reset()
                obs, _ = env.get_observations()
            clean_action = teacher(obs)
            image = get_images(base_env)
            driver_action = clean_action if student is None else student(image, student_state(obs))
            noisy_action = driver_action + torch.randn_like(driver_action) * args_cli.action_noise
            images[t] = image.cpu()
            states[t] = student_state(obs).cpu()
            actions[t] = clean_action.cpu()
            executed[t] = noisy_action.cpu()
            cube_pos[t] = cube_position(obs).cpu()
            if crops is not None:
                crop_images, centers = oracle_crops(base_env, cube_position(obs), crop)
                crops[t], crop_centers[t] = crop_images.cpu(), centers.cpu()
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

        if randomizer is not None:
            print(f"[collect] PROBE: round light scales {round_light.round(2).tolist()}")

        os.makedirs(args_cli.probe_dir, exist_ok=True)
        for t in sorted({0, num_steps // 2, num_steps - 1} | ({episode_length, episode_length + 5} if num_steps > episode_length + 5 else set())):
            strip = torch.cat([images[t, i] for i in range(min(4, num_envs))], dim=1).numpy()
            imageio.imwrite(os.path.join(args_cli.probe_dir, f"step_{t:03d}.png"), strip)
            if crops is not None:
                strip = torch.cat([crops[t, i] for i in range(min(4, num_envs))], dim=1).numpy()
                imageio.imwrite(os.path.join(args_cli.probe_dir, f"crop_{t:03d}.png"), strip)
        full_steps = args_cli.episodes_per_env * base_env.max_episode_length
        print(f"[collect] PROBE: full collection of {full_steps} steps would take about {full_steps / steps_per_s / 60:.1f} min")
        print(f"[collect] PROBE: image dtype/shape {tuple(images.shape[2:])}, mean pixel {images.float().mean():.1f}")
        env.close()
        return

    write_dataset(images, states, actions, executed, cube_pos, goal_dist, dones,
                  (round_light, round_cube, round_table, round_shift) if randomizer is not None else None,
                  (crops, crop_centers) if crops is not None else None)
    env.close()


def write_dataset(images, states, actions, executed, cube_pos, goal_dist, dones, visuals=None, oracle=None):
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
    compression = {} if args_cli.no_compression else {"compression": "gzip", "compression_opts": 4}
    with h5py.File(args_cli.out, "w") as f:
        image_ds = f.create_dataset("image", (total, size, size, 3), dtype="uint8", chunks=(256, size, size, 3), **compression)
        arrays = {"state": states, "action": actions, "executed_action": executed, "cube_position": cube_pos}
        if oracle is not None:
            crops, crop_centers = oracle
            crop = crops.shape[2]
            crop_ds = f.create_dataset("crop", (total, crop, crop, 3), dtype="uint8", chunks=(64, crop, crop, 3), **compression)
            arrays["crop_center"] = crop_centers
        float_ds = {k: f.create_dataset(k, (total, v.shape[-1]), dtype="float32") for k, v in arrays.items()}
        episode_index = np.zeros(total, dtype=np.int64)
        frame_index = np.zeros(total, dtype=np.int64)
        final_dist = np.zeros(len(spans), dtype=np.float32)
        cursor = 0
        for episode, (env_id, begin, end) in enumerate(spans):
            n = end - begin
            image_ds[cursor : cursor + n] = images[begin:end, env_id].numpy()
            if oracle is not None:
                crop_ds[cursor : cursor + n] = crops[begin:end, env_id].numpy()
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
        if visuals is not None:
            # the visuals each episode was recorded under (cube/table color NaN = original look)
            light, cube, table, shift = visuals
            rounds = [begin // 250 for _, begin, _ in spans]
            envs = [env_id for env_id, _, _ in spans]
            f.create_dataset("episode_light_scale", data=light[rounds])
            f.create_dataset("episode_cube_color", data=cube[rounds, envs])
            f.create_dataset("episode_table_color", data=table[rounds, envs])
            f.create_dataset("episode_camera_shift", data=shift[rounds, envs])
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
            "randomized_visuals": bool(visuals is not None),
            "light_range": args_cli.light_range,
            "max_camera_shift_m": args_cli.max_camera_shift,
            "keep_original": "env index % 5 == 0 keeps the original cube, == 2 the original table",
            "train_max_hue": TRAIN_MAX_HUE,
            "crop_size": args_cli.crop_size,
            "crop_render_size": args_cli.crop_render_size if args_cli.crop_size > 0 else 0,
            "crop": "oracle crop centered on the true cube position projected into the camera" if args_cli.crop_size > 0 else "",
        })
    print(
        f"[collect] wrote {args_cli.out}: {len(spans)} episodes ({int(is_val.sum())} val), {total} frames,"
        f" {os.path.getsize(args_cli.out) / 1e9:.2f} GB;"
        f" {args_cli.actor} (noise std {args_cli.action_noise:g}) within 2 cm at the last frame in {(final_dist < 0.02).mean() * 100:.0f}% of episodes"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
