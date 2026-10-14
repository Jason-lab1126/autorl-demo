"""Export the raw demonstrations to a LeRobot dataset, using LeRobot's own writer, and validate it.

Run this in the separate ``lerobot`` conda environment (it does not need Isaac Lab or a GPU):

    conda activate lerobot
    python scripts/export_lerobot.py --raw datasets/lift_vision_raw.h5 --root datasets/lift_cube_franka_lerobot

Training episodes are written first and validation episodes last, and the split is recorded in
meta/info.json. After writing, the dataset is loaded back with LeRobotDataset and compared with
the raw file.
"""

import argparse
import json
import os

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description="Export demonstrations to LeRobot format.")
parser.add_argument("--raw", type=str, default="datasets/lift_vision_raw.h5")
parser.add_argument("--root", type=str, default="datasets/lift_cube_franka_lerobot")
parser.add_argument("--repo_id", type=str, default="Jason-lab1126/lift_cube_franka_vision")
parser.add_argument("--video_backend", type=str, default="pyav", help="Decoder used when loading the dataset back.")
parser.add_argument("--validate_only", action="store_true", help="Skip writing; only load and check an existing export.")
args = parser.parse_args()

ACTION_NAMES = [f"arm_joint_{i}" for i in range(7)] + ["gripper"]
IMAGE_KEY = "observation.images.front"

with h5py.File(args.raw) as f:
    instruction, fps = f.attrs["instruction"], int(f.attrs["fps"])
    state_names = f.attrs["state_names"].split(",")
    lengths, is_val = f["episode_length"][:], f["episode_is_val"][:]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    size = f["image"].shape[1]
    order = list(np.flatnonzero(~is_val)) + list(np.flatnonzero(is_val))
    num_train = int((~is_val).sum())

    if not args.validate_only:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.root,
            fps=fps,
            robot_type="franka_panda",
            features={
                IMAGE_KEY: {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]},
                "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
                # privileged cube position in the robot base frame; the vision student never reads it
                "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": ["cube_x", "cube_y", "cube_z"]},
                "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
            },
            image_writer_threads=8,
        )
        for count, episode in enumerate(order):
            begin, end = starts[episode], starts[episode] + lengths[episode]
            images, states = f["image"][begin:end], f["state"][begin:end]
            cubes, actions = f["cube_position"][begin:end], f["action"][begin:end]
            for i in range(end - begin):
                dataset.add_frame({
                    IMAGE_KEY: images[i],
                    "observation.state": states[i],
                    "observation.environment_state": cubes[i],
                    "action": actions[i],
                    "task": instruction,
                })
            dataset.save_episode()
            if (count + 1) % 50 == 0:
                print(f"[export_lerobot] {count + 1}/{len(order)} episodes", flush=True)
        dataset.finalize()

        info_path = os.path.join(args.root, "meta", "info.json")
        with open(info_path) as fp:
            info = json.load(fp)
        info["splits"] = {"train": f"0:{num_train}", "val": f"{num_train}:{len(order)}"}
        with open(info_path, "w") as fp:
            json.dump(info, fp, indent=4)

    # validate: load the dataset back with LeRobot and compare against the raw file
    loaded = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
    report = {
        "codebase_version": loaded.meta.info["codebase_version"],
        "num_episodes": loaded.num_episodes,
        "num_frames": loaded.num_frames,
        "fps": loaded.fps,
        "splits": loaded.meta.info["splits"],
        "features": {k: v["shape"] for k, v in loaded.features.items()},
        "task": instruction,
        "video_backend": args.video_backend,
    }
    assert loaded.num_episodes == len(order) and loaded.num_frames == int(lengths.sum())
    rng = np.random.default_rng(0)
    state_err, action_err, image_err = 0.0, 0.0, []
    for index in rng.choice(loaded.num_frames, 200, replace=False):
        sample = loaded[int(index)]
        episode = order[int(sample["episode_index"])]
        raw_index = starts[episode] + int(sample["frame_index"])
        state_err = max(state_err, float(np.abs(sample["observation.state"].numpy() - f["state"][raw_index]).max()))
        action_err = max(action_err, float(np.abs(sample["action"].numpy() - f["action"][raw_index]).max()))
        decoded = sample[IMAGE_KEY].permute(1, 2, 0).numpy() * 255.0
        image_err.append(float(np.abs(decoded - f["image"][raw_index].astype(np.float32)).mean()))
        assert sample["task"] == instruction
    report.update({
        "checked_frames": 200,
        "max_abs_state_error": state_err,
        "max_abs_action_error": action_err,
        "mean_abs_pixel_error_0_255": float(np.mean(image_err)),
    })
    assert state_err < 1e-5 and action_err < 1e-5, "state/action mismatch after export"
    with open(os.path.join(args.root, "validation_report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    print("[export_lerobot] VALIDATED " + json.dumps(report, indent=2))
