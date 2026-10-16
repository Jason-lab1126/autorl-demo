"""Offline check for SmolVLA trained with the oracle crop: does it use the crop?

On the held-out episodes of the two-camera LeRobot export (every --stride-th frame, the same frames
as offline_action_check.py), compares the policy's first action with the teacher's, once with the
true crop, once with the crop of a different frame (shuffled) and once with a black crop. Errors are
in the CNN student's normalized action units, like offline_action_check.py.

    conda activate lerobot-gpu
    python scripts/offline_crop_check.py --smolvla logs/smolvla/smolvla_1250_oracle_crop/checkpoints/last/pretrained_model
"""

import argparse
import json

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from student_policy import load_student

parser = argparse.ArgumentParser(description="Does SmolVLA use the oracle crop?")
parser.add_argument("--smolvla", type=str, required=True)
parser.add_argument("--root", type=str, default="datasets/lift_cube_franka_lerobot_1250_oracle_crop")
parser.add_argument("--student", type=str, default="logs/student/bc_more_teacher/student.pt", help="Only for its action std.")
parser.add_argument("--stride", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--out", type=str, default="results/offline_crop_check.json")
args = parser.parse_args()
torch.manual_seed(0)

dataset = LeRobotDataset("local/lift_cube_franka_oracle_crop", root=args.root, video_backend="pyav")
first_val = int(dataset.meta.info["splits"]["val"].split(":")[0])
val_frames = np.nonzero(np.asarray(dataset.hf_dataset["episode_index"]) >= first_val)[0]
query = val_frames[::args.stride]
print(f"[offline_crop_check] {len(val_frames)} validation frames, querying {len(query)}", flush=True)

fronts, crops, states, actions, frame_index = [], [], [], [], []
for i in query:
    sample = dataset[int(i)]
    fronts.append(sample["observation.images.front"])
    crops.append(sample["observation.images.crop"])
    states.append(sample["observation.state"])
    actions.append(sample["action"].numpy())
    frame_index.append(int(sample["frame_index"]))
fronts, crops, states = torch.stack(fronts), torch.stack(crops), torch.stack(states)
actions, frame_index = np.stack(actions), np.array(frame_index)
action_std = load_student(args.student, "cpu").action_std.numpy()

policy = SmolVLAPolicy.from_pretrained(args.smolvla).to(args.device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.smolvla, preprocessor_overrides={"device_processor": {"device": args.device}}
)
instruction = "pick up the cube and move it to the target"
shuffled = crops[torch.randperm(len(crops))]
variants = {"true_crop": crops, "shuffled_crop": shuffled, "black_crop": torch.zeros_like(crops)}

predictions = {}
with torch.no_grad():
    for name, crop_images in variants.items():
        out = []
        for batch in np.array_split(np.arange(len(query)), max(1, len(query) // args.batch_size)):
            observation = {"observation.images.front": fronts[batch].to(args.device),
                           "observation.images.crop": crop_images[batch].to(args.device),
                           "observation.state": states[batch].to(args.device), "task": [instruction] * len(batch)}
            out.append(postprocessor(policy.predict_action_chunk(preprocessor(observation)))[:, 0].float().cpu().numpy())
        predictions[name] = np.concatenate(out)
        print(f"[offline_crop_check] {name} done", flush=True)


def mse(prediction, mask):
    return float((((prediction - actions) / action_std) ** 2)[mask].mean())


phases = {"all_frames": (0, 250), "frames_0_49": (0, 50), "frames_50_99": (50, 100), "frames_100_249": (100, 250)}
result = {"num_query_frames": int(len(query)), "stride": args.stride,
          "units": "MSE in the CNN student's normalized action units (per-dimension teacher action std)",
          "images": "decoded from the LeRobot videos"}
for name, prediction in predictions.items():
    result[name] = {phase: mse(prediction, (frame_index >= low) & (frame_index < high)) for phase, (low, high) in phases.items()}
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
