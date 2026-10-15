"""Offline check: SmolVLA and the CNN student against the teacher's actions on validation frames.

Both models see the same frames: the validation episodes of the raw HDF5 files (the ones the CNN
student held out, and the last 125 episodes of the LeRobot export SmolVLA held out). Errors are
reported in the CNN student's normalized action units, so the two models are on one scale.

    conda activate lerobot-gpu
    python scripts/offline_action_check.py --smolvla logs/smolvla/<run>/checkpoints/last/pretrained_model \
        --student logs/student/bc_more_teacher/student.pt --out results/offline_action_check.json
"""

import argparse
import json

import h5py
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from student_policy import load_student

parser = argparse.ArgumentParser(description="Offline action error of SmolVLA and the CNN student.")
parser.add_argument("--smolvla", type=str, required=True)
parser.add_argument("--student", type=str, required=True)
parser.add_argument("--data", nargs="+", default=["datasets/lift_vision_raw.h5", "datasets/lift_vision_teacher_extra.h5"])
parser.add_argument("--stride", type=int, default=5, help="Use every n-th validation frame.")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--out", type=str, default="results/offline_action_check.json")
args = parser.parse_args()
torch.manual_seed(0)

# validation frames, plus the teacher's next actions in the same episode for the chunk comparison
images, states, actions, frame_index, episode = [], [], [], [], []
for file_id, path in enumerate(args.data):
    with h5py.File(path) as f:
        is_val = f["episode_is_val"][:][f["episode_index"][:]]
        ids = np.nonzero(is_val)[0]
        images.append(f["image"][ids[0]:ids[-1] + 1][is_val[ids[0]:ids[-1] + 1]])
        states.append(f["state"][:][ids])
        actions.append(f["action"][:][ids])
        frame_index.append(f["frame_index"][:][ids])
        episode.append(f["episode_index"][:][ids] + 100000 * file_id)
images, states, actions = np.concatenate(images), np.concatenate(states), np.concatenate(actions)
frame_index, episode = np.concatenate(frame_index), np.concatenate(episode)
query = np.arange(0, len(states), args.stride)
print(f"[offline_check] {len(states)} validation frames, {len(np.unique(episode))} episodes; querying {len(query)}", flush=True)

student = load_student(args.student, args.device)
action_std = student.action_std.cpu().numpy()

policy = SmolVLAPolicy.from_pretrained(args.smolvla).to(args.device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.smolvla, preprocessor_overrides={"device_processor": {"device": args.device}}
)
chunk_size = policy.config.chunk_size
instruction = "pick up the cube and move it to the target"

student_pred, smolvla_chunks = [], []
with torch.no_grad():
    for batch in np.array_split(query, max(1, len(query) // args.batch_size)):
        image = torch.from_numpy(images[batch]).to(args.device)
        state = torch.from_numpy(states[batch]).to(args.device)
        student_pred.append(student(image, state).cpu().numpy())
        observation = {"observation.images.front": image.permute(0, 3, 1, 2).float() / 255.0, "observation.state": state,
                       "task": [instruction] * len(batch)}
        chunk = postprocessor(policy.predict_action_chunk(preprocessor(observation)))
        smolvla_chunks.append(chunk.float().cpu().numpy())
student_pred, smolvla_chunks = np.concatenate(student_pred), np.concatenate(smolvla_chunks)
teacher = actions[query]


def errors(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Normalized MSE overall, for the 7 arm joints and the gripper, plus gripper sign agreement."""
    squared = ((prediction - target) / action_std) ** 2
    return {"mse": float(squared.mean()), "mse_arm": float(squared[:, :7].mean()), "mse_gripper": float(squared[:, 7].mean()),
            "gripper_sign_agreement": float((np.sign(prediction[:, 7]) == np.sign(target[:, 7])).mean())}


# by phase of the episode (frame index), which separates reaching, grasping and carrying
phases = {"frames_0_49": (0, 50), "frames_50_99": (50, 100), "frames_100_249": (100, 250)}
result = {"num_val_frames": int(len(states)), "num_query_frames": int(len(query)), "stride": args.stride,
          "units": "MSE in the CNN student's normalized action units (per-dimension teacher action std)",
          "cnn_student": errors(student_pred, teacher), "smolvla_first_action": errors(smolvla_chunks[:, 0], teacher)}
for name, (low, high) in phases.items():
    mask = (frame_index[query] >= low) & (frame_index[query] < high)
    result[f"cnn_student_{name}"] = errors(student_pred[mask], teacher[mask])
    result[f"smolvla_first_action_{name}"] = errors(smolvla_chunks[mask, 0], teacher[mask])

# SmolVLA's chunk step k against the teacher's action k steps later (same episode only)
horizon = {}
for k in (0, 9, 24, 49):
    later = query + k
    valid = (later < len(states))
    valid[valid] &= episode[later[valid]] == episode[query[valid]]
    horizon[f"step_{k}"] = errors(smolvla_chunks[valid, k], actions[later[valid]])["mse"]
result["smolvla_chunk_mse_by_step"] = horizon
result["teacher_action_std"] = action_std.tolist()

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
