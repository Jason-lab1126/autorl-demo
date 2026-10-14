"""Behavior cloning: train the vision student on the teacher demonstrations (MSE on the action).

Reads the lossless HDF5 file from collect_lift_vision.py, splits train/validation by episode as
recorded in the file, and saves the checkpoint with the best validation loss. Needs no simulator.

    python scripts/train_student.py --data datasets/lift_vision_raw.h5 --out logs/student/bc_500ep
"""

import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_policy import StudentPolicy  # noqa: E402

parser = argparse.ArgumentParser(description="Train the vision student by behavior cloning.")
parser.add_argument("--data", type=str, nargs="+", default=["datasets/lift_vision_raw.h5"], help="One or more HDF5 files.")
parser.add_argument("--out", type=str, default="logs/student/bc")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)

# load everything into memory; images stay uint8 on the CPU and are moved to the GPU per batch
images, states, actions, is_val, num_episodes = [], [], [], [], 0
for path in args.data:
    with h5py.File(path) as f:
        images.append(f["image"][:])
        states.append(f["state"][:])
        actions.append(f["action"][:])
        is_val.append(f["episode_is_val"][:][f["episode_index"][:]])
        num_episodes += len(f["episode_length"])
        print(f"[train_student] {path}: {len(f['episode_length'])} episodes, {len(f['state'])} frames")
images = torch.from_numpy(np.concatenate(images))
states = torch.from_numpy(np.concatenate(states)).to(device)
actions = torch.from_numpy(np.concatenate(actions)).to(device)
is_val = torch.from_numpy(np.concatenate(is_val)).to(device)
train_ids, val_ids = (~is_val).nonzero().squeeze(-1), is_val.nonzero().squeeze(-1)


def image_batch(ids: torch.Tensor) -> torch.Tensor:
    return images[ids.cpu()].to(device, non_blocking=True)

print(f"[train_student] {len(train_ids)} train frames, {len(val_ids)} validation frames")

model = StudentPolicy(states.shape[1], actions.shape[1]).to(device)
model.set_normalization(states[train_ids], actions[train_ids])
steps_per_epoch = len(train_ids) // args.batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)


def evaluate(ids: torch.Tensor) -> tuple[float, float]:
    """Mean squared error on normalized actions and on raw actions."""
    model.eval()
    normalized, raw = 0.0, 0.0
    with torch.no_grad():
        for batch in ids.split(1024):
            prediction = model.forward_normalized(image_batch(batch), states[batch])
            target = model.normalize_action(actions[batch])
            normalized += ((prediction - target) ** 2).mean(1).sum().item()
            raw += (((prediction - target) * model.action_std) ** 2).mean(1).sum().item()
    model.train()
    return normalized / len(ids), raw / len(ids)


rows, best_val, start = [], float("inf"), time.time()
for epoch in range(1, args.epochs + 1):
    running = 0.0
    for batch in train_ids[torch.randperm(len(train_ids), device=device)][: steps_per_epoch * args.batch_size].split(args.batch_size):
        loss = ((model.forward_normalized(image_batch(batch), states[batch]) - model.normalize_action(actions[batch])) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()
    val_mse, val_mse_raw = evaluate(val_ids)
    rows.append({"epoch": epoch, "train_mse": running / steps_per_epoch, "val_mse": val_mse, "val_mse_raw_action": val_mse_raw})
    if val_mse < best_val:
        best_val = val_mse
        torch.save({"model": model.state_dict(), "state_dim": states.shape[1], "action_dim": actions.shape[1], "epoch": epoch, "val_mse": val_mse}, os.path.join(args.out, "student.pt"))
    print(f"[train_student] epoch {epoch:3d}  train {rows[-1]['train_mse']:.4f}  val {val_mse:.4f}  val (raw action) {val_mse_raw:.4f}  ({time.time() - start:.0f}s)", flush=True)

with open(os.path.join(args.out, "training_log.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump({**vars(args), "num_episodes": num_episodes, "train_frames": len(train_ids), "val_frames": len(val_ids), "best_val_mse": best_val, "train_seconds": time.time() - start}, f, indent=2)
print(f"[train_student] best validation MSE {best_val:.4f}; wrote {args.out}/student.pt")
