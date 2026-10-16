"""Serve a fine-tuned SmolVLA policy to the Isaac Lab eval over a local socket.

Isaac Lab (conda env WL) and LeRobot with GPU torch (conda env lerobot-gpu) cannot share one Python
process, so the policy runs here and eval_student.py --policy smolvla sends it observations.

    conda activate lerobot-gpu
    python scripts/smolvla_server.py --checkpoint logs/smolvla/<run>/checkpoints/last/pretrained_model

Each request is a dict with "image" (B, H, W, 3) uint8, "state" (B, 21) float32 and "task" (str),
plus "crop" (B, C, C, 3) uint8 for a policy trained with the oracle crop as a second camera; the reply is the action chunk (B, chunk_size, 8) float32 in environment units. Stops on "close".
"""

import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

parser = argparse.ArgumentParser(description="SmolVLA policy server.")
parser.add_argument("--checkpoint", type=str, required=True, help="A pretrained_model directory written by lerobot-train.")
parser.add_argument("--port", type=int, default=47310)
parser.add_argument("--authkey", type=str, default="autorl-demo")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.checkpoint, preprocessor_overrides={"device_processor": {"device": args.device}}
)
# the camera keys the policy was trained on (after the training rename map)
image_keys = list(policy.config.image_features)
print(f"[smolvla_server] loaded {args.checkpoint}; image keys {image_keys}; chunk {policy.config.chunk_size}", flush=True)


def to_pixels(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).to(args.device).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def predict(request: dict) -> np.ndarray:
    states = torch.from_numpy(request["state"]).to(args.device).float()
    observation = {"observation.images.front": to_pixels(request["image"]), "observation.state": states,
                   "task": [request["task"]] * len(states)}
    if "crop" in request:
        observation["observation.images.crop"] = to_pixels(request["crop"])
    batch = preprocessor(observation)
    # the rename step in the preprocessor maps the dataset's camera keys to the policy's keys
    chunk = policy.predict_action_chunk(batch)  # (B, chunk_size, action_dim), normalized
    return postprocessor(chunk).float().cpu().numpy()


with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener:
    print(f"[smolvla_server] listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        with listener.accept() as connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                if request == "close":
                    print("[smolvla_server] closing", flush=True)
                    raise SystemExit(0)
                start = time.time()
                chunk = predict(request)
                connection.send(chunk)
                print(f"[smolvla_server] batch {len(chunk)}: {1000 * (time.time() - start):.0f} ms", flush=True)
