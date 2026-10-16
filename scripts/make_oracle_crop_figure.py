"""Figure for the oracle-crop diagnostic: the two images SmolVLA gets for one held-out frame.

(a) the normal 84x84 view with the crop window drawn in, (b) the 128x128 crop of the 512x512 render,
centered on the cube's true position. Reads results/resolution/oracle_crop_example.npz (the two
images from the LeRobot export and the cube position from the raw file of the same episode).

    python scripts/make_oracle_crop_figure.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# student camera (vision_env.py)
EYE, LOOKAT, FOCAL, APERTURE = np.array([1.1, 0.7, 0.7]), np.array([0.45, 0.0, 0.35]), 26.0, 20.955
RENDER, CROP = 512, 128

parser = argparse.ArgumentParser(description="Oracle crop example figure.")
parser.add_argument("--example", type=str, default="results/resolution/oracle_crop_example.npz")
parser.add_argument("--out", type=str, default="results/oracle_crop_example.png")
args = parser.parse_args()


def project(point, size):
    forward = (LOOKAT - EYE) / np.linalg.norm(LOOKAT - EYE)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    offset = point - EYE
    depth = offset @ forward
    focal_px = size * FOCAL / APERTURE
    return np.array([size / 2 + focal_px * (offset @ right) / depth, size / 2 + focal_px * (offset @ down) / depth])


example = np.load(args.example)
# crop window in the 512 render, as in vision_env.oracle_crops, then scaled to the 84x84 view
top_left = np.clip(np.round(project(example["cube_position"], RENDER)) - CROP // 2, 0, RENDER - CROP)
size = example["front"].shape[0]
scale = size / RENDER

fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
axes[0].imshow(example["front"], interpolation="nearest")
axes[0].add_patch(plt.Rectangle(top_left * scale - 0.5, CROP * scale, CROP * scale, fill=False, color="#e8a33d", linewidth=2))
axes[0].set_title("(a) 84x84 full view, crop window outlined", fontsize=10)
axes[1].imshow(example["crop"], interpolation="nearest")
axes[1].set_title("(b) 128x128 crop of a 512x512 render,\ncentered on the true cube position", fontsize=10)
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"[make_oracle_crop_figure] wrote {args.out}")
