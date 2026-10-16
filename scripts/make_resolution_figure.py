"""Figure for Part 3: one frame as the policies get it (84x84), as SmolVLA sees it (upscaled to
512x512) and rendered directly at 512x512, with the cube circled in each.

The circle is the recorded cube position projected through the student camera (pinhole model,
same focal length, aperture and pose as vision_env.py). Reads the two files written by
render_resolution_frame.py.

    python scripts/make_resolution_figure.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CUBE_SIZE = 0.048  # edge of the task's cube [m] (DexCube scaled by 0.8)

parser = argparse.ArgumentParser(description="Resolution comparison figure.")
parser.add_argument("--low", type=str, default="results/resolution/frame_84.npz")
parser.add_argument("--high", type=str, default="results/resolution/frame_512.npz")
parser.add_argument("--out", type=str, default="results/resolution_comparison.png")
args = parser.parse_args()


def project(frame, size: int):
    """Pixel position (u right, v down) and edge length in pixels of the cube in a size x size image."""
    eye, lookat, point = frame["eye"], frame["lookat"], frame["cube_position"]
    forward = (lookat - eye) / np.linalg.norm(lookat - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    focal_px = size * float(frame["focal_length"]) / float(frame["horizontal_aperture"])
    offset = point - eye
    depth = offset @ forward
    u = size / 2 + focal_px * (offset @ right) / depth
    v = size / 2 + focal_px * (offset @ down) / depth
    return u, v, focal_px * CUBE_SIZE / depth


low, high = np.load(args.low), np.load(args.high)
assert np.allclose(low["cube_position"], high["cube_position"]), "the two renders show different scenes"
panels = [
    (low["image"], "(a) 84x84: what the policies get"),
    (low["upscaled"], "(b) (a) upscaled to 512x512: what SmolVLA sees"),
    (high["image"], "(c) rendered directly at 512x512"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, (image, title) in zip(axes, panels):
    size = image.shape[0]
    u, v, edge = project(low, size)
    ax.imshow(image, interpolation="nearest")
    ax.add_patch(plt.Circle((u - 0.5, v - 0.5), 1.3 * edge, fill=False, color="#e8a33d", linewidth=2))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.text(0.02, 0.02, f"cube about {edge:.0f} px wide", transform=ax.transAxes, fontsize=9, color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "linewidth": 0, "pad": 2})
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"[make_resolution_figure] wrote {args.out}; cube center at {project(low, 84)[:2]} in the 84x84 frame")
