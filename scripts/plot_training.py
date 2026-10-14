"""Plot training curves (mean episode reward vs. iteration) from the runs' TensorBoard logs.

Each run is given as LABEL=RUN_DIR. Writes the plot and a CSV of the plotted values. Needs no
simulator:

    python scripts/plot_training.py --runs "Baseline=logs/rsl_rl/franka_lift/<run>" --out results/training_curve.png
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

TAG = "Train/mean_reward"
# fixed colour per line, in the order the runs are given
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--runs", type=str, nargs="+", required=True, help="LABEL=RUN_DIR, one per training run.")
parser.add_argument("--out", type=str, default="results/training_curve.png")
args = parser.parse_args()
if len(args.runs) > len(COLORS):
    raise ValueError(f"At most {len(COLORS)} runs can be plotted, got {len(args.runs)}.")

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
curves = {}
for run, color in zip(args.runs, COLORS):
    label, run_dir = run.split("=", 1)
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file in {run_dir}.")
    accumulator = EventAccumulator(event_files[-1], size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(TAG)
    curves[label] = {e.step: e.value for e in events}
    ax.plot([e.step for e in events], [e.value for e in events], color=color, linewidth=2, label=label)

ax.set_xlim(0, max(max(c) for c in curves.values()))
ax.set_ylim(bottom=0)
ax.set_xlabel("Training iteration", color=MUTED)
ax.set_ylabel("Mean episode reward", color=MUTED)
ax.set_title("Training curve\nmean episode reward over the 4096 training environments", color=INK, fontsize=11, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
if len(curves) > 1:
    legend = ax.legend(title="Training run", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.setp(legend.get_texts(), color=INK)
    legend.get_title().set_color(MUTED)
fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)

csv_path = os.path.splitext(args.out)[0] + ".csv"
steps = sorted(set().union(*curves.values()))
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration"] + list(curves))
    for step in steps:
        writer.writerow([step] + [f"{curves[label][step]:.4f}" if step in curves[label] else "" for label in curves])
print(f"[plot_training] wrote {args.out} and {csv_path}")
