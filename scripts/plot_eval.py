"""Plot success rate vs. perturbation level from the eval matrix, one line per checkpoint.

Reads results/eval_matrix_cells.csv (written by eval_lift.py) and writes one PNG per swept
perturbation into the same directory. Error bars are 95% Wilson intervals. Needs no simulator:

    python scripts/plot_eval.py [--results_dir results]
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# perturbation -> (x value of the unperturbed baseline, x-axis label, title)
SWEEPS = {
    "obs_noise": (0.0, "Observation noise multiplier", "Success vs. observation noise"),
    "cube_mass": (1.0, "Cube mass multiplier", "Success vs. cube mass"),
}
# fixed colour and marker per line, in checkpoint order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, as (low, high) fractions."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results_dir", type=str, default="results")
args = parser.parse_args()

with open(os.path.join(args.results_dir, "eval_matrix_cells.csv")) as f:
    rows = list(csv.DictReader(f))
iters = sorted({int(r["iter"]) for r in rows})
if len(iters) > len(COLORS):
    raise ValueError(f"At most {len(COLORS)} checkpoints can be plotted, got {len(iters)}.")
num_episodes = rows[0]["num_episodes"]

for name, (baseline_x, xlabel, title) in SWEEPS.items():
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = []
    num_lines = len(iters)
    for index, (it, color, marker) in enumerate(zip(iters, COLORS, MARKERS)):
        points = [(baseline_x, r) for r in rows if int(r["iter"]) == it and r["perturbation"] == "none"]
        points += [(float(r["value"]), r) for r in rows if int(r["iter"]) == it and r["perturbation"] == name]
        points.sort(key=lambda p: p[0])
        xs = [x for x, _ in points]
        ys = [100.0 * float(r["success_rate"]) for _, r in points]
        # 95% Wilson interval per point; lines are nudged apart sideways so overlapping bars stay readable
        bounds = [wilson_interval(float(r["success_rate"]) * int(r["num_episodes"]), int(r["num_episodes"])) for _, r in points]
        yerr = [[max(0.0, y - 100.0 * lo) for y, (lo, _) in zip(ys, bounds)], [max(0.0, 100.0 * hi - y) for y, (_, hi) in zip(ys, bounds)]]
        dodge = (index - (num_lines - 1) / 2) * 0.008 * (max(xs) - min(xs))
        plot_xs = [x + dodge for x in xs]
        ax.errorbar(
            plot_xs, ys, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.2, capsize=3,
            capthick=1.2, clip_on=False, zorder=2,
        )
        ax.plot(
            plot_xs, ys, color=color, marker=marker, linewidth=2, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"iter {it}", clip_on=False, zorder=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"x{x:g}" for x in xs])
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("Success rate [%]", color=MUTED)
    ax.set_title(
        f"{title}\nfinal distance < 2 cm, {num_episodes} episodes per point, bars: 95% Wilson interval",
        color=INK, fontsize=11, loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    legend = ax.legend(title="Checkpoint", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.setp(legend.get_texts(), color=INK)
    legend.get_title().set_color(MUTED)
    out_path = os.path.join(args.results_dir, f"success_vs_{name}.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot_eval] wrote {out_path}")
