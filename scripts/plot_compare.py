"""Compare policies on one perturbation sweep: success rate vs. level, one line per policy.

Each policy is given as LABEL=RESULTS_DIR, where RESULTS_DIR holds an eval_matrix_cells.csv written
by eval_lift.py. Append @ITER to pick a checkpoint other than --iter for that policy. Needs no
simulator:

    python scripts/plot_compare.py --policies "Baseline=results" "Noise-trained=results/obs_noise_dr" \\
        --perturbation obs_noise --iter 1499 --out results/success_vs_obs_noise_compare.png
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
# fixed colour and marker per line, in the order the policies are given
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--policies", type=str, nargs="+", required=True, help="LABEL=RESULTS_DIR[@ITER], one per policy.")
parser.add_argument("--perturbation", type=str, default="obs_noise", choices=list(SWEEPS))
parser.add_argument("--iter", type=int, default=1499, help="Checkpoint iteration to plot, unless a policy names its own.")
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()
if len(args.policies) > len(COLORS):
    raise ValueError(f"At most {len(COLORS)} policies can be plotted, got {len(args.policies)}.")

baseline_x, xlabel, title = SWEEPS[args.perturbation]
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
num_episodes = None
for policy, color, marker in zip(args.policies, COLORS, MARKERS):
    label, results_dir = policy.split("=", 1)
    results_dir, _, policy_iter = results_dir.partition("@")
    policy_iter = int(policy_iter) if policy_iter else args.iter
    with open(os.path.join(results_dir, "eval_matrix_cells.csv")) as f:
        rows = [r for r in csv.DictReader(f) if int(r["iter"]) == policy_iter]
    points = [(baseline_x, r) for r in rows if r["perturbation"] == "none"]
    points += [(float(r["value"]), r) for r in rows if r["perturbation"] == args.perturbation]
    if len(points) < 2:
        raise ValueError(f"{results_dir} has no '{args.perturbation}' sweep for iteration {policy_iter}.")
    points.sort(key=lambda p: p[0])
    xs = [x for x, _ in points]
    ys = [100.0 * float(r["success_rate"]) for _, r in points]
    num_episodes = points[0][1]["num_episodes"]
    ax.plot(
        xs, ys, color=color, marker=marker, linewidth=2, markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, clip_on=False, zorder=3,
    )
ax.set_xticks(xs)
ax.set_xticklabels([f"x{x:g}" for x in xs])
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_xlabel(xlabel, color=MUTED)
ax.set_ylabel("Success rate [%]", color=MUTED)
ax.set_title(
    f"{title}\nfinal cube-to-goal distance < 2 cm, {num_episodes} episodes per point",
    color=INK, fontsize=11, loc="left",
)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
legend = ax.legend(title="Policy", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
plt.setp(legend.get_texts(), color=INK)
legend.get_title().set_color(MUTED)
fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
print(f"[plot_compare] wrote {args.out}")
