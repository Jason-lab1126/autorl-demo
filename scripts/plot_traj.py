"""Torque comparison plot and residual tables for the trajectories written by log_traj.py.

Usage (from ~/autorl-demo):
    python scripts/plot_traj.py [--traj_dir results/traj] [--env 0]

Outputs:
    <traj_dir>/torque_compare.png   env 0 torques of panda_joint2/4 and panda_finger_joint1, plus cube height
    stdout                          per-joint mean |dtau| vs nominal, nominal_repeat max |dq|, success rates
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = ["nominal", "nominal_repeat", "friction_x0.5", "mass_x5", "replay_fric0.5", "replay_mass5"]
PLOT_CELLS = {  # cell -> (color, linestyle); nominal_repeat dashed so it stays visible on top of nominal
    "nominal": ("#2a78d6", "-"),
    "nominal_repeat": ("#eb6834", "--"),
    "replay_fric0.5": ("#1baf7a", "-"),
    "replay_mass5": ("#eda100", "-"),
}
PLOT_JOINTS = ["panda_joint2", "panda_joint4", "panda_finger_joint1"]

parser = argparse.ArgumentParser(description="Plot and tabulate log_traj.py trajectories.")
parser.add_argument("--traj_dir", type=str, default="results/traj")
parser.add_argument("--env", type=int, default=0, help="Environment to plot.")
parser.add_argument("--torque", type=str, default="applied_torque", choices=["applied_torque", "computed_torque"])
args = parser.parse_args()


def load(cell: str) -> dict:
    with np.load(os.path.join(args.traj_dir, f"{cell}.npz")) as f:
        return {key: f[key] for key in f.files}


data = {cell: load(cell) for cell in CELLS}
nominal = data["nominal"]
joint_names = [str(name) for name in nominal["joint_names"]]


def both_alive(a: dict, b: dict) -> tuple[np.ndarray, int]:
    steps = min(a["alive"].shape[0], b["alive"].shape[0])
    return a["alive"][:steps] & b["alive"][:steps], steps


# plot
fig, axes = plt.subplots(len(PLOT_JOINTS) + 1, 1, figsize=(10, 9), sharex=True)
for cell, (color, style) in PLOT_CELLS.items():
    d = data[cell]
    alive = d["alive"][:, args.env]
    t = np.arange(len(alive))
    for ax, joint in zip(axes, PLOT_JOINTS):
        tau = np.where(alive, d[args.torque][:, args.env, joint_names.index(joint)], np.nan)
        ax.plot(t, tau, color=color, linestyle=style, linewidth=1.5, label=cell)
    height = np.where(alive, d["cube_pos"][:, args.env, 2], np.nan)
    axes[-1].plot(t, height, color=color, linestyle=style, linewidth=1.5, label=cell)
for ax, joint in zip(axes, PLOT_JOINTS):
    ax.set_ylabel(f"{joint}\ntorque [N·m]" if "finger" not in joint else f"{joint}\nforce [N]")
axes[-1].set_ylabel("cube height [m]")
axes[-1].set_xlabel("control step")
axes[0].legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
for ax in axes:
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"env {args.env}: {args.torque} and cube height (open-loop replays use nominal's actions)")
fig.tight_layout()
out = os.path.join(args.traj_dir, "torque_compare.png")
fig.savefig(out, dpi=130)
print(f"[plot_traj] wrote {out}\n")

# success rates
print("success rate (final cube-to-goal < 2 cm):")
for cell in CELLS:
    d = data[cell]
    print(f"  {cell:<15} {d['success'].mean() * 100:5.1f}%  ({d['success'].sum()}/{len(d['success'])})"
          f"  mean final dist {d['final_dist'].mean() * 100:6.2f} cm")

# residual table
print(f"\nmean |Δτ| vs nominal ({args.torque}, steps where both are alive; arm N·m, fingers N):")
short = [name.replace("panda_", "").replace("finger_joint", "f") for name in joint_names]
print(f"  {'cell':<15} {'steps':>6} " + " ".join(f"{s:>8}" for s in short))
for cell in CELLS[1:]:
    d = data[cell]
    mask, steps = both_alive(nominal, d)
    diff = np.abs(d[args.torque][:steps] - nominal[args.torque][:steps])[mask]  # (alive steps, joint)
    print(f"  {cell:<15} {mask.sum():>6} " + " ".join(f"{v:8.3f}" for v in diff.mean(axis=0)))

# determinism check
mask, steps = both_alive(nominal, data["nominal_repeat"])
rep = data["nominal_repeat"]
print("\ndeterminism (nominal_repeat vs nominal, both-alive steps):")
for key in ["joint_pos", "joint_vel", "applied_torque", "computed_torque", "action", "cube_pos"]:
    print(f"  max |Δ{key}| = {np.abs(rep[key][:steps] - nominal[key][:steps])[mask].max():.3e}")
alive_same = np.array_equal(rep["alive"], nominal["alive"])
dq = np.abs(rep["joint_pos"][:steps] - nominal["joint_pos"][:steps])[mask].max()
print(f"  alive masks identical: {alive_same}")
print(f"  => max |Δq| = {dq:.3e} rad: simulation is {'deterministic' if dq == 0.0 and alive_same else 'NOT bit-exact'}")
