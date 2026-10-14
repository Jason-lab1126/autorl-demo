"""Teacher vs. student table for the visual-perturbation eval (results/visual/).

    python scripts/make_visual_table.py

Writes results/visual_robustness.md and .csv, and a strip of the student's view per condition.
"""

import csv
import json
import os

import imageio.v2 as imageio
import numpy as np

ROOT = "results/visual"
CONDITIONS = [
    ("none", "None"),
    ("light_x0.5", "Light intensity x0.5"),
    ("light_x2", "Light intensity x2"),
    ("cube_red", "Cube color: flat red"),
    ("table_light_wood", "Table color: light wood"),
    ("camera_5cm", "Camera shifted 5 cm sideways"),
    ("camera_10cm", "Camera shifted 10 cm sideways"),
]
HEADER = ["Perturbation", "Teacher success", "Teacher mean distance [cm]", "Student success", "Student mean distance [cm]", "Student cube lifted"]

rows, views = [], []
for key, label in CONDITIONS:
    summaries = {}
    for policy in ("teacher", "student"):
        with open(os.path.join(ROOT, key, policy, "summary.json")) as f:
            summaries[policy] = json.load(f)
    t, s = summaries["teacher"], summaries["student"]
    rows.append([
        label, f"{t['success_rate'] * 100:.0f}%", f"{t['final_dist_mean_m'] * 100:.2f}",
        f"{s['success_rate'] * 100:.0f}%", f"{s['final_dist_mean_m'] * 100:.2f}", f"{s['lifted_rate'] * 100:.0f}%",
    ])
    view = os.path.join(ROOT, key, "student_view.png")
    if os.path.exists(view):
        views.append(imageio.imread(view)[:, :84])  # first frame of each condition

with open("results/visual_robustness.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)
with open("results/visual_robustness.md", "w") as f:
    f.write("# Visual robustness: teacher vs. camera-only student\n\n")
    f.write("Best student (BC + more teacher data) and the teacher, 100 episodes per cell, same seed and end-of-episode metric "
            "(success = cube ends within 2 cm of the goal). The perturbations change only what the camera sees.\n\n")
    f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
    for row in rows:
        f.write("| " + " | ".join(row) + " |\n")
if views:
    strip = np.concatenate([np.kron(v, np.ones((3, 3, 1), dtype=np.uint8)) for v in views], axis=1)
    imageio.imwrite("results/visual_student_views.png", strip)
print(open("results/visual_robustness.md").read())
