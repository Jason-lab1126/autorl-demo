"""Teacher vs. students table for the visual-change eval (results/visual/).

    python scripts/make_visual_table.py

Writes results/visual_robustness.md and .csv, and a strip of what each student sees per condition.
"""

import csv
import json
import os

import imageio.v2 as imageio
import numpy as np

ROOT = "results/visual"
# (key, label, held out of the randomized student's training ranges)
CONDITIONS = [
    ("none", "None", False),
    ("light_x0.5", "Light intensity x0.5", False),
    ("light_x2", "Light intensity x2", False),
    ("cube_red", "Cube color: flat red", False),
    ("table_light_wood", "Table color: light wood", False),
    ("camera_5cm", "Camera moved 5 cm sideways", False),
    ("camera_10cm", "Camera moved 10 cm sideways", False),
    ("light_x3", "Light intensity x3", True),
    ("cube_purple", "Cube color: purple", True),
    ("table_purple", "Table color: purple", True),
    ("camera_12cm", "Camera moved 12 cm sideways", True),
]
POLICIES = [("teacher", "Teacher"), ("student", "Student, fixed scene"), ("student_visual_rand", "Student, randomized visuals")]
HEADER = ["Change", "Held out"] + [f"{label} success" for _, label in POLICIES] + [f"{label} mean distance [cm]" for _, label in POLICIES[1:]]


def load(key: str, policy: str):
    path = os.path.join(ROOT, key, policy, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


rows = []
for key, label, held_out in CONDITIONS:
    summaries = [load(key, policy) for policy, _ in POLICIES]
    row = [label, "yes" if held_out else "no"]
    row += [f"{s['success_rate'] * 100:.0f}%" if s else "-" for s in summaries]
    row += [f"{s['final_dist_mean_m'] * 100:.2f}" if s else "-" for s in summaries[1:]]
    rows.append(row)

with open("results/visual_robustness.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)
with open("results/visual_robustness.md", "w") as f:
    f.write("# Visual robustness: teacher vs. camera-only students\n\n")
    f.write("100 episodes per cell, same seed and end-of-episode metric (success = cube ends within 2 cm of the goal). "
            "The changes affect only what the camera sees. \"Held out\" marks conditions outside the randomized "
            "student's training ranges (light x0.5 to x2, hues below 0.7, camera offsets up to 10 cm); for the "
            "fixed-scene student every change is unseen.\n\n")
    f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
    for row in rows:
        f.write("| " + " | ".join(row) + " |\n")

# first frame the student saw in each condition (the fixed-scene student's runs, then the held-out ones)
views = []
for key, _, _ in CONDITIONS:
    for name in ("student_view.png", "student_visual_rand_view.png"):
        path = os.path.join(ROOT, key, name)
        if os.path.exists(path):
            views.append(imageio.imread(path)[:, :84])
            break
if views:
    strip = np.concatenate([np.kron(v, np.ones((3, 3, 1), dtype=np.uint8)) for v in views], axis=1)
    imageio.imwrite("results/visual_student_views.png", strip)
print(open("results/visual_robustness.md").read())
