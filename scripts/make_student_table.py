"""Build the student comparison table from the closed-loop eval summaries.

    python scripts/make_student_table.py

Writes results/student_comparison.md and .csv. Needs no simulator.
"""

import csv
import json
import os

# (row label, eval summary, training config or None, what the extra data is)
ROWS = [
    ("Teacher (state-based RL)", "results/student_bc/teacher_reference/summary.json", None, "-"),
    ("BC 500", "results/student_bc/summary.json", "logs/student/bc_500ep/config.json", "-"),
    ("BC + more teacher data", "results/student_bc_more/summary.json", "logs/student/bc_more_teacher/config.json", "750 teacher episodes"),
    ("BC + DAgger round 1", "results/student_dagger_r1/summary.json", "logs/student/dagger_r1/config.json", "250 student episodes, teacher labels"),
    ("BC + DAgger round 2", "results/student_dagger_r2/summary.json", "logs/student/dagger_r2/config.json", "500 student episodes, teacher labels"),
    ("BC + DAgger round 3", "results/student_dagger_r3/summary.json", "logs/student/dagger_r3/config.json", "750 student episodes, teacher labels"),
]
HEADER = ["Policy", "Training episodes", "Extra data beyond BC 500", "Success at 2 cm [%]", "Mean final distance [cm]",
          "Median final distance [cm]", "Within 5 cm [%]", "Cube lifted [%]"]

table = []
for label, summary_path, config_path, extra in ROWS:
    if not os.path.exists(summary_path):
        continue
    with open(summary_path) as f:
        summary = json.load(f)
    episodes = "-"
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
        # every episode in these datasets is 250 frames long
        episodes = str(config.get("num_episodes", (config["train_frames"] + config["val_frames"]) // 250))
    table.append([
        label, episodes, extra,
        f"{summary['success_rate'] * 100:.0f}", f"{summary['final_dist_mean_m'] * 100:.2f}",
        f"{summary['final_dist_median_m'] * 100:.2f}", f"{summary['success_rate_5cm'] * 100:.0f}",
        f"{summary['lifted_rate'] * 100:.0f}",
    ])

with open("results/student_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(table)
with open("results/student_comparison.md", "w") as f:
    f.write("# Vision student: closed-loop comparison\n\n")
    f.write("100 episodes per row, same seed and same end-of-episode metric as the teacher's eval. ")
    f.write("Every student is trained from scratch for 20 epochs on all the data listed for its row.\n\n")
    f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
    for row in table:
        f.write("| " + " | ".join(row) + " |\n")
print(open("results/student_comparison.md").read())
