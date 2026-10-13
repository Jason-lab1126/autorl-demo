# autorl-demo

Train a PPO policy for the Franka Panda lift-cube task in Isaac Lab, then measure how robust it is: an end-of-episode success metric evaluated across checkpoints and physical/sensing perturbations.

![Close-up of the trained policy lifting the cube to its goal](media/closeup_1env_10s_clean.gif)

Videos of the final policy (`model_1499.pt`):
[single-robot close-up (mp4)](logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/videos/play/closeup_1env_10s_clean.mp4) ·
[16 environments (mp4)](logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/videos/play/play_16env_10s.mp4)

## Key findings

- **The task is solved in under half an hour.** 1500 PPO iterations (4096 envs, 27 minutes on one RTX 3090) take the policy from 0% to 100% success, with the cube ending 0.3 cm from its goal on average.
- **Precision arrives late.** At iteration 300 the policy already lifts the cube (mean final distance 9 cm, down from 77 cm at iteration 100) but only 1% of episodes end within 2 cm. Iteration 500 reaches 72%, the final policy 100%.
- **Cube mass has a cliff.** The final policy is almost unaffected up to 5x mass (94%) and collapses at 10x (3%).
- **Friction matters once it is low enough.** Halving cube and finger friction costs nothing (99%); at 0.2x success drops to 65% and at 0.1x to 2%.
- **Observation noise degrades smoothly.** 100 → 97 → 65 → 28 → 1% at 0x, 1x, 2x, 3x, 5x noise.
- **Action noise and a wider spawn range are tolerated best.** The final policy keeps 80% with strong action noise and 83% with a doubled spawn range.
- **More training buys robustness, not just nominal success.** Iteration 500 loses ground faster than the final policy under every perturbation.

## Results

Success is measured at the **end** of each episode: the cube must finish within 2 cm of the commanded goal position. Each cell is 100 episodes (100 environments, one episode each) on `Isaac-Lift-Cube-Franka-Play-v0` with deterministic policy actions, and every cell is reset with the same seed so all cells see the same goals. With 100 episodes per cell, differences of a few points are within sampling error.

### Success rate [%] (final distance < 2 cm)

| Perturbation | Level | iter 100 | iter 300 | iter 500 | iter 1499 |
|---|---|---|---|---|---|
| None | – | 0 | 1 | 72 | 100 |
| Cube mass | x2 | 0 | 2 | 65 | 100 |
| | x3 | 0 | 2 | 58 | 97 |
| | x5 | 0 | 2 | 41 | 94 |
| | x10 | 0 | 1 | 0 | 3 |
| Friction (cube + fingers) | x0.5 | 0 | 4 | 72 | 99 |
| | x0.2 | 0 | 1 | 29 | 65 |
| | x0.1 | 0 | 0 | 0 | 2 |
| Observation noise | x1 | 0 | 1 | 54 | 97 |
| | x2 | 0 | 1 | 30 | 65 |
| | x3 | 0 | 0 | 8 | 28 |
| | x5 | 0 | 0 | 0 | 1 |
| Action noise | medium (std 0.1) | 0 | 1 | 68 | 99 |
| | strong (std 0.5) | 0 | 1 | 39 | 80 |
| Spawn range | medium (x1.5) | 0 | 3 | 65 | 99 |
| | strong (x2) | 0 | 0 | 60 | 83 |

The mean final distance per cell is in [`results/eval_matrix.md`](results/eval_matrix.md); all per-cell statistics are in [`results/eval_matrix_cells.csv`](results/eval_matrix_cells.csv).

### Success vs. observation noise

![Success rate vs. observation noise, one line per checkpoint](results/success_vs_obs_noise.png)

### Success vs. cube mass

![Success rate vs. cube mass, one line per checkpoint](results/success_vs_cube_mass.png)

The x0 point on the noise plot and the x1 point on the mass plot are the shared unperturbed baseline.

### Perturbations

| Perturbation | What changes |
|---|---|
| Cube mass | Multiplier on the cube's mass and inertia |
| Friction | Multiplier on the static and dynamic friction of the cube and both gripper fingers (nominal 0.5) |
| Observation noise | Per-step Gaussian noise on joint positions (0.01 rad), joint velocities (0.1 rad/s) and cube position (1 cm), times the multiplier; the goal and last-action entries stay clean |
| Action noise | Per-step Gaussian noise on the raw policy actions, arm and gripper |
| Spawn range | Multiplier on the cube's x/y spawn range (nominal x ±10 cm, y ±25 cm) |

The levels are defined in one table at the top of [`scripts/eval_lift.py`](scripts/eval_lift.py).

## Setup

Tested with:

- Isaac Lab 2.0.2 (git tag `v2.0.2`) on Isaac Sim 4.5.0 (binary install), Ubuntu 20.04, one NVIDIA RTX 3090
- Python 3.10 conda environment with Isaac Lab installed (`./isaaclab.sh -i`), which provides `rsl-rl-lib` 2.3.3

Isaac Lab itself is not modified and is not part of this repository. `env.sh` assumes:

| What | Default | Change it in |
|---|---|---|
| Isaac Lab checkout | `~/IsaacLab` | `ISAACLAB_PATH` in `env.sh` |
| Isaac Sim install | `~/isaac-sim` | `ISAACSIM_PATH` in `env.sh` |
| Conda environment | `WL` | `conda activate` line in `env.sh` |
| GPU | index 0 | `DEMO_GPU=<index>` before sourcing |

```bash
git clone git@github.com:Jason-lab1126/autorl-demo.git
cd autorl-demo
source env.sh        # activates the environment and pins everything to one GPU
```

## Usage

Run everything from the repository root.

### Train

```bash
./scripts/train_lift.sh --run_name ppo_baseline
```

PPO (rsl_rl) on `Isaac-Lift-Cube-Franka-v0`, headless, 4096 environments, 1500 iterations, a checkpoint every 50 iterations. The run is written to `logs/rsl_rl/franka_lift/<timestamp>_<run_name>/`. Extra arguments are passed to Isaac Lab's `train.py` (for example `--num_envs 2048 --max_iterations 500`).

### Evaluate

```bash
RUN=logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline
python scripts/eval_lift.py --run_dir $RUN --headless --device cuda:0
```

Runs the full matrix (4 checkpoints x 16 cells, about 5 minutes) in a single simulator instance and writes `eval_matrix.md`, `eval_matrix.csv` and `eval_matrix_cells.csv` into `results/`. Useful options: `--iters 500 1499`, `--perturbations obs_noise cube_mass`, `--num_envs 100`, `--threshold 0.02`.

### Plot

```bash
python scripts/plot_eval.py
```

Reads `results/eval_matrix_cells.csv` and writes the two plots above. It does not start the simulator.

### Videos

```bash
# Isaac Lab's stock play.py (wide view of all environments)
python scripts/play_lift.py --task Isaac-Lift-Cube-Franka-Play-v0 --num_envs 16 --headless --device cuda:0 \
    --video --video_length 500 --checkpoint $RUN/model_1499.pt

# close-up of one robot with a custom camera
python scripts/record_video.py --checkpoint $RUN/model_1499.pt --name closeup --no_markers \
    --eye 1.45 0.85 0.9 --lookat 0.3 0.0 0.35 --headless --device cuda:0
```

`play_lift.py` runs Isaac Lab's `play.py` unmodified; it only adds a one-line alias because Isaac Lab 2.0.2's `play.py` expects an attribute that `rsl-rl-lib` 2.3.3 renamed.

## Repository layout

| Path | Contents |
|---|---|
| `env.sh` | Environment setup and GPU pinning |
| `scripts/` | Train, eval matrix, plotting and video scripts |
| `results/` | Eval tables (markdown, CSV) and plots |
| `media/` | GIF preview used in this README |
| `logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/` | The training run: checkpoints for iterations 100, 300, 500 and 1499, the exported policy (`exported/policy.pt`, `policy.onnx`), configs (`params/`), the first single-checkpoint eval (`eval/`) and the two videos |
