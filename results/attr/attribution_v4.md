# Inverse failure attribution v4: what the replay buys (C2) and what the posterior buys (C1)

Same logs as attribution_v3 (test: seeds 0-2, causes off the candidate grid; train: seeds 3-5, other intensities),
plus a generalisation suite. Ours = attribution_v3 version B. Sections 1-3 use the v3 sensor-noise model
(q 1e-3 rad, tau 0.5 on all joints at x1); section 4 uses a per-joint Panda datasheet model. 95% intervals:
cluster bootstrap over batches (2000 resamples). Implicit actuators; `applied_torque` is the PD estimate.

## 1. Does attribution need the replay? (test logs, class-balanced accuracy %)

| method | uses replay | x0 | x1 | x2 | x5 |
|---|---|---|---|---|---|
| ours | yes | 97 [94, 99] | 95 [89, 99] | 90 [82, 95] | 86 [78, 92] |
| threshold (replay) | yes | 96 [90, 99] | 94 [88, 99] | 92 [85, 97] | 88 [83, 92] |
| MLP (replay) | yes | 84 [78, 94] | 90 [85, 95] | 91 [85, 96] | 90 [84, 96] |
| rule tree (log only) | no | 54 [40, 61] | 54 [41, 61] | 54 [39, 61] | 48 [35, 57] |
| MLP (log only) | no | 57 [52, 62] | 55 [51, 60] | 56 [49, 63] | 51 [44, 57] |

Per class at x1:

| method | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| ours | 84 | 96 | 100 | 100 |
| threshold (replay) | 76 | 100 | 100 | 100 |
| MLP (replay) | 82 | 100 | 79 | 100 |
| rule tree (log only) | 82 | 26 | 100 | 7 |
| MLP (log only) | 84 | 70 | 67 | 0 |

Log-only rule tree fitted at x1 (features: wrist_saturation, tau2_hold, finger_force_hold, grip_closed_frac, episode_len, failed):

```
if wrist_saturation <= 0.108:
  -> H_pi
else:
  if wrist_saturation <= 0.442:
    if tau2_hold <= -10.2:
      -> H_theta-mass
    else:
      -> H_theta-friction
  else:
    if grip_closed_frac <= 0.464:
      -> H_theta-friction
    else:
      -> H_nu
```

Log-only MLP input: per-joint mean/std/min/max of q and tau, std and max |qd|, action mean/std, the six rule-tree
features, and q, tau subsampled every 10 steps (560 features).

## 2. What the posterior adds beyond accuracy

### Calibration (test logs; ECE over 10 confidence bins, %)

| method | x0 | x1 | x2 | x5 |
|---|---|---|---|---|
| ours | 3.9 (acc 96, mean conf 100) | 18.1 (acc 93, mean conf 76) | 23.2 (acc 89, mean conf 66) | 24.8 (acc 87, mean conf 62) |
| MLP (replay) | 10.7 (acc 84, mean conf 94) | 7.6 (acc 89, mean conf 95) | 6.3 (acc 89, mean conf 95) | 7.8 (acc 88, mean conf 96) |
| MLP (log only) | 34.3 (acc 59, mean conf 93) | 35.8 (acc 59, mean conf 93) | 36.4 (acc 59, mean conf 93) | 38.6 (acc 55, mean conf 93) |

`ours` at x0 uses sigma^2 = 1e-6 (the replay floor is exactly 0), so its posterior is essentially one-hot there.
Reliability diagram: `reliability.png`.

### Parameter estimates (test logs of the mass and friction classes, noise x1)

Estimated multiplier along the true class's axis. `posterior mean`: likelihood-weighted geometric mean over the
class's grid candidates (mass 2/3/5, friction 0.5/0.3/0.2). `continuous fit`: parabola in log-multiplier through
the residuals of nominal and the grid (can leave the grid). The threshold rule and the MLPs give no estimate.

| true setting | n | posterior mean: median (median abs rel. error) | continuous fit: median (median abs rel. error) |
|---|---|---|---|
| mass_x1.5 | 10 | 2.55 (70%) | 1.78 (24%) |
| mass_x2.5 | 17 | 2.68 (9%) | 2.38 (10%) |
| mass_x4 | 24 | 3.39 (17%) | 3.96 (12%) |
| fric_x0.7 | 3 | 0.33 (52%) | 0.41 (41%) |
| fric_x0.4 | 9 | 0.38 (6%) | 0.39 (3%) |
| fric_x0.25 | 11 | 0.32 (26%) | 0.39 (55%) |

## 3. Generalisation without refitting (class-balanced accuracy %, noise x1 unless stated)

New policy: obs-noise `model_2999` (nominal success 71% over these initial states); faults injected only from initial states it solves at nominal physics; its H_pi fault is obs-noise `model_1499`. New numerics: physics dt 0.005 with decimation 4 (same control rate), solver position
iterations x0.25 (robot 8 -> 2, cube 16 -> 4); baseline policy. Compound: cube mass x2.5 and friction x0.4.

Kept logs: new policy H_theta-mass 41, H_theta-friction 39, H_nu 17, H_pi 17; new numerics 45 (nu_substeps_x2 21/24, nu_solver_iters_x0.25 24/24); compound 19/24.

| method | Lift test (ref) | new policy | drop | new numerics -> H_nu | compound -> mass or friction |
|---|---|---|---|---|---|
| ours | 95 [89, 99] | 93 [87, 97] | +2 | 0 [0, 0] | 100 [100, 100] |
| threshold (replay) | 94 [87, 99] | 91 [83, 95] | +3 | 0 [0, 0] | 100 [100, 100] |
| MLP (replay) | 90 [85, 95] | 84 [76, 91] | +6 | 2 [0, 7] | 100 [100, 100] |
| rule tree (log only) | 54 [40, 61] | 44 [25, 58] | +10 | 2 [0, 7] | 89 [67, 100] |
| MLP (log only) | 55 [51, 60] | 25 [25, 33] | +30 | 0 [0, 0] | 100 [100, 100] |

Per class on the new policy (x1):

| method | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| ours | 80 | 90 | 100 | 100 |
| threshold (replay) | 78 | 85 | 100 | 100 |
| MLP (replay) | 80 | 79 | 76 | 100 |
| rule tree (log only) | 0 | 74 | 100 | 0 |
| MLP (log only) | 0 | 100 | 0 | 0 |

New numerics, predictions per setting (x1; `ours+other` adds the class-conditional H_other criterion):

| setting | method | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|---|
| nu_substeps_x2 | ours | 5 | 16 | 0 | 0 | 0 |
| nu_substeps_x2 | threshold (replay) | 6 | 15 | 0 | 0 | 0 |
| nu_substeps_x2 | MLP (replay) | 1 | 18 | 0 | 2 | 0 |
| nu_substeps_x2 | rule tree (log only) | 18 | 3 | 0 | 0 | 0 |
| nu_substeps_x2 | MLP (log only) | 11 | 10 | 0 | 0 | 0 |
| nu_substeps_x2 | ours+other | 5 | 16 | 0 | 0 | 0 |
| nu_substeps_x2 | (median min residual 0.0498, most common best candidate nominal; floor 0.0159) | | | | | |
| nu_solver_iters_x0.25 | ours | 11 | 13 | 0 | 0 | 0 |
| nu_solver_iters_x0.25 | threshold (replay) | 0 | 24 | 0 | 0 | 0 |
| nu_solver_iters_x0.25 | MLP (replay) | 19 | 4 | 1 | 0 | 0 |
| nu_solver_iters_x0.25 | rule tree (log only) | 16 | 7 | 1 | 0 | 0 |
| nu_solver_iters_x0.25 | MLP (log only) | 6 | 18 | 0 | 0 | 0 |
| nu_solver_iters_x0.25 | ours+other | 0 | 6 | 0 | 0 | 18 |
| nu_solver_iters_x0.25 | (median min residual 2.92, most common best candidate fric_x0.2; floor 0.0159) | | | | | |

Compound fault (x1): predictions, and ours' continuous estimates along each axis (true mass 2.5, friction 0.4):

| method | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|
| ours | 9 | 10 | 0 | 0 | 0 |
| threshold (replay) | 7 | 12 | 0 | 0 | 0 |
| MLP (replay) | 9 | 10 | 0 | 0 | 0 |
| rule tree (log only) | 9 | 8 | 2 | 0 | 0 |
| MLP (log only) | 10 | 9 | 0 | 0 | 0 |
| ours+other | 7 | 9 | 0 | 0 | 3 |

Median continuous estimates: mass x2.54, friction x0.38 (each axis fitted with the other at nominal, since the grid has no joint candidates).

model_2999's own nominal failures (7 episodes; ideal answer H_pi), x1:

| method | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| ours | 0 | 0 | 0 | 7 |
| threshold (replay) | 0 | 0 | 0 | 7 |
| MLP (replay) | 0 | 1 | 0 | 6 |
| rule tree (log only) | 0 | 7 | 0 | 0 |
| MLP (log only) | 0 | 7 | 0 | 0 |

Generalisation at every noise level (class-balanced accuracy on the new-policy logs):

| method | x0 | x1 | x2 | x5 |
|---|---|---|---|---|
| ours | 95 | 93 | 86 | 79 |
| threshold (replay) | 94 | 91 | 86 | 78 |
| MLP (replay) | 77 | 84 | 87 | 87 |
| rule tree (log only) | 44 | 44 | 44 | 49 |
| MLP (log only) | 25 | 25 | 25 | 25 |

## 4. Torque signals under a per-joint sensor-noise model

x1 = Panda datasheet torque noise (RMS): 0.055 N·m on joints 1-4 (the larger of the two published figures, 0.005 and
0.055), scaled by sensor range (12/87) to 0.008 N·m on the wrist joints 5-7; q noise 1e-3 rad. The finger joints are left
out of the torque residual: the Franka Hand has no joint torque sensors, and PhysX's projected finger joint force is
ill-posed (below). Ours, version B, test logs, class-balanced accuracy %; then H_other detection / 4-class accuracy.

| signal | x0 | x1 | x2 | x5 | H_other detected / 4-class, x1 |
|---|---|---|---|---|---|
| q | 95 | 92 | 85 | 84 | 45 / 90 |
| q+applied | 95 | 94 | 88 | 85 | 63 / 92 |
| q+solver | 100 | 98 | 96 | 92 | 63 / 96 |

Per class at x1:

| signal | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| q | 80 | 87 | 100 | 100 |
| q+applied | 78 | 96 | 100 | 100 |
| q+solver | 90 | 100 | 100 | 100 |

### Why the two fingers' tau_solver differ

The two finger joints are defined identically in the USD (prismatic, no mimic joint, same drive, limits 0-0.04 m), and
Isaac Lab's and PhysX's DOF orders match. During a grasp the PD estimate gives both fingers ~45-55 N, but PhysX's
projected joint force reports the full grip force (~-44 N) on at most ONE finger for the whole grasp and ~0.1 N on the
other. Reference episodes (finger1 only: 9, finger2 only: 4, neither: 11, both: 0). Finger 1 carries it more
often, hence its larger sd. Joint limits are not the cause (<2% of the large readings occur at a limit). Interpretation
(not verified): the two fingers and the cube close a kinematic loop; the squeeze is an internal force of that loop,
which is statically indeterminate, and the solver attributes it to one joint arbitrarily. So the finger channels of
tau_solver are not a physical measurement, and the real Franka Hand has no joint torque sensor either: section 4
leaves them out. The arm channels track the PD estimate at steady state (e.g. joint2 -25.8 vs -25.7 N·m) but differ
in transients and on joint7, where the PD estimate sits at its 12 N·m limit while the solver force is ~0.

Torque-noise sources: Panda datasheet (torque noise RMS listed as 0.005 N·m in one revision and 0.055 N·m in
another; repeatability < 0.05 N·m; relative accuracy 0.15 N·m, a bias-like error not modelled here; 13-bit
resolution). The datasheet gives one figure for all joints; the wrist value is scaled by the sensor range (±12 vs
±87 N·m). [Panda datasheet](https://www.wiredworkers.io/wp-content/uploads/2019/12/Panda_FrankaEmika_ENG.pdf),
[instruction handbook](https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf),
[datasheet (alternate revision)](https://centers.ulbsibiu.ro/incon/wp-content/uploads/Datasheet-Franka-Emika-Panda-Research-collaborative-robot.pdf).

## 5. Second task: Factory peg insertion (survey only, nothing trained)

- Local stack: Isaac Lab 2.0.2 on Isaac Sim 4.5; task `Isaac-Factory-PegInsert-Direct-v0` (rl_games, LSTM policy:
  MLP 512-128-64 + LSTM 1024; obs 21, state 72, actions 6; sim dt 1/120, decimation 8, 10 s episodes = 150 steps).
- Published pretrained checkpoint (`--use_pretrained_checkpoint` path on NVIDIA's asset server): **404 for the
  Isaac Sim 4.5 asset root** that this install uses; **present for the 5.0 and 5.1 roots** (`checkpoint.pth`,
  213 MB, dated Aug and Oct 2025), i.e. released for Isaac Lab 2.2/2.3. Whether it loads on 2.0.2 (same obs/state/
  action sizes and network) is not verified: downloading and unpickling it was blocked in this session, and the
  file needs `weights_only=False`, so it should be inspected deliberately.
- Training from scratch: `scripts/factory_throughput.py` measures 552 env-steps/s with 128 envs (random actions,
  one RTX 3090). The shipped config (200 epochs x 128 envs x 128-step horizon = 3.3M env steps) is then ~1.7 h of
  simulation plus PPO/LSTM updates: roughly 2-2.5 h per seed on one GPU. More envs may raise throughput (untested).
