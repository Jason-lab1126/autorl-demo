# Inverse failure attribution v3: baselines, out-of-hypothesis faults, torque signals

Test logs are the attribution_v2 settings (causes off the candidate grid, seeds 0-2, one log per episode),
re-recorded with PhysX joint forces. Training logs: seeds 3-5, mass x1.75/3.5/6, friction x0.55/0.35/0.15,
model_100/700, arm-only explicit actuators. Implicit actuators throughout unless stated; `applied_torque` is
Isaac Lab's PD estimate. Sensor noise x1 = q 0.001 rad, tau 0.5 N·m (N on the fingers),
applied to the logs' q and both torque signals. Accuracy is on kept logs; brackets are 95% cluster-bootstrap
intervals over batches (2000 resamples). Methods are described in `scripts/attribution_eval.py`.

Kept logs (failed, or noise-free nominal residual above the x1 floor threshold 0.02):

| suite | class | kept / total |
|---|---|---|
| test | H_theta-mass | 51 / 72 |
| test | H_theta-friction | 23 / 72 |
| test | H_nu | 24 / 24 |
| test | H_pi | 30 / 48 |
| train | H_theta-mass | 57 / 72 |
| train | H_theta-friction | 39 / 72 |
| train | H_nu | 24 / 24 |
| train | H_pi | 24 / 48 |
| other | H_other | 38 / 96 |

Reproducibility in this run (reference closed loop vs its two nominal replays, max |diff| of q and both torques): ref_s0 0, ref_s1 0, ref_s2 0

## Summary: version A (as first run) vs B (fixed)

A: per-joint sd normalisation, pooled tau_other. B: sqrt(sd^2 + sigma_noise^2) normalisation, tau_other conditional on the predicted class. Class-balanced accuracy on the 4-class test logs and H_other detection rate, %.

| metric | version | x0 | x1 | x2 | x5 |
|---|---|---|---|---|---|
| ours, q+applied | A | 97 | 95 | 90 | 86 |
| ours, q+applied | B | 97 | 95 | 90 | 86 |
| ours+other, 4-class | A | 97 | 95 | 90 | 86 |
| ours+other, 4-class | B | 97 | 95 | 90 | 86 |
| H_other detected | A | 8 | 8 | 5 | 5 |
| H_other detected | B | 63 | 63 | 74 | 68 |
| ours, q only | A | 95 | 92 | 84 | 82 |
| ours, q only | B | 95 | 92 | 85 | 84 |
| ours, q+solver | A | 98 | 82 | 81 | 80 |
| ours, q+solver | B | 98 | 86 | 85 | 83 |
| H_other detected, q+solver | A | 61 | 61 | 61 | 61 |
| H_other detected, q+solver | B | 63 | 63 | 63 | 66 |

## Version A: as first run

### 1. Methods (test logs, 4 classes), class-balanced accuracy %

| method | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| ours | 97 [94, 99] | 95 [89, 99] | 90 [82, 95] | 86 [78, 92] |
| ours+other | 97 [93, 99] | 95 [89, 99] | 90 [83, 95] | 86 [78, 92] |
| old rule (v1) | 45 [39, 59] | 45 [40, 60] | 41 [33, 53] | 36 [29, 48] |
| SysID-only | 42 [36, 56] | 44 [39, 57] | 44 [40, 58] | 43 [39, 58] |
| supervised MLP | 84 [78, 94] | 90 [85, 95] | 91 [85, 97] | 90 [83, 96] |
| threshold | 96 [91, 99] | 94 [87, 99] | 92 [84, 97] | 88 [83, 92] |
| LLM-only | not run (interface in `llm_attribution.py`) |  |  |  |

Per class, noise x1:

| method | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| ours | 84 [63, 100] | 96 [86, 100] | 100 [100, 100] | 100 [100, 100] |
| ours+other | 84 [62, 100] | 96 [86, 100] | 100 [100, 100] | 100 [100, 100] |
| old rule (v1) | 84 [68, 97] | 96 [86, 100] | 0 [0, 0] | 0 [0, 0] |
| SysID-only | 78 [61, 94] | 96 [86, 100] | 0 [0, 0] | 0 [0, 0] |
| supervised MLP | 82 [62, 97] | 100 [100, 100] | 79 [75, 88] | 100 [100, 100] |
| threshold | 76 [50, 94] | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] |

Threshold constants (fitted on training logs): wrist |dtau| > 0.285 sd -> H_nu, finger/joint2 deviation ratio > 9.08 -> friction.

How SysID-only labels the policy and engine logs (noise x1):

| true | n | H_theta-mass | H_theta-friction | none |
|---|---|---|---|---|
| H_pi | 30 | 5 | 9 | 16 |
| H_nu | 24 | 10 | 11 | 3 |

(engine logs: median fitted multiplier 0.94, median fitted residual 44.2 vs noise floor 0.0164)

### 2. Faults outside the hypothesis space (H_other)

Criterion: H_other if every candidate's residual exceeds tau_other = the 95th percentile of the minimum candidate residual over the kept training logs (all classes pooled).

| noise | tau_other | floor mean |
|---|---|---|
| x0 | 13.8 | 0 |
| x1 | 13.8 | 0.0164 |
| x2 | 13.9 | 0.0647 |
| x5 | 14.3 | 0.411 |

Noise x0: predictions for the H_other logs with the criterion (without it in brackets):

| setting | kept / total | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|---|
| other_action_delay | 24 / 24 | 11 (14) | 10 (10) | 0 (0) | 0 (0) | 3 (0) |
| other_grip_x0.25 | 7 / 24 | 1 (1) | 6 (6) | 0 (0) | 0 (0) | 0 (0) |
| other_grip_x0.5 | 0 / 24 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| other_q_bias | 7 / 24 | 5 (5) | 2 (2) | 0 (0) | 0 (0) | 0 (0) |

Noise x1: predictions for the H_other logs with the criterion (without it in brackets):

| setting | kept / total | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|---|
| other_action_delay | 24 / 24 | 10 (13) | 11 (11) | 0 (0) | 0 (0) | 3 (0) |
| other_grip_x0.25 | 7 / 24 | 1 (1) | 6 (6) | 0 (0) | 0 (0) | 0 (0) |
| other_grip_x0.5 | 0 / 24 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| other_q_bias | 7 / 24 | 5 (5) | 2 (2) | 0 (0) | 0 (0) | 0 (0) |

H_other detection rate on the H_other logs and accuracy on the 4-class test logs:

| noise | H_other logs -> H_other | 4-class balanced acc, ours | with criterion | drop (points) |
|---|---|---|---|---|
| x0 | 8 [3, 11] | 97 [94, 99] | 97 [93, 99] | 0.0 (0% of test logs called H_other) |
| x1 | 8 [0, 11] | 95 [89, 99] | 95 [90, 99] | 0.0 (0% of test logs called H_other) |
| x2 | 5 [0, 10] | 90 [82, 95] | 90 [82, 96] | 0.0 (0% of test logs called H_other) |
| x5 | 5 [0, 10] | 86 [79, 92] | 86 [79, 92] | 0.0 (0% of test logs called H_other) |

### 3. Residual signal (ours), class-balanced accuracy %

`tau_solver` = PhysX `get_dof_projected_joint_forces` (the solver's joint force along each DOF); `get_dof_actuation_forces` reads all zeros for implicit drives.

| signal | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| q | 95 [89, 99] | 92 [86, 96] | 84 [75, 90] | 82 [72, 89] |
| q+applied | 97 [93, 99] | 95 [89, 99] | 90 [83, 95] | 86 [78, 91] |
| q+solver | 98 [94, 100] | 82 [76, 87] | 81 [74, 87] | 80 [72, 86] |

With the H_other criterion, H_other detection rate / 4-class balanced accuracy:

| signal | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| q | 0 / 95 | 0 / 92 | 0 / 84 | 0 / 82 |
| q+applied | 8 / 97 | 8 / 95 | 5 / 90 | 5 / 86 |
| q+solver | 61 / 98 | 61 / 82 | 61 / 81 | 61 / 80 |

Per class, noise x1:

| signal | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| q | 80 | 87 | 100 | 100 |
| q+applied | 84 | 96 | 100 | 100 |
| q+solver | 76 | 52 | 100 | 100 |

## Version B: noise-aware normalisation, class-conditional H_other threshold

### 1. Methods (test logs, 4 classes), class-balanced accuracy %

| method | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| ours | 97 [94, 99] | 95 [89, 99] | 90 [82, 95] | 86 [78, 92] |
| ours+other | 97 [93, 99] | 95 [89, 99] | 90 [83, 95] | 86 [78, 92] |
| old rule (v1) | 45 [39, 59] | 45 [40, 60] | 41 [33, 53] | 36 [29, 48] |
| SysID-only | 42 [36, 56] | 44 [39, 57] | 44 [40, 58] | 41 [36, 55] |
| supervised MLP | 84 [78, 94] | 90 [85, 95] | 91 [85, 97] | 90 [83, 96] |
| threshold | 96 [91, 99] | 94 [87, 99] | 92 [84, 97] | 88 [83, 92] |
| LLM-only | not run (interface in `llm_attribution.py`) |  |  |  |

Per class, noise x1:

| method | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| ours | 84 [63, 100] | 96 [86, 100] | 100 [100, 100] | 100 [100, 100] |
| ours+other | 84 [62, 100] | 96 [86, 100] | 100 [100, 100] | 100 [100, 100] |
| old rule (v1) | 84 [68, 97] | 96 [86, 100] | 0 [0, 0] | 0 [0, 0] |
| SysID-only | 78 [61, 94] | 96 [86, 100] | 0 [0, 0] | 0 [0, 0] |
| supervised MLP | 82 [62, 97] | 100 [100, 100] | 79 [75, 88] | 100 [100, 100] |
| threshold | 76 [50, 94] | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] |

Threshold constants (fitted on training logs): wrist |dtau| > 0.283 sd -> H_nu, finger/joint2 deviation ratio > 9.08 -> friction.

How SysID-only labels the policy and engine logs (noise x1):

| true | n | H_theta-mass | H_theta-friction | none |
|---|---|---|---|---|
| H_pi | 30 | 5 | 9 | 16 |
| H_nu | 24 | 9 | 11 | 4 |

(engine logs: median fitted multiplier 0.94, median fitted residual 44 vs noise floor 0.0159)

### 2. Faults outside the hypothesis space (H_other)

Criterion: H_other if every candidate's residual exceeds tau_other = the 95th percentile of the minimum candidate residual over the kept training logs of the class ours predicts.

| noise | tau_other H_theta-mass | tau_other H_theta-friction | tau_other H_nu | tau_other H_pi | floor mean |
|---|---|---|---|---|---|
| x0 | 0.298 | 1.81 | 96.2 | 0 | 0 |
| x1 | 0.309 | 1.81 | 96 | 0.0171 | 0.0159 |
| x2 | 0.331 | 1.75 | 95.5 | 0.0626 | 0.0574 |
| x5 | 0.481 | 1.52 | 93.4 | 0.256 | 0.237 |

Noise x0: predictions for the H_other logs with the criterion (without it in brackets):

| setting | kept / total | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|---|
| other_action_delay | 24 / 24 | 0 (14) | 0 (10) | 0 (0) | 0 (0) | 24 (0) |
| other_grip_x0.25 | 7 / 24 | 1 (1) | 6 (6) | 0 (0) | 0 (0) | 0 (0) |
| other_grip_x0.5 | 0 / 24 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| other_q_bias | 7 / 24 | 5 (5) | 2 (2) | 0 (0) | 0 (0) | 0 (0) |

Noise x1: predictions for the H_other logs with the criterion (without it in brackets):

| setting | kept / total | H_theta-mass | H_theta-friction | H_nu | H_pi | H_other |
|---|---|---|---|---|---|---|
| other_action_delay | 24 / 24 | 0 (13) | 0 (11) | 0 (0) | 0 (0) | 24 (0) |
| other_grip_x0.25 | 7 / 24 | 1 (1) | 6 (6) | 0 (0) | 0 (0) | 0 (0) |
| other_grip_x0.5 | 0 / 24 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| other_q_bias | 7 / 24 | 5 (5) | 2 (2) | 0 (0) | 0 (0) | 0 (0) |

H_other detection rate on the H_other logs and accuracy on the 4-class test logs:

| noise | H_other logs -> H_other | 4-class balanced acc, ours | with criterion | drop (points) |
|---|---|---|---|---|
| x0 | 63 [25, 87] | 97 [94, 99] | 97 [93, 99] | 0.0 (0% of test logs called H_other) |
| x1 | 63 [0, 87] | 95 [89, 99] | 95 [90, 99] | 0.0 (0% of test logs called H_other) |
| x2 | 74 [35, 94] | 90 [82, 95] | 90 [82, 96] | 0.0 (0% of test logs called H_other) |
| x5 | 68 [29, 90] | 86 [79, 92] | 86 [79, 92] | 0.0 (0% of test logs called H_other) |

### 3. Residual signal (ours), class-balanced accuracy %

`tau_solver` = PhysX `get_dof_projected_joint_forces` (the solver's joint force along each DOF); `get_dof_actuation_forces` reads all zeros for implicit drives.

| signal | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| q | 95 [89, 99] | 92 [86, 96] | 85 [77, 91] | 84 [76, 90] |
| q+applied | 97 [93, 99] | 95 [89, 99] | 90 [83, 95] | 86 [78, 91] |
| q+solver | 98 [94, 100] | 86 [80, 91] | 85 [79, 90] | 83 [74, 88] |

With the H_other criterion, H_other detection rate / 4-class balanced accuracy:

| signal | noise x0 | noise x1 | noise x2 | noise x5 |
|---|---|---|---|---|
| q | 45 / 95 | 45 / 90 | 39 / 85 | 55 / 84 |
| q+applied | 63 / 97 | 63 / 95 | 74 / 90 | 68 / 86 |
| q+solver | 63 / 98 | 63 / 85 | 63 / 84 | 66 / 82 |

Per class, noise x1:

| signal | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| q | 80 | 87 | 100 | 100 |
| q+applied | 84 | 96 | 100 | 100 |
| q+solver | 84 | 61 | 100 | 100 |
