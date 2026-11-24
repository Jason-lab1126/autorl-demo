# Inverse failure attribution v2: off-grid causes, Gaussian likelihood, sensor noise

Checkpoint model_1499 unless noted, implicit actuators unless noted, seeds [0, 1, 2] x 8 envs per setting, one log per episode. Torques are Isaac Lab's `applied_torque`: a PD **estimate** under implicit actuators, the applied torque only under explicit IdealPD.

## 1. Replay determinism (check_replay_determinism.py)

Residual of a nominal replay against the nominal log after each history (mean over 8 envs). `fresh_*` run in a new process. First differing step in brackets.

| case | default | default_preroll10 | enhanced |
|---|---|---|---|
| direct | 0 | 0 | 0 |
| mass_set_reset | 0.0054 (23) | 0.0048 (24) | 0.0054 (23) |
| fric_set_reset | 0.0054 (23) | 0.0048 (24) | 0.0054 (23) |
| mass_rollout | 0 | 0.00065 (23) | 0 |
| fric_rollout | 0.00065 (23) | 0 | 0.00065 (23) |
| other_policy | 0.00065 (23) | 0.00065 (23) | 0.00065 (23) |
| other_seed | 0.00065 (23) | 0.0054 (23) | 0.00065 (23) |
| closed_loop_again | 0.0074 (23) | 0.002 (23) | 0.0074 (23) |
| fresh_replay | 0 | 0 | 0 |
| fresh_closed_loop | 0 | 0 | 0 |

In this run every rollout is the only rollout of its own process. Reference closed loop vs its nominal replays (max per-episode residual, noise-free):

| reference | log vs replay | replay vs replay |
|---|---|---|
| ref_s0 | 0 | 0 |
| ref_s1 | 0 | 0 |
| ref_s2 | 0 | 0 |

## 2. Noise floor and likelihood scale

Sensor noise x1 = q 0.001 rad, tau 0.5 N·m (N on the fingers).
sigma^2 = mean floor residual (clamped at 1e-6); H_pi / deviation threshold = mean + 4 std.

| noise | floor mean | floor std | sigma^2 | threshold |
|---|---|---|---|---|
| x0 | 0 | 0 | 1e-06 | 1e-12 |
| x1 | 0.0163 | 0.000508 | 0.0163 | 0.0184 |
| x2 | 0.0653 | 0.00262 | 0.0653 | 0.0758 |
| x5 | 0.407 | 0.0223 | 0.407 | 0.497 |

## 3. Kept logs

Kept = failed, or noise-free nominal-replay residual above the x1 floor threshold (0.0184).

| true class | episodes | failed | deviating | kept |
|---|---|---|---|---|
| H_theta-mass | 72 | 3 | 51 | 51 |
| H_theta-friction | 72 | 2 | 24 | 24 |
| H_nu | 24 | 23 | 24 | 24 |
| H_pi | 48 | 30 | 0 | 30 |

## 4. Accuracy per class (kept logs)

| rule | noise | H_theta-mass | H_theta-friction | H_nu | H_pi | overall | class-balanced |
|---|---|---|---|---|---|---|---|
| new | x0 | 92% | 96% | 100% | 100% | 96% | 97% |
| new | x1 | 82% | 96% | 100% | 100% | 92% | 95% |
| new | x2 | 82% | 71% | 100% | 100% | 88% | 88% |
| new | x5 | 82% | 58% | 100% | 100% | 85% | 85% |
| old | x0 | 80% | 96% | 0% | 3% | 50% | 45% |
| old | x1 | 82% | 96% | 0% | 0% | 50% | 45% |
| old | x2 | 82% | 75% | 0% | 0% | 47% | 39% |
| old | x5 | 86% | 58% | 0% | 0% | 45% | 36% |

## Confusion, new rule, noise x0

| true \ predicted | H_theta-mass | H_theta-friction | H_nu | H_pi | none | n |
|---|---|---|---|---|---|---|
| H_theta-mass | 47 | 4 | 0 | 0 | 0 | 51 |
| H_theta-friction | 1 | 23 | 0 | 0 | 0 | 24 |
| H_nu | 0 | 0 | 24 | 0 | 0 | 24 |
| H_pi | 0 | 0 | 0 | 30 | 0 | 30 |

## Confusion, old rule, noise x0

| true \ predicted | H_theta-mass | H_theta-friction | H_nu | H_pi | none | n |
|---|---|---|---|---|---|---|
| H_theta-mass | 41 | 10 | 0 | 0 | 0 | 51 |
| H_theta-friction | 1 | 23 | 0 | 0 | 0 | 24 |
| H_nu | 12 | 12 | 0 | 0 | 0 | 24 |
| H_pi | 12 | 17 | 0 | 1 | 0 | 30 |

## Confusion, new rule, noise x1

| true \ predicted | H_theta-mass | H_theta-friction | H_nu | H_pi | none | n |
|---|---|---|---|---|---|---|
| H_theta-mass | 42 | 9 | 0 | 0 | 0 | 51 |
| H_theta-friction | 1 | 23 | 0 | 0 | 0 | 24 |
| H_nu | 0 | 0 | 24 | 0 | 0 | 24 |
| H_pi | 0 | 0 | 0 | 30 | 0 | 30 |

## Confusion, old rule, noise x1

| true \ predicted | H_theta-mass | H_theta-friction | H_nu | H_pi | none | n |
|---|---|---|---|---|---|---|
| H_theta-mass | 42 | 9 | 0 | 0 | 0 | 51 |
| H_theta-friction | 1 | 23 | 0 | 0 | 0 | 24 |
| H_nu | 12 | 12 | 0 | 0 | 0 | 24 |
| H_pi | 13 | 17 | 0 | 0 | 0 | 30 |

## Predictions per true setting, new rule, noise x1

| setting | true class | kept / total | H_theta-mass | H_theta-friction | H_nu | H_pi | none |
|---|---|---|---|---|---|---|---|
| mass_x1.5 | H_theta-mass | 10 / 24 | 3 | 7 | 0 | 0 | 0 |
| mass_x2.5 | H_theta-mass | 17 / 24 | 15 | 2 | 0 | 0 | 0 |
| mass_x4 | H_theta-mass | 24 / 24 | 24 | 0 | 0 | 0 | 0 |
| fric_x0.7 | H_theta-friction | 4 / 24 | 0 | 4 | 0 | 0 | 0 |
| fric_x0.4 | H_theta-friction | 9 / 24 | 1 | 8 | 0 | 0 | 0 |
| fric_x0.25 | H_theta-friction | 11 / 24 | 0 | 11 | 0 | 0 | 0 |
| policy_300 | H_pi | 24 / 24 | 0 | 0 | 0 | 24 | 0 |
| policy_500 | H_pi | 6 / 24 | 0 | 0 | 0 | 6 | 0 |
| engine_explicit | H_nu | 24 / 24 | 0 | 0 | 24 | 0 | 0 |

Per-episode residuals and posteriors for every noise level: `v2/episodes.csv`. Plot: `acc_vs_noise.png`.
