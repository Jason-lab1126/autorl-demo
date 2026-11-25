# Inverse failure attribution v5: calibrated posterior, parameter-vector H_theta, LLM-proposed candidates

Re-analysis of the attribution_v3/v4 rollouts (no new closed-loop logs). Sensor-noise model as in v3/v4 sections
1-3 (q 1e-3 rad, tau 0.5 at x1, all joints), noise-aware normalisation. Implicit actuators; `applied_torque` is
the PD estimate. Kept logs as before. Method details: `scripts/attribution_v5.py`.

## 1. Calibrated likelihood (test logs)

T is fitted per noise level on training logs only. n_eff from the lag-1 autocorrelation of the per-step residual
(median n_eff / n over test logs at x1: 0.437).
In-hypothesis training logs: the true class has a candidate within 10x the x1 noise floor. Excluded at x1: engine_explicit_arm 24, fric_x0.15 12, mass_x3.5 1, mass_x6 6 (43 of 144).

| likelihood | noise | balanced acc % | ECE % | mean confidence % | accuracy % | T |
|---|---|---|---|---|---|---|
| v4: -r/(2 sigma_floor^2) | x0 | 97 | 3.9 | 100 | 96 | 1e-06 |
| v4: -r/(2 sigma_floor^2) | x1 | 95 | 18.1 | 76 | 93 | 0.0159 |
| v4: -r/(2 sigma_floor^2) | x2 | 90 | 23.2 | 66 | 89 | 0.0574 |
| v4: -r/(2 sigma_floor^2) | x5 | 86 | 24.8 | 62 | 87 | 0.237 |
| mean residual, T fitted on all training logs | x0 | 85 | 29.8 | 56 | 86 | 3 |
| mean residual, T fitted on all training logs | x1 | 85 | 29.9 | 56 | 86 | 3 |
| mean residual, T fitted on all training logs | x2 | 85 | 29.3 | 56 | 85 | 3 |
| mean residual, T fitted on all training logs | x5 | 85 | 32.6 | 55 | 86 | 3.32 |
| mean residual, T fitted on in-hypothesis training logs | x0 | 95 | 6.3 | 88 | 93 | 0.00552 |
| mean residual, T fitted on in-hypothesis training logs | x1 | 95 | 6.1 | 88 | 93 | 0.00552 |
| mean residual, T fitted on in-hypothesis training logs | x2 | 95 | 14.3 | 79 | 92 | 0.0123 |
| mean residual, T fitted on in-hypothesis training logs | x5 | 88 | 23.3 | 65 | 88 | 0.0608 |
| summed, per-candidate n_eff, T in-hypothesis | x0 | 93 | 13.0 | 78 | 91 | 1.11 |
| summed, per-candidate n_eff, T in-hypothesis | x1 | 83 | 11.7 | 70 | 81 | 2.01 |
| summed, per-candidate n_eff, T in-hypothesis | x2 | 66 | 7.2 | 64 | 63 | 4.95 |
| summed, per-candidate n_eff, T in-hypothesis | x5 | 52 | 16.0 | 47 | 51 | 18.2 |
| summed, common n_eff, T in-hypothesis (v5) | x0 | 94 | 12.0 | 83 | 91 | 0.741 |
| summed, common n_eff, T in-hypothesis (v5) | x1 | 95 | 10.8 | 85 | 92 | 0.819 |
| summed, common n_eff, T in-hypothesis (v5) | x2 | 95 | 8.9 | 86 | 92 | 1 |
| summed, common n_eff, T in-hypothesis (v5) | x5 | 95 | 10.4 | 82 | 92 | 1.49 |
| MLP (replay), for reference | x0 | 84 | 10.7 | 94 | 84 | - |
| MLP (replay), for reference | x1 | 90 | 7.6 | 95 | 89 | - |
| MLP (replay), for reference | x2 | 91 | 6.3 | 95 | 89 | - |
| MLP (replay), for reference | x5 | 90 | 7.8 | 96 | 88 | - |

Reliability diagram: `reliability_v5.png` (v4 likelihood, v5 likelihood, replay MLP).

## 2. H_theta as a parameter vector (cube mass, friction), noise x1

Search range [0.05, 20] per parameter, log scale. Replays exist only on the two axes, so the surrogate is separable
and the mass-friction interaction is not identifiable from them; values beyond the replayed range (mass > 5,
friction < 0.2, either < or > nominal on the other side) are extrapolated. A parameter 'deviates' when its 95%
marginal interval excludes 1.

Point estimate = marginal posterior median (the surrogate is piecewise linear, so its MAP always sits on a replayed
grid value; the MAP is shown for comparison).

| true setting | n | estimate of the true parameter: median over logs (median abs rel. error) | MAP: median (error) | true value in 95% interval | other parameter flagged as deviating | median 95% interval |
|---|---|---|---|---|---|---|
| mass_x1.5 | 10 | 1.89 (35%) | 1.99 (34%) | 100% | 20% | [0.62, 3.49] |
| mass_x2.5 | 17 | 2.06 (20%) | 2.01 (20%) | 88% | 6% | [0.64, 4.05] |
| mass_x4 | 24 | 5.07 (36%) | 3.98 (24%) | 100% | 8% | [1.61, 18.33] |
| fric_x0.7 | 3 | 0.44 (37%) | 0.50 (29%) | 100% | 33% | [0.21, 1.00] |
| fric_x0.4 | 9 | 0.44 (23%) | 0.30 (25%) | 100% | 33% | [0.30, 0.78] |
| fric_x0.25 | 11 | 0.33 (33%) | 0.30 (21%) | 55% | 0% | [0.09, 1.05] |

Compound fault (mass x2.5 + friction x0.4, 19 logs): median estimate mass x1.53, friction x0.51; both parameters flagged as deviating in 11% of logs; true mass in its interval 95%, true friction 89%.

Compared with attribution_v4's per-axis parabola fit (mass x2.5 / x4 within 10-12%, friction x0.4 within 3%,
compound 2.54 / 0.38), the point estimates here are not better: with replays only on the two axes, re-analysis
cannot locate the minimum between grid points or capture the mass-friction interaction. What the posterior adds
is an interval with stated coverage. Better point estimates need a few adaptive replays near the current
estimate (e.g. 2-3 rounds of one off-axis replay per log batch).

Friction x0.25: residual profiles along the friction axis are flat while the cube does not slip (0.5, 0.3 and
0.2 all sit near the noise floor) and jump when it does (0.2 often 100x the floor), so friction is only
identifiable from episodes where slip occurs; the intervals above show how wide that leaves it.

## 3. LLM-proposed candidates

Logs called H_other at x1 with the v5 likelihood and the class-conditional threshold: 52 (compound_mass2.5_fric0.4 3, nu_solver_iters_x0.25 18, op_fric_x0.25 2, op_fric_x0.4 1, op_fric_x0.7 2, op_mass_x1.5 1, op_mass_x4 1, other_action_delay 24).
nu_solver_iters_x0.25: 24 kept logs, 18 called H_other before any proposal; the rest: H_theta-friction 6.

**Proposer not run: not requested (--llm); ANTHROPIC_API_KEY is not set; the anthropic Python SDK is not installed in this environment.** The proposer (`llm_attribution.propose`) and the replay /
re-scoring pipeline are implemented; run `python scripts/attribution_v5.py --llm` once both are available.

### Oracle check of the pipeline (NOT an LLM result)

The true configuration (robot 2 / cube 4 solver iterations) and a near miss (4 / 8) are inserted by hand as
proposals for the nu_solver_iters_x0.25 logs, replayed and re-scored like LLM proposals would be. This checks
the executable-config path and bounds what a proposer could achieve.

Re-scored with the proposals (oracle), noise x1:

| setting | logs re-scored | H_nu (correct for solver logs) | H_other | other classes | best candidate is a proposal | median best residual / floor |
|---|---|---|---|---|---|---|
| nu_solver_iters_x0.25 | 24 | 24 | 0 | - | 24 | 1.0 |

nu_solver_iters_x0.25: 24 of the 24 kept logs attributed to H_nu after re-scoring (24 of them were re-scored).

## 4. New numerics faults (generalisation), v5 likelihood, x1

| setting | kept / total | min residual / noise floor (median) | ours (v5) | status |
|---|---|---|---|---|
| nu_substeps_x2 | 21 / 24 | 3.1 | H_theta-mass 6, H_theta-friction 15 | undetectable: indistinguishable from nominal within sensor noise (best candidate is nominal); not counted as an error |
| nu_solver_iters_x0.25 | 24 / 24 | 183.4 | H_theta-friction 6, H_other 18 | 0 correct (H_nu), 18 flagged H_other |

## 5. Factory peg insertion

NVIDIA's pretrained `Isaac-Factory-PegInsert-Direct-v0` checkpoint (published for the Isaac Sim 5.0 asset root,
212,889,115 bytes, sha256 1a138840...081496) loads on the local Isaac Lab 2.0.2 with the standard rl_games loader
(`scripts/factory_eval.py`) and succeeds in 98.8% of 2 x 128 nominal episodes (100.0% and 97.7%). No training needed.

