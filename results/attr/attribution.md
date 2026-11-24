# Inverse failure attribution by open-loop replay

Seed 0, 8 envs per log. ref_nominal success 100%.
Torques are Isaac Lab's `applied_torque`: a PD **estimate** under the default implicit actuators,
the torque actually applied only under explicit IdealPD (real_engine, do(numerics)).

## Confusion (rows: true cause, columns: class with the highest posterior)

| true \ predicted | H_theta-mass | H_theta-friction | H_nu | H_pi |
|---|---|---|---|---|
| H_theta-mass | 1 | 0 | 0 | 0 |
| H_theta-friction | 0 | 1 | 0 | 0 |
| H_nu | 1 | 0 | 0 | 0 |
| H_pi | 1 | 0 | 0 | 0 |

## Class posteriors per log

| log | true cause | task success | predicted | H_theta-mass | H_theta-friction | H_nu | H_pi | none |
|---|---|---|---|---|---|---|---|---|
| log_A | real_fric | 88% | H_theta-friction | 0.275 | 0.557 | 0.000 | 0.169 | 0.000 |
| log_B | real_engine | 0% | H_theta-mass | 0.309 | 0.309 | 0.279 | 0.103 | 0.000 |
| log_C | real_mass | 88% | H_theta-mass | 0.494 | 0.383 | 0.000 | 0.122 | 0.000 |
| log_D | real_policy | 0% | H_theta-mass | 0.472 | 0.283 | 0.000 | 0.245 | 0.000 |

## Residual / posterior per candidate

| log | T | nominal | mass_x2 | mass_x3 | mass_x5 | fric_x0.5 | fric_x0.3 | fric_x0.2 | numerics |
|---|---|---|---|---|---|---|---|---|---|
| log_A | 0.0385 | 0.0305 / 0.169 | 0.0376 / 0.140 | 0.0393 / 0.134 | 0.948 / 0.000 | 0.0172 / 0.238 | 0.00615 / 0.318 | 0.282 / 0.000 | 67.9 / 0.000 |
| log_B | 57.8 | 57.8 / 0.103 | 57.8 / 0.103 | 57.8 / 0.103 | 57.8 / 0.103 | 57.8 / 0.103 | 57.9 / 0.103 | 57.9 / 0.103 | 0.085 / 0.279 |
| log_C | 0.119 | 0.128 / 0.122 | 0.11 / 0.142 | 0.0024 / 0.352 | 0.931 / 0.000 | 0.107 / 0.146 | 0.0502 / 0.235 | 0.606 / 0.002 | 67.1 / 0.000 |
| log_D | 1.08 | 0.0003 / 0.245 | 0.0201 / 0.240 | 0.147 / 0.214 | 2.77 / 0.019 | 0.019 / 0.240 | 2.01 / 0.038 | 4.28 / 0.005 | 7.11 / 0.000 |

## Diagnostic: class of the single lowest-residual candidate (not the specified posterior rule)

Summing posteriors over classes favours classes with more candidates (mass and friction have 3 each,
numerics and nominal 1), and T = median residual flattens the posterior when one candidate is far ahead.

| log | true cause | min-residual candidate | class | second-lowest / lowest residual |
|---|---|---|---|---|
| log_A | real_fric | fric_x0.3 | H_theta-friction | 2.8 |
| log_B | real_engine | numerics | H_nu | 680 |
| log_C | real_mass | mass_x3 | H_theta-mass | 20.9 |
| log_D | real_policy | nominal | H_pi | 63.3 |

Coverage (both-alive steps / log alive steps) is in residuals.csv.
