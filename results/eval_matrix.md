# Lift-cube eval matrix

Run: `/home/zxu27/autorl-demo/logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline`  
Task: `Isaac-Lift-Cube-Franka-Play-v0`, 100 episodes per cell, seed 0, deterministic policy actions. Distances are cube-to-goal at the end of each episode.

## Success rate [%] (final distance < 2 cm)

| perturbation | level | value | iter_100 | iter_300 | iter_500 | iter_1499 |
|---|---|---|---|---|---|---|
| none | none | 1 | 0 | 1 | 72 | 100 |
| cube_mass | x2 | 2 | 0 | 2 | 65 | 100 |
| cube_mass | x3 | 3 | 0 | 2 | 58 | 97 |
| cube_mass | x5 | 5 | 0 | 2 | 41 | 94 |
| cube_mass | x10 | 10 | 0 | 1 | 0 | 3 |
| friction | x0.5 | 0.5 | 0 | 4 | 72 | 99 |
| friction | x0.2 | 0.2 | 0 | 1 | 29 | 65 |
| friction | x0.1 | 0.1 | 0 | 0 | 0 | 2 |
| obs_noise | x1 | 1 | 0 | 1 | 54 | 97 |
| obs_noise | x2 | 2 | 0 | 1 | 30 | 65 |
| obs_noise | x3 | 3 | 0 | 0 | 8 | 28 |
| obs_noise | x5 | 5 | 0 | 0 | 0 | 1 |
| action_noise | medium | 0.1 | 0 | 1 | 68 | 99 |
| action_noise | strong | 0.5 | 0 | 1 | 39 | 80 |
| spawn_range | medium | 1.5 | 0 | 3 | 65 | 99 |
| spawn_range | strong | 2 | 0 | 0 | 60 | 83 |

## Mean final distance [cm]

| perturbation | level | value | iter_100 | iter_300 | iter_500 | iter_1499 |
|---|---|---|---|---|---|---|
| none | none | 1 | 77.41 | 8.99 | 1.70 | 0.31 |
| cube_mass | x2 | 2 | 68.43 | 9.10 | 1.77 | 0.30 |
| cube_mass | x3 | 3 | 66.46 | 11.73 | 2.67 | 2.73 |
| cube_mass | x5 | 5 | 62.13 | 23.96 | 6.21 | 4.33 |
| cube_mass | x10 | 10 | 52.01 | 48.98 | 41.63 | 43.25 |
| friction | x0.5 | 0.5 | 72.40 | 9.93 | 2.23 | 0.44 |
| friction | x0.2 | 0.2 | 62.86 | 45.35 | 26.18 | 24.37 |
| friction | x0.1 | 0.1 | 61.95 | 60.43 | 51.77 | 60.78 |
| obs_noise | x1 | 1 | 73.49 | 8.37 | 2.08 | 0.96 |
| obs_noise | x2 | 2 | 70.00 | 10.25 | 4.53 | 3.91 |
| obs_noise | x3 | 3 | 72.68 | 13.49 | 10.37 | 12.24 |
| obs_noise | x5 | 5 | 64.82 | 43.91 | 41.39 | 42.39 |
| action_noise | medium | 0.1 | 77.87 | 8.22 | 1.70 | 0.42 |
| action_noise | strong | 0.5 | 68.66 | 8.88 | 2.83 | 3.37 |
| spawn_range | medium | 1.5 | 69.19 | 9.10 | 2.62 | 1.06 |
| spawn_range | strong | 2 | 67.21 | 17.20 | 10.15 | 11.53 |

## Perturbation values

- `cube_mass`: multiplier on the cube's mass and inertia
- `friction`: multiplier on the static and dynamic friction of the cube and the gripper fingers
- `obs_noise`: multiplier on per-step Gaussian observation noise with std {'joint_pos': 0.01, 'joint_vel': 0.1, 'object_position': 0.01}
- `action_noise`: std of per-step Gaussian noise on the raw policy actions (arm and gripper)
- `spawn_range`: multiplier on the cube's x/y spawn range
