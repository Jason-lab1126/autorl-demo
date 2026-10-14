# Lift-cube eval matrix

Run: `/home/zxu27/autorl-demo/logs/rsl_rl/franka_lift/2026-10-13_16-36-46_ppo_obs_noise`  
Task: `Isaac-Lift-Cube-Franka-Play-v0`, 100 episodes per cell, seed 0, deterministic policy actions. Distances are cube-to-goal at the end of each episode.

## Success rate [%] (final distance < 2 cm)

| perturbation | level | value | iter_1499 |
|---|---|---|---|
| none | none | 1 | 4 |
| obs_noise | x1 | 1 | 5 |
| obs_noise | x2 | 2 | 15 |
| obs_noise | x3 | 3 | 18 |
| obs_noise | x5 | 5 | 16 |

## Mean final distance [cm]

| perturbation | level | value | iter_1499 |
|---|---|---|---|
| none | none | 1 | 3.49 |
| obs_noise | x1 | 1 | 3.58 |
| obs_noise | x2 | 2 | 3.19 |
| obs_noise | x3 | 3 | 3.35 |
| obs_noise | x5 | 5 | 10.76 |

## Perturbation values

- `cube_mass`: multiplier on the cube's mass and inertia
- `friction`: multiplier on the static and dynamic friction of the cube and the gripper fingers
- `obs_noise`: multiplier on per-step Gaussian observation noise with std {'joint_pos': 0.01, 'joint_vel': 0.1, 'object_position': 0.01}
- `action_noise`: std of per-step Gaussian noise on the raw policy actions (arm and gripper)
- `spawn_range`: multiplier on the cube's x/y spawn range
