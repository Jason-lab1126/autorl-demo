# Lift-cube eval matrix

Run: `/home/zxu27/autorl-demo/logs/rsl_rl/franka_lift/2026-10-13_17-10-08_ppo_obs_noise_resume`  
Task: `Isaac-Lift-Cube-Franka-Play-v0`, 100 episodes per cell, seed 0, deterministic policy actions. Distances are cube-to-goal at the end of each episode.

## Success rate [%] (final distance < 2 cm)

| perturbation | level | value | iter_2999 |
|---|---|---|---|
| none | none | 1 | 63 |
| obs_noise | x1 | 1 | 69 |
| obs_noise | x2 | 2 | 56 |
| obs_noise | x3 | 3 | 49 |
| obs_noise | x5 | 5 | 15 |

## Mean final distance [cm]

| perturbation | level | value | iter_2999 |
|---|---|---|---|
| none | none | 1 | 1.93 |
| obs_noise | x1 | 1 | 1.88 |
| obs_noise | x2 | 2 | 2.88 |
| obs_noise | x3 | 3 | 4.58 |
| obs_noise | x5 | 5 | 14.42 |

## Perturbation values

- `cube_mass`: multiplier on the cube's mass and inertia
- `friction`: multiplier on the static and dynamic friction of the cube and the gripper fingers
- `obs_noise`: multiplier on per-step Gaussian observation noise with std {'joint_pos': 0.01, 'joint_vel': 0.1, 'object_position': 0.01}
- `action_noise`: std of per-step Gaussian noise on the raw policy actions (arm and gripper)
- `spawn_range`: multiplier on the cube's x/y spawn range
