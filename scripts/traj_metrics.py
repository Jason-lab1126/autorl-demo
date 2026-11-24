"""Numpy-only helpers shared by the attribution scripts: success, residuals, divergence.

A trajectory is the dict lift_sim.LiftSim.rollout returns (also what its .npz files hold): arrays of
shape (step, env, ...) for joint_pos, joint_vel, applied_torque, computed_torque, cube_pos, action
and alive, plus goal_pos (env, 3).
"""

import numpy as np

SUCCESS_RADIUS = 0.02  # [m]


def final_success(traj: dict, threshold: float = SUCCESS_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Cube-to-goal distance at each env's last alive step, and whether it is within ``threshold``."""
    alive = traj["alive"]
    last = alive.shape[0] - 1 - np.argmax(alive[::-1], axis=0)
    final_pos = traj["cube_pos"][last, np.arange(alive.shape[1])]
    final_dist = np.linalg.norm(final_pos - traj["goal_pos"], axis=1)
    return final_dist, final_dist < threshold


def joint_sd(traj: dict, torque_key: str = "applied_torque") -> tuple[np.ndarray, np.ndarray]:
    """Per-joint std of a torque signal and of joint position over alive steps (residual normalisation)."""
    alive = traj["alive"]
    return (
        np.maximum(traj[torque_key][alive].std(axis=0), 1e-6),
        np.maximum(traj["joint_pos"][alive].std(axis=0), 1e-6),
    )


def residual_per_env(log: dict, rep: dict, sd_tau: np.ndarray, sd_q: np.ndarray,
                     torque_key: str | None = "applied_torque") -> dict[str, np.ndarray]:
    """Per-env residual of a replay against a log, over the steps where both are alive.

    residual = mean over steps of [ mean_j (dtau_j / sd_tau_j)^2 ] + mean over steps of [ mean_j (dq_j / sd_q_j)^2 ]
    with tau = ``torque_key`` (None: joint positions only). Also returns the two terms and the coverage
    (both-alive steps / log alive steps).
    """
    steps = min(len(log["alive"]), len(rep["alive"]))
    mask = (log["alive"][:steps] & rep["alive"][:steps]).astype(float)  # (step, env)
    n = np.maximum(mask.sum(axis=0), 1.0)
    if torque_key is None:
        z_tau = np.zeros(mask.shape)
    else:
        z_tau = (((rep[torque_key][:steps] - log[torque_key][:steps]) / sd_tau) ** 2).mean(axis=2)
    z_q = (((rep["joint_pos"][:steps] - log["joint_pos"][:steps]) / sd_q) ** 2).mean(axis=2)
    tau_term = (z_tau * mask).sum(axis=0) / n
    q_term = (z_q * mask).sum(axis=0) / n
    return {
        "residual": tau_term + q_term, "torque_term": tau_term, "pos_term": q_term,
        "coverage": mask.sum(axis=0) / np.maximum(log["alive"].sum(axis=0), 1),
    }


def first_divergence(a: dict, b: dict, keys=("joint_pos", "joint_vel", "applied_torque", "cube_pos")) -> dict:
    """For each key: first step at which any env differs (bitwise), and the max |difference| over both-alive steps."""
    steps = min(len(a["alive"]), len(b["alive"]))
    mask = a["alive"][:steps] & b["alive"][:steps]
    out = {}
    for key in keys:
        diff = np.abs(a[key][:steps] - b[key][:steps]).reshape(steps, mask.shape[1], -1).max(axis=2)
        diff = np.where(mask, diff, 0.0)
        differs = np.nonzero(diff.max(axis=1) > 0.0)[0]
        out[key] = (int(differs[0]) if len(differs) else None, float(diff.max()))
    return out


def add_sensor_noise(traj: dict, q_std: float, tau_std: float, rng: np.random.Generator,
                     torque_keys: tuple[str, ...] = ("applied_torque",)) -> dict:
    """Copy of ``traj`` with Gaussian noise on the joint positions and on the given torque signals."""
    noisy = dict(traj)
    if q_std > 0.0:
        noisy["joint_pos"] = traj["joint_pos"] + rng.normal(0.0, q_std, traj["joint_pos"].shape)
    for key in torque_keys if tau_std > 0.0 else ():
        if key in traj:
            noisy[key] = traj[key] + rng.normal(0.0, tau_std, traj[key].shape)
    return noisy
