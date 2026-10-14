"""Observation-noise model shared by training-time randomization and the eval matrix."""

import torch

# Base noise std per observation term; terms not listed (goal command, last action) stay noise-free.
OBS_NOISE_STD = {"joint_pos": 0.01, "joint_vel": 0.1, "object_position": 0.01}  # rad, rad/s, m


def obs_noise_std_vector(base_env, group: str = "policy") -> torch.Tensor:
    """Per-dimension base noise std, laid out like the flattened observation of ``group``."""
    terms = base_env.observation_manager.active_terms[group]
    dims = [int(torch.tensor(d).prod()) for d in base_env.observation_manager.group_obs_term_dim[group]]
    unknown = set(OBS_NOISE_STD) - set(terms)
    if unknown:
        raise KeyError(f"OBS_NOISE_STD names terms the policy does not observe: {unknown}")
    return torch.cat(
        [torch.full((dim,), OBS_NOISE_STD.get(name, 0.0), device=base_env.device) for name, dim in zip(terms, dims)]
    )
