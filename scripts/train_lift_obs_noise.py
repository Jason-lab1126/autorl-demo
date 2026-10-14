"""Train with observation-noise randomization, using Isaac Lab's stock rsl_rl train.py unmodified.

Every environment draws a noise multiplier uniformly from [0, OBS_NOISE_MAX] at the start of each
episode. Each step, Gaussian noise with std ``OBS_NOISE_STD * multiplier`` (see obs_noise.py, the
same model the eval matrix sweeps) is added to the observations the agent receives. The task has a
single observation group, so actor and critic both see the noisy observations.

The noise is attached by wrapping the environment as it is handed to the PPO runner; task, PPO
settings, logging and checkpointing are exactly those of train.py. All arguments are passed
straight through to train.py. Set OBS_NOISE_MAX in the environment to change the range (default 3).
"""

import json
import os
import runpy
import sys

import torch
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_noise import OBS_NOISE_STD, obs_noise_std_vector  # noqa: E402

OBS_NOISE_MAX = float(os.environ.get("OBS_NOISE_MAX", "3.0"))


def add_obs_noise(env, log_dir):
    """Makes ``env`` (an RslRlVecEnvWrapper) return noisy observations."""
    std = obs_noise_std_vector(env.unwrapped)
    multiplier = torch.rand(env.num_envs, 1, device=std.device) * OBS_NOISE_MAX
    base_cls = env.__class__

    def noisy(obs):
        return obs + torch.randn_like(obs) * std * multiplier

    class ObsNoiseEnv(base_cls):
        def get_observations(self):
            obs, extras = base_cls.get_observations(self)
            return noisy(obs), extras

        def step(self, actions):
            obs, rew, dones, extras = base_cls.step(self, actions)
            # environments that just reset start a new episode with a new noise level
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            multiplier[reset_ids] = torch.rand(len(reset_ids), 1, device=std.device) * OBS_NOISE_MAX
            return noisy(obs), rew, dones, extras

    env.__class__ = ObsNoiseEnv
    print(f"[train_lift_obs_noise] observation noise multiplier ~ U(0, {OBS_NOISE_MAX:g}) per episode, std {OBS_NOISE_STD}")
    if log_dir is not None:
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        with open(os.path.join(log_dir, "params", "obs_noise.json"), "w") as f:
            json.dump({"multiplier_range": [0.0, OBS_NOISE_MAX], "base_std": OBS_NOISE_STD, "resampled": "per episode"}, f, indent=2)


original_init = OnPolicyRunner.__init__


def init_with_obs_noise(self, env, train_cfg, log_dir=None, device="cpu"):
    add_obs_noise(env, log_dir)
    original_init(self, env, train_cfg, log_dir=log_dir, device=device)


OnPolicyRunner.__init__ = init_with_obs_noise

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
train_script = os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl", "train.py")
sys.path.insert(0, os.path.dirname(train_script))
sys.argv[0] = train_script
runpy.run_path(train_script, run_name="__main__")
