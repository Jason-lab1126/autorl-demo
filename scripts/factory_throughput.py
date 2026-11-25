"""Throughput of the Isaac Lab Factory peg-insertion env with random actions (no training), for estimating training time.

Usage (from ~/autorl-demo after ``source env.sh``): python scripts/factory_throughput.py
"""
import sys, time
sys.argv += ["--headless", "--device", "cuda:0"]
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
app = AppLauncher(p.parse_args()).app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
task = "Isaac-Factory-PegInsert-Direct-v0"
cfg = parse_env_cfg(task, device="cuda:0", num_envs=128)
print(f"[factory] sim dt {cfg.sim.dt}, decimation {cfg.decimation}, episode {cfg.episode_length_s}s, obs {cfg.observation_space}, state {cfg.state_space}, act {cfg.action_space}", flush=True)
env = gym.make(task, cfg=cfg)
env.reset()
act = lambda: 2 * torch.rand(128, env.unwrapped.cfg.action_space, device="cuda:0") - 1
for _ in range(20):
    env.step(act())
torch.cuda.synchronize(); t = time.time(); n = 200
for _ in range(n):
    env.step(act())
torch.cuda.synchronize(); dt = time.time() - t
print(f"[factory] {n} steps x 128 envs in {dt:.1f}s -> {n * 128 / dt:.0f} env-steps/s, max_episode_length {env.unwrapped.max_episode_length}", flush=True)
env.close(); app.close()
