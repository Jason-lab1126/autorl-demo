"""Success rate of an rl_games checkpoint on the Isaac Lab Factory peg-insertion task.

Loads the checkpoint the way scripts/reinforcement_learning/rl_games/play.py does (rl_games Runner +
player.restore), runs --episodes episodes per env deterministically and reports the fraction of envs
that satisfy the task's success check at the end of each episode (the env's extras["successes"]).

Usage (from ~/autorl-demo after ``source env.sh``):
    python scripts/factory_eval.py --checkpoint logs/rl_games/factory_pretrained/isaac_5.0/checkpoint.pth --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an rl_games checkpoint on Factory peg insertion.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=2, help="Episodes per env.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    path = os.path.abspath(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(path)
    agent.reset()
    print(f"[factory_eval] loaded {path}", flush=True)

    obs = env.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    rates = []
    with torch.inference_mode():
        while len(rates) < args_cli.episodes:
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if len(dones) > 0 and bool(dones.any()):
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                if bool(dones.all()) and "successes" in env.unwrapped.extras:
                    rates.append(float(env.unwrapped.extras["successes"]))
                    print(f"[factory_eval] episode {len(rates)}: success {rates[-1] * 100:.1f}% of {env.unwrapped.num_envs} envs", flush=True)
    print(f"[factory_eval] mean success over {len(rates)} x {env.unwrapped.num_envs} episodes: {sum(rates) / len(rates) * 100:.1f}%", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
