"""Run Isaac Lab's stock rsl_rl play.py unmodified, with a compatibility shim.

Isaac Lab 2.0.2's play.py reads ``runner.alg.actor_critic`` when exporting the policy, but the
installed rsl-rl-lib (2.3.3) renamed that attribute to ``policy``. Alias it here instead of
editing Isaac Lab or downgrading the library the checkpoint was trained with.

All arguments are passed straight through to play.py.
"""

import os
import runpy
import sys

from rsl_rl.algorithms import PPO

if not hasattr(PPO, "actor_critic"):
    PPO.actor_critic = property(lambda self: self.policy)

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
play_script = os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl", "play.py")
sys.path.insert(0, os.path.dirname(play_script))
sys.argv[0] = play_script
runpy.run_path(play_script, run_name="__main__")
