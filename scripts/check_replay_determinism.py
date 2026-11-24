"""Where does open-loop replay stop reproducing a log bit for bit?

--mode record (one process): records a nominal closed-loop log as the process's first rollout, then
replays its actions at nominal physics after a series of different histories:
    direct              right after the log
    mass_set_reset      after setting cube mass x5 and back, no stepping in between
    fric_set_reset      after setting friction x0.3 and back, no stepping in between
    mass_rollout        after a closed-loop rollout at mass x5 (then back to nominal)
    fric_rollout        after a closed-loop rollout at friction x0.3 (then back to nominal)
    other_policy        after a nominal closed-loop rollout with model_300 (no physics change)
    other_seed          after a nominal closed-loop rollout from seed 1 (no physics change)
    closed_loop_again   the closed-loop log itself re-run at the end
--mode fresh (new process): replays the saved log as that process's first rollout, then re-runs it
closed loop.
--mode report: compares everything with the log.

Usage (from ~/autorl-demo after ``source env.sh``; add --enhanced_determinism to test PhysX's flag and
--preroll N to put the scene through a canonical N-step pre-roll before every rollout):
    python scripts/check_replay_determinism.py --mode record --headless --device cuda:0
    python scripts/check_replay_determinism.py --mode fresh --headless --device cuda:0
    python scripts/check_replay_determinism.py --mode report
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.append(os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Replay determinism check.")
parser.add_argument("--mode", type=str, required=True, choices=["record", "fresh", "report"])
parser.add_argument("--enhanced_determinism", action="store_true", help="Set PhysX enable_enhanced_determinism.")
parser.add_argument("--preroll", type=int, default=0, help="Canonical pre-roll steps before every rollout.")
parser.add_argument("--run_dir", type=str, default="logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="results/attr/determinism")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402

from traj_metrics import first_divergence, joint_sd, residual_per_env  # noqa: E402

OUT_DIR = os.path.join(
    args_cli.out_dir,
    ("enhanced" if args_cli.enhanced_determinism else "default") + (f"_preroll{args_cli.preroll}" if args_cli.preroll else ""),
)


def simulate():
    simulation_app = AppLauncher(args_cli).app
    import torch

    from lift_sim import LiftSim, load_traj, save_traj

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    sim = LiftSim(args_cli.task, args_cli.num_envs, args_cli.device, agent_cfg,
                  enhanced_determinism=args_cli.enhanced_determinism, preroll_steps=args_cli.preroll)
    ckpt = lambda it: os.path.join(args_cli.run_dir, f"model_{it}.pt")  # noqa: E731
    policy = sim.load_policy(ckpt(1499))
    save = lambda name, traj: save_traj(os.path.join(OUT_DIR, f"{name}.npz"), traj, **sim.meta(0))  # noqa: E731

    if args_cli.mode == "record":
        log = sim.rollout(0, policy=policy)
        save("log", log)

        def replay(name):
            sim.apply_perturbation()
            save(name, sim.rollout(0, replay=log))

        replay("direct")
        view = sim.cube.root_physx_view
        sim.apply_perturbation("cube_mass", 5.0)
        sim.apply_perturbation()
        print("[determinism] masses restored bitwise:", torch.equal(view.get_masses(), sim.default_masses),
              " inertias:", torch.equal(view.get_inertias(), sim.default_inertias), flush=True)
        replay("mass_set_reset")
        sim.apply_perturbation("friction", 0.3)
        sim.apply_perturbation()
        print("[determinism] materials restored bitwise: cube",
              torch.equal(view.get_material_properties(), sim.default_materials), " robot",
              torch.equal(sim.robot.root_physx_view.get_material_properties(), sim.default_robot_materials), flush=True)
        replay("fric_set_reset")
        sim.apply_perturbation("cube_mass", 5.0)
        sim.rollout(0, policy=policy)
        replay("mass_rollout")
        sim.apply_perturbation("friction", 0.3)
        sim.rollout(0, policy=policy)
        replay("fric_rollout")
        sim.rollout(0, policy=sim.load_policy(ckpt(300)))
        policy = sim.load_policy(ckpt(1499))
        replay("other_policy")
        sim.rollout(1, policy=policy)
        replay("other_seed")
        save("closed_loop_again", sim.rollout(0, policy=policy))
    else:
        log = load_traj(os.path.join(OUT_DIR, "log.npz"))
        save("fresh_replay", sim.rollout(0, replay=log))
        save("fresh_closed_loop", sim.rollout(0, policy=policy))
    sim.env.close()
    simulation_app.close()


def report():
    order = ["direct", "mass_set_reset", "fric_set_reset", "mass_rollout", "fric_rollout", "other_policy",
             "other_seed", "closed_loop_again", "fresh_replay", "fresh_closed_loop"]
    for variant in sorted(glob.glob(os.path.join(args_cli.out_dir, "*", "log.npz"))):
        folder = os.path.dirname(variant)
        log = dict(np.load(variant))
        sd_tau, sd_q = joint_sd(log)
        print(f"\n[{os.path.basename(folder)}] replay vs log (residual = mean over envs; first differing step / max |diff|)")
        print(f"  {'case':<18} {'residual':>10} {'max|dq|':>10}  first diff: q, qd, tau, cube")
        for name in order:
            path = os.path.join(folder, f"{name}.npz")
            if not os.path.isfile(path):
                continue
            rep = dict(np.load(path))
            res = residual_per_env(log, rep, sd_tau, sd_q)["residual"].mean()
            div = first_divergence(log, rep)
            firsts = ", ".join("-" if div[k][0] is None else str(div[k][0]) for k in div)
            print(f"  {name:<18} {res:10.3e} {div['joint_pos'][1]:10.3e}  {firsts}")


if __name__ == "__main__":
    report() if args_cli.mode == "report" else simulate()
