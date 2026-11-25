"""Shared simulator harness for the lift-cube attribution scripts: env creation, physics perturbations
and closed-/open-loop rollouts with per-step logging.

Import this only after the Isaac Sim app has been launched (AppLauncher).

Perturbations follow eval_lift.py: cube mass (inertia scaled with it), and the static and dynamic
friction of the cube and of the gripper-finger collision shapes. A rollout resets every env with the
given seed and runs one episode per env. Step t holds the state after action t; on the step where an
env terminates it holds the last state before the reset (captured in a _reset_idx hook). ``alive`` is
True up to and including that terminal step. Replays are also not alive past the replayed log's own
episode, since the recorded actions after it belong to the next episode.
"""

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from explicit_actuators import use_explicit_actuators
from traj_metrics import final_success

FINGER_BODIES = "panda_.*finger"
PREROLL_SEED = 1_000_003  # seed of the canonical pre-roll episode (see LiftSim.rollout)
# tau_solver: PhysX's projected joint forces (get_dof_projected_joint_forces), the joint force the solver
# actually transmits along each DOF; applied_torque is Isaac Lab's PD estimate under implicit actuators.
STATE_KEYS = ["joint_pos", "joint_vel", "applied_torque", "computed_torque", "tau_solver", "cube_pos"]
ARM_JOINTS = 7


class LiftSim:
    def __init__(self, task: str, num_envs: int, device: str, agent_cfg, seed: int = 0,
                 explicit_actuators: bool | list[str] = False, enhanced_determinism: bool = False,
                 preroll_steps: int = 0, q_bias: float = 0.0, action_delay: list[int] | None = None,
                 gripper_effort_scale: float = 1.0, substep_scale: int = 1, solver_iter_scale: float = 1.0,
                 robot_solver_iters: int | None = None, cube_solver_iters: int | None = None):
        """Injected faults (all off by default):
        explicit_actuators   True for all actuator groups, or a list of groups, as explicit IdealPD
        q_bias               constant offset [rad] on the measured arm joint positions: seen by the policy and
                             written to the log, while the joint controllers act on the true state
        action_delay         per-env delay in control steps (cycled over the envs); the robot executes the command
                             from that many steps ago (zero action before that) while the log records the command
        gripper_effort_scale multiplier on the finger drives' effort limit
        substep_scale        physics steps per control step x this (physics dt / this, same control rate)
        solver_iter_scale    multiplier on the robot's and the cube's solver position iteration counts
        robot_solver_iters / cube_solver_iters   absolute solver position iteration counts (override the scale)
        """
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        env_cfg.seed = seed
        if explicit_actuators:
            groups = None if explicit_actuators is True else list(explicit_actuators)
            print(f"[lift_sim] explicit IdealPD actuators for {use_explicit_actuators(env_cfg, groups)}")
        if substep_scale != 1:
            env_cfg.sim.dt /= substep_scale
            env_cfg.decimation *= substep_scale
            env_cfg.sim.render_interval = env_cfg.decimation
        if solver_iter_scale != 1.0:
            for props in (env_cfg.scene.robot.spawn.articulation_props, env_cfg.scene.object.spawn.rigid_props):
                props.solver_position_iteration_count = max(1, round(props.solver_position_iteration_count * solver_iter_scale))
        if robot_solver_iters is not None:
            env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = int(robot_solver_iters)
        if cube_solver_iters is not None:
            env_cfg.scene.object.spawn.rigid_props.solver_position_iteration_count = int(cube_solver_iters)
        if gripper_effort_scale != 1.0:
            hand = env_cfg.scene.robot.actuators["panda_hand"]
            env_cfg.scene.robot.actuators["panda_hand"] = hand.replace(effort_limit=hand.effort_limit * gripper_effort_scale)
        env_cfg.sim.physx.enable_enhanced_determinism = enhanced_determinism
        self.explicit_actuators = explicit_actuators
        self.enhanced_determinism = enhanced_determinism
        self.preroll_steps = preroll_steps
        self.faults = {"q_bias": q_bias, "action_delay": action_delay or [0], "gripper_effort_scale": gripper_effort_scale}
        self._perturbation = ("none", 1.0)

        self.env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg))
        self.base_env = self.env.unwrapped
        self.num_envs, self.device = self.base_env.num_envs, self.base_env.device
        self.runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        self.robot = self.base_env.scene["robot"]
        self.cube = self.base_env.scene["object"]
        self.joint_names = np.array(self.robot.joint_names)
        self._all_ids_cpu = torch.arange(self.num_envs, dtype=torch.int32, device="cpu")
        self._q_bias = torch.zeros(len(self.joint_names), device=self.device)
        self._q_bias[:ARM_JOINTS] = q_bias
        delays = action_delay or [0]
        self._delay = torch.tensor([delays[i % len(delays)] for i in range(self.num_envs)], device=self.device)
        # where the policy observation holds the joint positions (the lift task's "joint_pos" term)
        obs_mgr = self.base_env.observation_manager
        names, dims = obs_mgr.active_terms["policy"], obs_mgr.group_obs_term_dim["policy"]
        start = sum(d[0] for d in dims[: names.index("joint_pos")])
        self._obs_q = slice(start, start + ARM_JOINTS)
        self._setup_perturbations()
        self._setup_reset_hook()

    # perturbations

    def _setup_perturbations(self):
        robot, cube_view = self.robot, self.cube.root_physx_view
        self.default_masses = cube_view.get_masses().clone()
        self.default_inertias = cube_view.get_inertias().clone()
        self.default_materials = cube_view.get_material_properties().clone()
        self.default_robot_materials = robot.root_physx_view.get_material_properties().clone()
        # the articulation view exposes materials per collision shape, flattened over links
        shapes_per_body = [
            robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
            for link_path in robot.root_physx_view.link_paths[0]
        ]
        if sum(shapes_per_body) != robot.root_physx_view.max_shapes:
            raise RuntimeError("Could not map the robot's collision shapes to its links.")
        finger_ids, finger_names = robot.find_bodies(FINGER_BODIES)
        self.finger_shape_ids = [
            shape_id
            for body_id in finger_ids
            for shape_id in range(sum(shapes_per_body[:body_id]), sum(shapes_per_body[: body_id + 1]))
        ]
        if not self.finger_shape_ids:
            raise RuntimeError(f"No collision shapes found on the gripper fingers {finger_names}.")

    def apply_perturbation(self, name: str = "none", value=1.0):
        """Sets cube mass ("cube_mass") or cube+finger friction ("friction") to ``value`` x nominal, or both
        ("mass_friction", value = (mass multiplier, friction multiplier)); "none" restores."""
        if name not in ("none", "cube_mass", "friction", "mass_friction"):
            raise ValueError(f"Unknown perturbation {name}.")
        self._perturbation = (name, value)
        if name == "mass_friction":
            mass_scale, friction_scale = value
        else:
            mass_scale = value if name == "cube_mass" else 1.0
            friction_scale = value if name == "friction" else 1.0
        cube_view, ids = self.cube.root_physx_view, self._all_ids_cpu
        cube_view.set_masses(self.default_masses * mass_scale, ids)
        cube_view.set_inertias(self.default_inertias * mass_scale, ids)
        materials = self.default_materials.clone()
        materials[..., :2] *= friction_scale
        cube_view.set_material_properties(materials, ids)
        robot_materials = self.default_robot_materials.clone()
        robot_materials[:, self.finger_shape_ids, :2] *= friction_scale
        self.robot.root_physx_view.set_material_properties(robot_materials, ids)

    # rollouts

    def _snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "joint_pos": self.robot.data.joint_pos + self._q_bias,
            "joint_vel": self.robot.data.joint_vel.clone(),
            "applied_torque": self.robot.data.applied_torque.clone(),
            "computed_torque": self.robot.data.computed_torque.clone(),
            "tau_solver": self.robot.root_physx_view.get_dof_projected_joint_forces().clone(),
            "cube_pos": (self.cube.data.root_pos_w - self.base_env.scene.env_origins).clone(),
        }

    def _setup_reset_hook(self):
        # the env resets terminated sub-environments inside step(); keep their last pre-reset state
        self._recording = False
        self._done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_now = torch.zeros_like(self._done)
        self._pre_reset = self._snapshot()
        original_reset_idx = self.base_env._reset_idx

        def recording_reset_idx(env_ids):
            if self._recording:
                ids = torch.arange(self.num_envs, device=self.device)[env_ids]
                ids = ids[~self._done[ids]]
                for key, value in self._snapshot().items():
                    self._pre_reset[key][ids] = value[ids]
                self._reset_now[ids] = True
            original_reset_idx(env_ids)

        self.base_env._reset_idx = recording_reset_idx

    def load_policy(self, checkpoint: str):
        self.runner.load(checkpoint)
        return self.runner.get_inference_policy(device=self.device)

    def _goal_pos(self) -> torch.Tensor:
        """Commanded cube position relative to the env origin (commands are in the robot base frame)."""
        command = self.base_env.command_manager.get_command("object_pose")
        root = self.robot.data.root_state_w
        pos_w, _ = combine_frame_transforms(root[:, :3], root[:, 3:7], command[:, :3])
        return pos_w - self.base_env.scene.env_origins

    @torch.inference_mode()
    def rollout(self, seed: int, policy=None, replay: dict | None = None) -> dict[str, np.ndarray]:
        """One episode per env: closed loop with ``policy``, or open loop replaying ``replay['action']``."""
        if (policy is None) == (replay is None):
            raise ValueError("Pass exactly one of policy and replay.")
        env, base_env, done, reset_now = self.env, self.base_env, self._done, self._reset_now
        if self.preroll_steps > 0:
            # GPU PhysX is only reproducible for an identical history: a replay after a different previous
            # rollout diverges at first gripper-cube contact. Put the scene in the same state before every
            # rollout: nominal physics, a fixed seed, zero actions (arm at home, gripper open, cube untouched).
            perturbation = self._perturbation
            self.apply_perturbation()
            base_env.reset(seed=PREROLL_SEED)
            zeros = torch.zeros(self.num_envs, base_env.action_manager.total_action_dim, device=self.device)
            for _ in range(self.preroll_steps):
                env.step(zeros)
            self.apply_perturbation(*perturbation)
        base_env.reset(seed=seed)
        obs, _ = env.get_observations()
        done[:] = False
        out = {f"init_{key}": value.cpu().numpy() for key, value in self._snapshot().items()}
        goal = self._goal_pos()
        actions_in = None if replay is None else torch.from_numpy(replay["action"]).to(self.device)
        max_steps = base_env.max_episode_length + 5 if replay is None else len(actions_in)
        log = {key: [] for key in STATE_KEYS + ["action", "alive"]}
        max_delay = int(self._delay.max())
        commands = []  # past commands, for the action delay
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._recording = True
        for t in range(max_steps):
            if replay is None:
                if bool(self._q_bias.any()):
                    obs = obs.clone()
                    obs[:, self._obs_q] += self._q_bias[:ARM_JOINTS]
                actions = policy(obs)
            else:
                actions = actions_in[t]
            executed = actions
            if replay is None and max_delay > 0:
                commands.append(actions.clone())
                history = torch.stack([torch.zeros_like(actions)] * max_delay + commands)  # index max_delay + t is now
                executed = history[max_delay + t - self._delay, env_ids]
            alive = ~done
            reset_now[:] = False
            obs, _, _, _ = env.step(executed)
            state = self._snapshot()
            for key in state:
                state[key][reset_now] = self._pre_reset[key][reset_now]
                log[key].append(state[key])
            log["action"].append(actions.clone())
            log["alive"].append(alive.clone())
            done[reset_now] = True
            if replay is None and bool(done.all()):
                break
        self._recording = False
        if replay is None and not bool(done.all()):
            raise RuntimeError("Some environments never finished their episode.")
        out.update({key: torch.stack(values).cpu().numpy() for key, values in log.items()})
        if replay is not None:
            out["alive"] &= replay["alive"]
        out["goal_pos"] = goal.cpu().numpy()
        out["final_dist"], out["success"] = final_success(out)
        return out

    def meta(self, seed: int) -> dict:
        return {
            "joint_names": self.joint_names, "seed": seed, "explicit_actuators": self.explicit_actuators,
            "enhanced_determinism": self.enhanced_determinism, "preroll_steps": self.preroll_steps,
            **{f"fault_{k}": np.asarray(v) for k, v in self.faults.items()},
        }


def save_traj(path: str, traj: dict, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **traj, **meta)


def load_traj(path: str) -> dict:
    with np.load(path) as f:
        return {key: f[key] for key in f.files}
