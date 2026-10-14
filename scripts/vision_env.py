"""Lift-cube environment with one fixed RGB camera per environment.

Shared by data collection and closed-loop student evaluation, so both see exactly the same images.
Import this only after the simulator app has been launched.
"""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Lift-Cube-Franka-Play-v0"
INSTRUCTION = "pick up the cube and move it to the target"
FPS = 50

# camera pose in each environment's frame (robot base at the origin, table along +x)
CAMERA_EYE = (1.1, 0.7, 0.7)
CAMERA_LOOKAT = (0.45, 0.0, 0.35)
CAMERA_FOCAL_LENGTH = 26.0  # with the 20.955 aperture below: about 44 degrees field of view

# layout of the teacher's observation vector
TEACHER_OBS_TERMS = ["joint_pos", "joint_vel", "object_position", "target_object_position", "actions"]
TEACHER_OBS_DIMS = [9, 9, 3, 7, 8]
# the student sees joint positions, joint velocities and the goal position; never the cube position
STATE_NAMES = (
    [f"joint_pos_{i}" for i in range(9)] + [f"joint_vel_{i}" for i in range(9)] + ["goal_x", "goal_y", "goal_z"]
)
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = 8


def make_vision_env(
    num_envs: int,
    device: str,
    seed: int,
    image_size: int = 84,
    eye: tuple = CAMERA_EYE,
    lookat: tuple = CAMERA_LOOKAT,
    focal_length: float = CAMERA_FOCAL_LENGTH,
    viewer_eye: tuple | None = None,
    viewer_lookat: tuple | None = None,
    viewer_env_index: int = 0,
):
    """Creates the lift-cube environment (wrapped for rsl_rl) with a camera added to every environment.

    Passing ``viewer_eye`` additionally enables ``base_env.render()``: an external 1280x720 view for
    videos, placed relative to environment ``viewer_env_index``. It does not affect the student's camera.
    """
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    # render again after a reset, so the first image of an episode shows the new episode
    env_cfg.rerender_on_reset = True
    # the goal and gripper-frame markers would give the goal away in the image
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.scene.ee_frame.debug_vis = False
    env_cfg.scene.camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=image_size,
        height=image_size,
    )
    if viewer_eye is not None:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = viewer_env_index
        env_cfg.viewer.eye = tuple(viewer_eye)
        env_cfg.viewer.lookat = tuple(viewer_lookat)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if viewer_eye is not None else None))
    base_env = env.unwrapped

    terms = base_env.observation_manager.active_terms["policy"]
    dims = [int(torch.tensor(d).prod()) for d in base_env.observation_manager.group_obs_term_dim["policy"]]
    if terms != TEACHER_OBS_TERMS or dims != TEACHER_OBS_DIMS:
        raise RuntimeError(f"Unexpected observation layout: {list(zip(terms, dims))}")

    origins = base_env.scene.env_origins
    eyes = origins + torch.tensor(eye, device=origins.device)
    targets = origins + torch.tensor(lookat, device=origins.device)
    base_env.scene["camera"].set_world_poses_from_view(eyes, targets)
    base_env.reset(seed=seed)
    return env


def get_images(base_env) -> torch.Tensor:
    """Current camera images as (num_envs, H, W, 3) uint8."""
    rgb = base_env.scene["camera"].data.output["rgb"][..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb


def student_state(teacher_obs: torch.Tensor) -> torch.Tensor:
    """Student state input: joint positions, joint velocities and the goal position."""
    return torch.cat([teacher_obs[:, :18], teacher_obs[:, 21:24]], dim=1)


def cube_position(teacher_obs: torch.Tensor) -> torch.Tensor:
    """Cube position in the robot base frame (privileged: recorded, never given to the student)."""
    return teacher_obs[:, 18:21]


class EpisodeEndRecorder:
    """Records the cube-to-goal distance at the end of each environment's first episode.

    The env resets terminated sub-environments inside step(), so the final state is gone by the
    time step() returns. This captures it right before the reset instead.
    """

    def __init__(self, base_env, command_name: str = "object_pose"):
        self.base_env = base_env
        self.command_name = command_name
        self.robot = base_env.scene["robot"]
        self.cube = base_env.scene["object"]
        num_envs, device = base_env.num_envs, base_env.device
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.final_dist = torch.zeros(num_envs, device=device)
        self.final_height = torch.zeros(num_envs, device=device)
        self.timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.goal_start_w = self.goal_pos_w().clone()
        self._original_reset_idx = base_env._reset_idx
        base_env._reset_idx = self._reset_idx

    def goal_pos_w(self) -> torch.Tensor:
        command = self.base_env.command_manager.get_command(self.command_name)
        root = self.robot.data.root_state_w
        pos_w, _ = combine_frame_transforms(root[:, :3], root[:, 3:7], command[:, :3])
        return pos_w

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.base_env.num_envs, device=self.base_env.device)[env_ids]
        ids = ids[~self.done[ids]]
        cube_pos_w = self.cube.data.root_pos_w
        self.final_dist[ids] = torch.norm(self.goal_start_w[ids] - cube_pos_w[ids], dim=1)
        self.final_height[ids] = cube_pos_w[ids, 2] - self.base_env.scene.env_origins[ids, 2]
        self.timed_out[ids] = self.base_env.termination_manager.time_outs[ids]
        self.steps[ids] = self.base_env.episode_length_buf[ids]
        self.done[ids] = True
        self._original_reset_idx(env_ids)
