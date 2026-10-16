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

# with randomizable=True, environments whose index modulo 5 equals these keep the original cube/table look
KEEP_ORIGINAL_CUBE = 0
KEEP_ORIGINAL_TABLE = 2


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
    light_scale: float = 1.0,
    cube_color: tuple | None = None,
    table_color: tuple | None = None,
    camera_shift: tuple = (0.0, 0.0, 0.0),
    randomizable: bool = False,
    crop_render_size: int = 0,
):
    """Creates the lift-cube environment (wrapped for rsl_rl) with a camera added to every environment.

    Passing ``viewer_eye`` additionally enables ``base_env.render()``: an external 1280x720 view for
    videos, placed relative to environment ``viewer_env_index``. It does not affect the student's camera.

    Visual perturbations (they change only what the camera sees, not the physics):
    ``light_scale`` multiplies the scene light's intensity, ``cube_color`` and ``table_color`` replace
    the cube's and table's materials with one flat RGB color, and ``camera_shift`` moves the student's
    camera (eye and look-at point together) by that offset in meters.

    ``randomizable`` prepares the cube and table for per-environment recoloring at runtime (see
    :class:`VisualRandomizer`): every environment except one in five gets its own cube material, and
    likewise for the table (a different one in five), all bound before the simulation starts.

    ``crop_render_size`` > 0 adds a second camera at the same pose that renders at that resolution,
    for oracle crops around the cube (see :func:`oracle_crops`). The normal camera is unchanged.
    """
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    # render again after a reset, so the first image of an episode shows the new episode
    env_cfg.rerender_on_reset = True
    # the goal and gripper-frame markers would give the goal away in the image
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.scene.ee_frame.debug_vis = False
    env_cfg.scene.light.spawn.intensity *= light_scale
    if randomizable:
        env_cfg.scene.object.spawn.func = _spawn_uninstanced(env_cfg.scene.object.spawn.func)
        env_cfg.scene.table.spawn.func = _spawn_uninstanced(env_cfg.scene.table.spawn.func)
    if cube_color is not None:
        env_cfg.scene.object.spawn.func = _spawn_with_flat_color(env_cfg.scene.object.spawn.func, "cube", cube_color)
    if table_color is not None:
        env_cfg.scene.table.spawn.func = _spawn_with_flat_color(env_cfg.scene.table.spawn.func, "table", table_color)
    env_cfg.scene.camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=image_size,
        height=image_size,
    )
    if crop_render_size > 0:
        env_cfg.scene.crop_camera = env_cfg.scene.camera.replace(
            prim_path="{ENV_REGEX_NS}/CropCamera", width=crop_render_size, height=crop_render_size
        )
    if viewer_eye is not None:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = viewer_env_index
        env_cfg.viewer.eye = tuple(viewer_eye)
        env_cfg.viewer.lookat = tuple(viewer_lookat)
    if randomizable:
        # bind the per-environment materials once the scene is cloned, before the simulation starts
        from isaaclab.scene import InteractiveScene

        original_scene_init = InteractiveScene.__init__

        def scene_init(scene, cfg):
            original_scene_init(scene, cfg)
            _bind_random_materials(scene.num_envs)

        InteractiveScene.__init__ = scene_init
    try:
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if viewer_eye is not None else None))
    finally:
        if randomizable:
            InteractiveScene.__init__ = original_scene_init
    base_env = env.unwrapped

    terms = base_env.observation_manager.active_terms["policy"]
    dims = [int(torch.tensor(d).prod()) for d in base_env.observation_manager.group_obs_term_dim["policy"]]
    if terms != TEACHER_OBS_TERMS or dims != TEACHER_OBS_DIMS:
        raise RuntimeError(f"Unexpected observation layout: {list(zip(terms, dims))}")

    origins = base_env.scene.env_origins
    shift = torch.tensor(camera_shift, device=origins.device)
    eyes = origins + torch.tensor(eye, device=origins.device) + shift
    targets = origins + torch.tensor(lookat, device=origins.device) + shift
    base_env.scene["camera"].set_world_poses_from_view(eyes, targets)
    if crop_render_size > 0:
        base_env.scene["crop_camera"].set_world_poses_from_view(eyes, targets)
    base_env.reset(seed=seed)
    return env


def _uninstance(root):
    """Makes every instanceable prim under ``root`` editable, so a material bound on top applies."""
    from pxr import Usd

    while True:
        instanced = [p for p in Usd.PrimRange(root) if p.IsInstanceable()]
        if not instanced:
            return
        for p in instanced:
            p.SetInstanceable(False)


def _spawn_with_flat_color(spawn_func, name: str, color: tuple):
    """Wraps a USD spawner so the spawned asset is drawn in one flat color.

    The cube and table are instanceable USD files whose own materials win over a material bound at
    spawn time, so the asset is made non-instanceable first and the color is bound over all of its
    descendants. This happens before the simulation starts, and changes nothing about the physics.
    """
    import omni.usd

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = spawn_func(prim_path, cfg, *args, **kwargs)
        stage = omni.usd.get_context().get_stage()
        material_path = f"/World/Looks/flat_{name}"
        if not stage.GetPrimAtPath(material_path).IsValid():
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color))
            material_cfg.func(material_path, material_cfg)
        for root in sim_utils.find_matching_prims(prim_path):
            _uninstance(root)
            sim_utils.bind_visual_material(root.GetPath().pathString, material_path, stronger_than_descendants=True)
        return prim

    return spawn


def _spawn_uninstanced(spawn_func):
    """Wraps a USD spawner so the spawned asset is made non-instanceable (its look is unchanged)."""

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = spawn_func(prim_path, cfg, *args, **kwargs)
        for root in sim_utils.find_matching_prims(prim_path):
            _uninstance(root)
        return prim

    return spawn


def _bind_random_materials(num_envs: int):
    """Gives each environment's cube and table its own recolorable material.

    Runs after the scene has been cloned (at spawn time only environment 0 exists) and before the
    simulation starts: binding a material changes the rigid body's prim, which PhysX cannot take
    once it runs. Environments whose index modulo 5 equals KEEP_ORIGINAL_CUBE / KEEP_ORIGINAL_TABLE
    get no material for that asset and keep its original look.
    """
    import omni.usd
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    for asset, name, keep_original in (("Object", "object", KEEP_ORIGINAL_CUBE), ("Table", "table", KEEP_ORIGINAL_TABLE)):
        for i in range(num_envs):
            if i % 5 == keep_original:
                # the cloned environments inherit from environment 0, including its material binding,
                # so an explicitly empty binding is needed to keep the original look
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/World/envs/env_{i}/{asset}")).UnbindDirectBinding()
                continue
            material_path = f"/World/Looks/random_{name}_{i}"
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
            material_cfg.func(material_path, material_cfg)
            sim_utils.bind_visual_material(f"/World/envs/env_{i}/{asset}", material_path, stronger_than_descendants=True)


class VisualRandomizer:
    """Changes what the camera sees between episodes, per environment, without touching the physics.

    Needs an environment made with ``make_vision_env(..., randomizable=True)``. At runtime it only
    changes values: the shared scene light's intensity (one value per call), each environment's cube
    and table color, and each environment's camera offset. Environments that keep the original cube
    or table look (see KEEP_ORIGINAL_CUBE/TABLE) ignore the colors given for them.
    """

    def __init__(self, base_env, eye: tuple = CAMERA_EYE, lookat: tuple = CAMERA_LOOKAT):
        import omni.usd
        from pxr import UsdShade

        self.base_env = base_env
        self.stage = omni.usd.get_context().get_stage()
        self.num_envs = base_env.num_envs
        self.light = self.stage.GetPrimAtPath("/World/light").GetAttribute("inputs:intensity")
        if not self.light.IsValid():
            raise RuntimeError("Could not find the intensity of /World/light.")
        self.base_intensity = self.light.Get()
        origins = base_env.scene.env_origins
        self.eyes = origins + torch.tensor(eye, device=origins.device)
        self.targets = origins + torch.tensor(lookat, device=origins.device)

        self.inputs = {}
        for name in ("object", "table"):
            for i in range(self.num_envs):
                material = self.stage.GetPrimAtPath(f"/World/Looks/random_{name}_{i}")
                if not material.IsValid():
                    continue
                shaders = [UsdShade.Shader(p) for p in material.GetChildren() if p.IsA(UsdShade.Shader)]
                self.inputs[name, i] = shaders[0].GetInput("diffuseColor")
        if not self.inputs:
            raise RuntimeError("No recolorable materials found; was the env made with randomizable=True?")

    def recolorable(self, name: str, i: int) -> bool:
        return (name, i) in self.inputs

    def apply(self, light_scale: float, cube_colors: list, table_colors: list, camera_shifts: torch.Tensor):
        """light_scale: one float; cube_colors/table_colors: per-env RGB; camera_shifts: (num_envs, 3) [m]."""
        from pxr import Gf

        self.light.Set(float(self.base_intensity * light_scale))
        for name, colors in (("object", cube_colors), ("table", table_colors)):
            for i in range(self.num_envs):
                if self.recolorable(name, i):
                    self.inputs[name, i].Set(Gf.Vec3f(*[float(c) for c in colors[i]]))
        shifts = camera_shifts.to(self.eyes.device)
        self.base_env.scene["camera"].set_world_poses_from_view(self.eyes + shifts, self.targets + shifts)


def get_images(base_env, camera: str = "camera") -> torch.Tensor:
    """Current camera images as (num_envs, H, W, 3) uint8."""
    rgb = base_env.scene[camera].data.output["rgb"][..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb


def project_to_camera(points: torch.Tensor, size: int, eye: tuple = CAMERA_EYE, lookat: tuple = CAMERA_LOOKAT,
                      focal_length: float = CAMERA_FOCAL_LENGTH, aperture: float = 20.955) -> torch.Tensor:
    """Pixel coordinates (u right, v down) of points given in each environment's frame, (N, 3) -> (N, 2).

    Pinhole model of the student camera (unshifted pose) in a size x size image.
    """
    eye_t = torch.tensor(eye, device=points.device, dtype=points.dtype)
    forward = torch.tensor(lookat, device=points.device, dtype=points.dtype) - eye_t
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    offset = points - eye_t
    depth = offset @ forward
    focal_px = size * focal_length / aperture
    return torch.stack([size / 2 + focal_px * (offset @ right) / depth, size / 2 + focal_px * (offset @ down) / depth], dim=1)


def oracle_crops(base_env, cube_pos: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crops of the high-resolution camera centered on the cube's TRUE position (privileged).

    This stands in for a perfect cube detector. cube_pos is (N, 3) in the robot base frame (the
    environment origin). Windows that would leave the image are shifted back inside it. Returns the
    crops (N, crop_size, crop_size, 3) uint8 and the window centers in pixels (N, 2).
    """
    rgb = get_images(base_env, "crop_camera")
    num, size = rgb.shape[0], rgb.shape[1]
    center = project_to_camera(cube_pos.float(), size)
    top_left = (center.round().long() - crop_size // 2).clamp(0, size - crop_size)
    steps = torch.arange(crop_size, device=rgb.device)
    rows = (top_left[:, 1, None] + steps)[:, :, None]
    cols = (top_left[:, 0, None] + steps)[:, None, :]
    crops = rgb[torch.arange(num, device=rgb.device)[:, None, None], rows, cols]
    return crops, top_left.float() + crop_size / 2


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
