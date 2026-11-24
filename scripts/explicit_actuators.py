"""Swap the Franka's implicit (PhysX-internal) PD drives for explicit IdealPD actuators.

With ImplicitActuatorCfg the PD law is integrated inside the PhysX solver and Isaac Lab's
``applied_torque`` is only an estimate. With IdealPDActuatorCfg Isaac Lab computes the torque itself,
clips it at ``effort_limit`` and sends it to PhysX as a joint effort, so ``applied_torque`` is exactly
what the simulator applies. Stiffness, damping, effort and velocity limits, armature and friction are
copied unchanged (IdealPD does not use ``velocity_limit``; neither does the implicit drive).
Isaac Lab sources are left untouched: this edits the env cfg before ``gym.make``.
"""

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg


def use_explicit_actuators(env_cfg, groups: list[str] | None = None, asset_name: str = "robot") -> list[str]:
    """Replaces the implicit actuator groups of ``env_cfg.scene.<asset_name>`` in place; returns their names.

    ``groups`` limits the swap to those actuator groups (e.g. ``["panda_shoulder", "panda_forearm"]`` for the arm
    only); by default every implicit group is swapped.
    """
    actuators = getattr(env_cfg.scene, asset_name).actuators
    swapped = []
    for name, cfg in actuators.items():
        if not isinstance(cfg, ImplicitActuatorCfg) or (groups is not None and name not in groups):
            continue
        actuators[name] = IdealPDActuatorCfg(
            joint_names_expr=cfg.joint_names_expr,
            effort_limit=cfg.effort_limit,
            velocity_limit=cfg.velocity_limit,
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            friction=cfg.friction,
        )
        swapped.append(name)
    if groups is not None and set(groups) - set(swapped):
        raise ValueError(f"No implicit actuator groups named {sorted(set(groups) - set(swapped))}.")
    return swapped
