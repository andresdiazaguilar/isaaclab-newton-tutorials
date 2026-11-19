"""Task-specific MDP terms for parkour: one termination and one reward, written as plain functions.

Isaac Lab calls each term with the environment as first argument (plus the `params` from its config) and
expects one value per environment back: a bool tensor (num_envs,) for terminations, a float tensor for rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def height_scan_with_holes(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5, miss_depth: float = 1.0
) -> torch.Tensor:
    """Height scan where a ray that hits nothing (a gap) reads as a ``miss_depth``-metre hole.

    Isaac Lab's ``height_scan`` computes ``sensor_height - hit_height - offset``. A ray that misses the terrain
    reports ``hit_height = +inf``, so the value becomes ``-inf`` and, after clipping, looks like a *wall* 1 m
    higher than the robot. Here a miss is replaced by a hit ``miss_depth`` below the reference level instead.
    Positive values = terrain lower than the robot, negative = higher.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_z = sensor.data.pos_w.torch[:, 2].unsqueeze(1)
    hit_z = sensor.data.ray_hits_w.torch[..., 2]
    hit_z = torch.where(torch.isfinite(hit_z), hit_z, sensor_z - offset - miss_depth)
    return sensor_z - hit_z - offset


def fell_into_gap(
    env: ManagerBasedRLEnv, depth: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the robot's base is more than ``depth`` metres below the height it spawned at.

    The gap terrains have no floor, so a robot that misses a jump would otherwise fall forever (and its episode
    would run until the time-out, teaching nothing). The env origin is the spawn point of the robot's current tile.
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w.torch[:, 2] < env.scene.env_origins[:, 2] - depth


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 2.0) -> torch.Tensor:
    """Penalize feet that hit a vertical surface (a wall or box edge) instead of stepping on top of it.

    A foot "stumbles" when the horizontal part of its contact force is more than ``ratio`` times the vertical part.
    Returns the number of stumbling feet per environment (0..4).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]  # (num_envs, num_feet, 3)
    horizontal = torch.linalg.norm(forces[..., :2], dim=-1)
    vertical = torch.abs(forces[..., 2])
    return torch.sum(horizontal > ratio * vertical, dim=1).float()
