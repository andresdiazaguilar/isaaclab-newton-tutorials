"""Parkour terrains: a custom "gap + step up" sub-terrain and the terrain generator config of the task."""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def gap_step_terrain(difficulty: float, cfg: GapStepTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A platform surrounded by a gap; the ground beyond the gap is raised by a step.

    The robot spawns on the platform and must jump across the gap *and* up the step.
    Both the gap width and the step height grow linearly with ``difficulty`` (0 = easiest, 1 = hardest).
    Like Isaac Lab's own gap terrain, the gap has no floor: a robot that falls in keeps falling.

    Returns the meshes (the terrain spans x in [0, size_x], y in [0, size_y]) and the spawn origin.
    """
    gap = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    step = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    thickness = 1.0  # how far the solid blocks extend below their top surface
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]

    # central platform, top surface at z = 0
    platform = trimesh.creation.box(
        (cfg.platform_width, cfg.platform_width, thickness),
        trimesh.transformations.translation_matrix((cx, cy, -thickness / 2)),
    )
    # outer ring beyond the gap, top surface at z = step
    inner = cfg.platform_width + 2 * gap
    ring = make_border(cfg.size, (inner, inner), thickness + step, (cx, cy, step - (thickness + step) / 2))
    return [platform, *ring], np.array([cx, cy, 0.0])


@configclass
class GapStepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`gap_step_terrain`."""

    function = gap_step_terrain
    gap_width_range: tuple[float, float] = (0.05, 0.35)
    """Gap width [m] at difficulty 0 and 1."""
    step_height_range: tuple[float, float] = (0.0, 0.15)
    """Height [m] of the far side above the platform at difficulty 0 and 1."""
    platform_width: float = 3.0
    """Side length [m] of the square spawn platform."""


PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # each tile is 8 m x 8 m
    border_width=20.0,  # flat border around the whole grid
    num_rows=10,  # rows = difficulty levels (with the curriculum)
    num_cols=20,  # columns = obstacle types, split by `proportion`
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # warm-up: mildly uneven ground
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        # spawn at the bottom of a pit, climb the wall to get out
        "climb": terrain_gen.MeshPitTerrainCfg(proportion=0.25, pit_depth_range=(0.05, 0.30), platform_width=3.0),
        # spawn on top of a box, step or jump down
        "drop": terrain_gen.MeshBoxTerrainCfg(proportion=0.15, box_height_range=(0.05, 0.40), platform_width=3.0),
        # spawn on a platform surrounded by a gap, jump across
        "gap": terrain_gen.MeshGapTerrainCfg(proportion=0.25, gap_width_range=(0.05, 0.45), platform_width=3.0),
        # our own: jump across a gap onto higher ground
        "gap_step": GapStepTerrainCfg(proportion=0.25),
    },
)
"""The parkour terrain grid: 10 difficulty levels x 20 columns of obstacle types."""
