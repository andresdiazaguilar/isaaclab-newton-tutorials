"""The Go2 parkour environment: Isaac Lab's Go2 rough-terrain task with parkour terrain and two new MDP terms."""

from __future__ import annotations

import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from . import mdp
from .terrains import PARKOUR_TERRAINS_CFG


@configclass
class Go2ParkourEnvCfg(UnitreeGo2RoughEnvCfg):
    """Everything not changed here (robot, observations incl. the height scan, actions, the other rewards,
    domain randomization, the terrain-level curriculum) is inherited from the Go2 rough-terrain task."""

    def __post_init__(self):
        super().__post_init__()  # builds the Go2 rough-terrain task

        # --- terrain: our parkour grid instead of the rough one; the curriculum orders rows by difficulty
        self.scene.terrain.terrain_generator = copy.deepcopy(PARKOUR_TERRAINS_CFG)
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 3  # start on the 4 easiest levels

        # --- commands: mostly forward, so the robot runs *into* the obstacles around its spawn platform
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (0.3, 1.0)
        ranges.lin_vel_y = (-0.3, 0.3)

        # --- observations: gaps must look like holes, not walls (see mdp.height_scan_with_holes)
        self.observations.policy.height_scan.func = mdp.height_scan_with_holes

        # --- terminations: falling into a gap ends the episode
        self.terminations.fell_into_gap = DoneTerm(func=mdp.fell_into_gap, params={"depth": 0.5})

        # --- rewards: penalize feet hitting walls / box edges
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
