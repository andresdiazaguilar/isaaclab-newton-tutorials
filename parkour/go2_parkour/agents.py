"""PPO settings for the parkour task: the Go2 rough-terrain settings, with a longer training and its own log folder."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg


@configclass
class Go2ParkourPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_parkour"  # runs are saved to logs/rsl_rl/go2_parkour/
        self.max_iterations = 3000
