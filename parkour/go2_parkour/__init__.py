"""Go2 parkour task for Isaac Lab, built step by step in notebook 02.

Importing this package registers the task with gymnasium. The `isaaclab` command-line tool imports it
automatically through the `isaaclab.tasks` entry point declared in ../pyproject.toml.
"""

import gymnasium as gym

gym.register(
    id="Go2-Parkour",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:Go2ParkourPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
