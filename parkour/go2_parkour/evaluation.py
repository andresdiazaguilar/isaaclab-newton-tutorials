"""Evaluate a parkour policy: success rate for every obstacle type x difficulty level.

Robots are placed on every tile of a terrain grid with one column per obstacle type and one row per
difficulty level, all facing +x, and commanded to walk forward. A robot succeeds if it gets
``success_distance`` metres away from its spawn point (past every obstacle) without its episode ending.
"""

from __future__ import annotations

import numpy as np
import torch


def type_columns(gen_cfg) -> dict[str, list[int]]:
    """Which grid columns hold which obstacle type, exactly as the curriculum terrain generator assigns them."""
    names = list(gen_cfg.sub_terrains)
    p = np.array([s.proportion for s in gen_cfg.sub_terrains.values()], dtype=float)
    cum = np.cumsum(p / p.sum())
    cols = {n: [] for n in names}
    for c in range(gen_cfg.num_cols):
        cols[names[int(np.min(np.where(c / gen_cfg.num_cols + 0.001 < cum)[0]))]].append(c)
    return cols


def obstacle_size(sub_cfg, difficulty: float) -> str:
    """Human-readable obstacle size of a sub-terrain at a difficulty, e.g. 'gap 0.27 m'."""
    parts = []
    for field, label in (("pit_depth_range", "wall"), ("box_height_range", "drop"), ("gap_width_range", "gap"),
                         ("step_height_range", "step")):
        rng = getattr(sub_cfg, field, None)
        if rng is not None:
            parts.append(f"{label} {rng[0] + difficulty * (rng[1] - rng[0]):.2f}")
    return " + ".join(parts) if parts else "-"


def make_eval_env_cfg(robots_per_tile: int = 8, seconds: float = 8.0, forward_speed: float = 0.8):
    """The parkour env config, changed for evaluation: one column per type, fixed forward command, no pushes."""
    from isaaclab_tasks.utils.hydra import resolve_presets

    from .env_cfg import Go2ParkourEnvCfg

    cfg = Go2ParkourEnvCfg()
    gen = cfg.scene.terrain.terrain_generator
    gen.num_cols = len(gen.sub_terrains)
    for sub in gen.sub_terrains.values():
        sub.proportion = 1.0  # equal proportions -> exactly one column per type
    cfg.curriculum.terrain_levels = None  # we choose the tiles ourselves...
    gen.curriculum = True  # ...but keep rows sorted by difficulty
    cfg.scene.num_envs = gen.num_rows * gen.num_cols * robots_per_tile
    cfg.episode_length_s = seconds + 5.0
    cmd = cfg.commands.base_velocity
    cmd.heading_command, cmd.rel_standing_envs, cmd.resampling_time_range = False, 0.0, (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_x = (forward_speed, forward_speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.sim.visualizer_cfgs = []
    return resolve_presets(cfg, selected=("newton_mjwarp",))


def evaluate(checkpoint: str, robots_per_tile: int = 8, seconds: float = 8.0, success_distance: float = 3.0,
             forward_speed: float = 0.8) -> dict:
    """Run the policy on every (type, level) tile. Returns success rates (types x levels) and labels.

    Must be called when no other simulation is running; it launches and closes its own.
    """
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from .agents import Go2ParkourPPORunnerCfg

    cfg = make_eval_env_cfg(robots_per_tile, seconds, forward_speed)
    gen = cfg.scene.terrain.terrain_generator
    names, n_levels = list(gen.sub_terrains), gen.num_rows
    agent_cfg = Go2ParkourPPORunnerCfg()
    with launch_simulation(cfg, {"visualizer": None, "visualizer_explicit": True}):
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent_cfg.clip_actions)
        base = env.unwrapped
        try:
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=base.device)
            # put robots_per_tile robots on every tile
            terrain, robot = base.scene.terrain, base.scene["robot"]
            n = base.num_envs
            tile = torch.arange(n, device=base.device) // robots_per_tile
            terrain.terrain_levels[:] = tile % n_levels
            terrain.terrain_types[:] = tile // n_levels
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            obs, _ = env.reset()
            done = torch.zeros(n, dtype=torch.bool, device=base.device)  # result decided
            success = torch.zeros(n, dtype=torch.bool, device=base.device)
            with torch.inference_mode():
                for _ in range(int(seconds / base.step_dt)):
                    obs, _, dones, _ = env.step(policy(obs))
                    # an episode that ended this step is a failure (unless it already succeeded)
                    done |= dones.bool()
                    dist = torch.linalg.norm(robot.data.root_pos_w.torch[:, :2] - base.scene.env_origins[:, :2], dim=1)
                    reached = (dist > success_distance) & ~done
                    success |= reached
                    done |= reached
            rate = success.float().view(len(names), n_levels, robots_per_tile).mean(-1).cpu().numpy()
        finally:
            env.close()
    sizes = [[obstacle_size(gen.sub_terrains[t], (lvl + 0.5) / n_levels) for lvl in range(n_levels)] for t in names]
    return {"success": rate, "types": names, "sizes": sizes, "checkpoint": str(checkpoint)}


def plot_success(result: dict, ax=None):
    """Heat map of success rate: rows = obstacle types, columns = difficulty levels, labelled with obstacle sizes."""
    import matplotlib.pyplot as plt

    rate = result["success"]
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * rate.shape[1] + 2, 0.7 * rate.shape[0] + 1.5), layout="constrained")
    im = ax.imshow(rate, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            ax.text(j, i, f"{rate[i, j]:.0%}\n{result['sizes'][i][j]}", ha="center", va="center", fontsize=6.5)
    ax.set_yticks(range(rate.shape[0]), result["types"])
    ax.set_xticks(range(rate.shape[1]), [f"level {j}" for j in range(rate.shape[1])], fontsize=8)
    ax.set_title("Success rate per obstacle type and difficulty (sizes in m)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    return ax
