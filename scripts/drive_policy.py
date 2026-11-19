"""Run any trained RSL-RL policy live in a viewer window, and drive it with the keyboard if the task has a velocity command.

Usage (from the repository folder):
    ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint checkpoints/go2_flat_model_999.pt
    ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint logs/go2_flat   # notebook 1's runs
    ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Rough-UnitreeGo2 --num_envs 4
    ./run.sh python scripts/drive_policy.py --task Isaac-Cartpole-Direct
Extra Hydra-style overrides go at the end, like with `isaaclab play`, e.g. `env.episode_length_s=10`.

--checkpoint accepts a model_*.pt file, a run folder, or a folder of runs (the newest run and its latest model are used).
Without it, the newest run in logs/rsl_rl/<experiment_name>/ (where `./run.sh isaaclab train` saves) is used.
The checkpoint must come from the same task: the network's input/output sizes depend on the task's observations.

Keys (click into the viewer window first):
    W / S   forward / backward          A / D   sideways left / right
    Q / E   turn left / right           Space   stop
    R       reset the environments      C       toggle the follow camera
The W/A/S/D/Q/E keys only do something for tasks with a velocity command (e.g. the Velocity-* locomotion tasks);
the maximum speeds are taken from the task's own command ranges, i.e. what the policy was trained on.
"""

import argparse
import contextlib
import glob
import os
import re
import time

from pathlib import Path

# WSL: draw the viewer on the NVIDIA GPU (via Direct3D 12) instead of the CPU software renderer "llvmpipe",
# which only reaches ~0.16x real time. Must be set before any OpenGL library is loaded. (Native Linux doesn't need it.)
if "microsoft" in Path("/proc/version").read_text().lower():
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

parser = argparse.ArgumentParser(description="Run a trained RSL-RL policy live, with keyboard velocity commands.")
parser.add_argument("--task", required=True, help="registered task name, see `uv run isaaclab list_envs`")
parser.add_argument("--checkpoint", default=None, help="model_*.pt, a run folder, or a folder of runs")
parser.add_argument("--num_envs", type=int, default=1, help="number of robots (all get the same command)")
parser.add_argument("--speed", type=float, default=1.0, help="scale the max command (>1 = beyond the training range)")
parser.add_argument("--max_steps", type=int, default=-1, help="stop after this many policy steps (-1 = until closed)")
parser.add_argument("--hold", nargs="*", default=[], help="keys held from the start, e.g. --hold KeyW (for testing)")
args, hydra_overrides = parser.parse_known_args()

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

import isaaclab_tasks  # noqa: F401  (registers Isaac Lab's own tasks with gym)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import resolve_task_config

# Also register tasks from other packages (e.g. Go2-Parkour from ./parkour), the same way the `isaaclab`
# command does: import every package declared under the "isaaclab.tasks" entry-point group.
import importlib.metadata

for entry_point in importlib.metadata.entry_points(group="isaaclab.tasks"):
    entry_point.load()

REPO = Path(__file__).resolve().parent.parent
ISAACLAB_DIR = os.environ.get("ISAACLAB_DIR", "")  # set by run.sh


def latest_model(folder):
    """Return the latest model_*.pt in a folder, or in its newest sub-folder (run) that has one."""
    def number(path):
        return int(re.search(r"model_(\d+)\.pt$", path).group(1))

    models = glob.glob(os.path.join(folder, "model_*.pt"))
    if models:
        return max(models, key=number)
    for run in sorted(glob.glob(os.path.join(folder, "*/")), reverse=True):  # run folders are named by date
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=number)
    raise FileNotFoundError(f"no model_*.pt found in {folder} or its sub-folders; pass --checkpoint")


def resolve_checkpoint(agent_cfg):
    if args.checkpoint:
        path = os.path.abspath(os.path.expanduser(args.checkpoint))
        return latest_model(path) if os.path.isdir(path) else path
    # `isaaclab train` saves to ./logs/rsl_rl/<experiment>/ relative to where it was run
    candidates = [Path.cwd(), REPO] + ([Path(ISAACLAB_DIR)] if ISAACLAB_DIR else [])
    for root in dict.fromkeys(candidates):
        folder = root / "logs" / "rsl_rl" / agent_cfg.experiment_name
        if folder.is_dir():
            with contextlib.suppress(FileNotFoundError):
                return latest_model(str(folder))
    raise FileNotFoundError(
        f"no run of experiment '{agent_cfg.experiment_name}' found in logs/rsl_rl/; pass --checkpoint"
    )


def setup_velocity_command(env_cfg):
    """If the task has a velocity command, stop random resampling and return (term name, max |vx|, |vy|, |yaw|)."""
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return None
    for name, term in vars(commands).items():
        ranges = getattr(term, "ranges", None)
        if ranges is not None and hasattr(ranges, "lin_vel_x") and hasattr(ranges, "ang_vel_z"):
            limits = [max(abs(r[0]), abs(r[1])) for r in (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)]
            term.resampling_time_range = (1.0e6, 1.0e6)  # never sample a random command: the keys decide
            if hasattr(term, "heading_command"):
                term.heading_command = False  # we command the yaw rate directly
            if hasattr(term, "rel_standing_envs"):
                term.rel_standing_envs = 0.0
            return name, [args.speed * x for x in limits]
    return None


def main():
    # ---- configs: exactly what `isaaclab play` would use (play_mode = evaluation settings, e.g. no pushes) ----
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=hydra_overrides)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1.0e6  # no time-out: run as long as you like (failures still reset)
    velocity = setup_velocity_command(env_cfg)
    checkpoint = resolve_checkpoint(agent_cfg)
    print(f"task      : {args.task}\ncheckpoint: {checkpoint}")
    if velocity:
        print(f"keyboard  : command '{velocity[0]}', max vx={velocity[1][0]:.2f} vy={velocity[1][1]:.2f} yaw={velocity[1][2]:.2f}")
    else:
        print("keyboard  : this task has no velocity command, so only R (reset) does something")

    with launch_simulation(env_cfg, {"visualizer": ["newton_gl"], "visualizer_explicit": True}):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        base = env.unwrapped
        subscriptions = []
        try:
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=base.device)

            cmd_term = robot = None
            if velocity:
                cmd_term = base.command_manager.get_term(velocity[0])
                robot = getattr(cmd_term, "robot", None)
                key_axes = {
                    "KeyW": (0, +1), "ArrowUp": (0, +1), "KeyS": (0, -1), "ArrowDown": (0, -1),
                    "KeyA": (1, +1), "KeyD": (1, -1),
                    "KeyQ": (2, +1), "ArrowLeft": (2, +1), "KeyE": (2, -1), "ArrowRight": (2, -1),
                }  # fmt: skip
                max_cmd = torch.tensor(velocity[1], device=base.device)

            # ---- keyboard: remember which keys are held ----
            held, state = set(), {"follow": robot is not None, "reset": False}

            def on_key(code, pressed):
                if pressed and code == "KeyC" and robot is not None:
                    state["follow"] = not state["follow"]
                elif pressed and code == "KeyR":
                    state["reset"] = True
                (held.add if pressed else held.discard)(code)

            for viz in base.sim.visualizers:
                if viz.key_event_source is not None:
                    if velocity:
                        viz.set_camera_keys_enabled(False)  # WASD/QE/arrows drive the robot, not the camera
                    subscriptions.append(viz.key_event_source.add_key_listener(on_key, held.clear))
            if not subscriptions:
                print("WARNING: the viewer reports no keys")

            cmd = torch.zeros(3, device=base.device)
            obs = env.get_observations()
            step, next_time = 0, time.perf_counter()
            rt_start, rt_steps, rt_factor = time.perf_counter(), 0, 1.0
            while base.sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
                status = ""
                if velocity:
                    # target command from the held keys, smoothed so the robot isn't jerked around
                    keys = held | set(args.hold)  # --hold keys stay pressed even without window focus
                    target = torch.zeros(3, device=base.device)
                    if "Space" not in keys:
                        for code in keys:
                            if code in key_axes:
                                axis, sign = key_axes[code]
                                target[axis] += sign * max_cmd[axis]
                    cmd += 0.15 * (target - cmd)
                    cmd_term.vel_command_b[:] = cmd  # <- the command every robot's policy observes
                    status = f"command vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}"
                    if robot is not None:
                        vel = robot.data.root_lin_vel_b.torch[0]
                        status += f"   actual vx={vel[0]:+.2f} vy={vel[1]:+.2f}"

                if state["reset"]:
                    state["reset"] = False
                    obs, _ = env.reset()

                with torch.inference_mode():
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)  # clears the memory of recurrent policies; no-op for MLPs

                if state["follow"]:
                    pos = robot.data.root_pos_w.torch[0]
                    for viz in base.sim.visualizers:
                        viz.set_camera_view(
                            eye=(float(pos[0]) - 2.0, float(pos[1]) - 2.0, float(pos[2]) + 0.9),
                            target=(float(pos[0]), float(pos[1]), float(pos[2])),
                        )

                rt_steps += 1
                if rt_steps == 50:  # simulated time / wall-clock time (1.00 = real time)
                    now = time.perf_counter()
                    rt_factor = rt_steps * base.step_dt / (now - rt_start)
                    if args.max_steps > 0:
                        print(f"\n[timing] real-time factor {rt_factor:.2f}")
                    rt_start, rt_steps = now, 0
                print(f"\r{status}   speed {rt_factor:.2f}x real time   ", end="", flush=True)

                # run in real time: one policy step = step_dt seconds
                next_time += base.step_dt
                time.sleep(max(0.0, next_time - time.perf_counter()))
                if time.perf_counter() - next_time > 0.5:  # fell far behind (e.g. window dragged): resync
                    next_time = time.perf_counter()
                step += 1
            print()
        finally:
            for s in subscriptions:
                s.close()
            with contextlib.suppress(Exception):
                env.close()


if __name__ == "__main__":
    main()
