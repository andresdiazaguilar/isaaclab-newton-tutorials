"""Drive your trained Go2 with the keyboard, in real time.

A viewer window opens. Click into it, then hold:
    W / S       forward / backward          (vx, up to 1.0 m/s)
    A / D       sideways left / right       (vy, up to 0.6 m/s)
    Q / E       turn left / right           (yaw rate, up to 1.0 rad/s)
    Space       stop (zero command)
    R           reset the robot
    C           toggle the follow camera (off = orbit with the mouse)
Close the window (or Ctrl+C in the terminal) to quit.

How it works: the policy was trained to follow a velocity *command* that is part of its observation.
During training the command was random; here we overwrite it every step with what the keys say.

This is the minimal, Go2-only version, written to be read. scripts/drive_policy.py does the same for any task.

Run (from the repository folder):
    ./run.sh python scripts/drive_go2.py          # newest run in logs/go2_flat (notebook 1), else the bundled checkpoint
    ./run.sh python scripts/drive_go2.py --checkpoint path/to/model_999.pt
"""

import argparse
import glob
import os
import time
from pathlib import Path

# WSL: by default the viewer window is drawn by "llvmpipe", a CPU software renderer (-> ~0.16x real time).
# These make Mesa translate OpenGL to Direct3D 12 on the NVIDIA GPU instead (-> real time).
# They must be set before any OpenGL library is loaded. (Native Linux doesn't need them.)
if "microsoft" in Path("/proc/version").read_text().lower():
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Drive a trained Go2 with the keyboard.")
parser.add_argument("--checkpoint", default=None, help="model_*.pt to load (default: newest Go2 flat run).")
parser.add_argument("--max_steps", type=int, default=-1, help="stop after this many policy steps (-1 = run until closed)")
parser.add_argument("--hold", nargs="*", default=[], help="keys held from the start, e.g. --hold KeyW (for testing)")
args = parser.parse_args()

import torch

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import to_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from rsl_rl.runners import OnPolicyRunner


def newest_checkpoint():
    runs = sorted(glob.glob(str(REPO / "logs" / "go2_flat" / "*")))  # notebook 1 saves its Go2 runs here
    for run in reversed(runs):  # newest run that has checkpoints
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(p.rsplit("_", 1)[-1][:-3]))
    bundled = REPO / "checkpoints" / "go2_flat_model_999.pt"  # trained for 1000 iterations, ships with the repo
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError("no Go2 checkpoint found; pass --checkpoint")


# key -> (command index, value). Index 0 = vx [m/s], 1 = vy [m/s], 2 = yaw rate [rad/s]
KEYMAP = {
    "KeyW": (0, +1.0), "ArrowUp": (0, +1.0),
    "KeyS": (0, -1.0), "ArrowDown": (0, -1.0),
    "KeyA": (1, +0.6),  # a gentler sideways speed; the policy was trained on up to 1.0
    "KeyD": (1, -0.6),
    "KeyQ": (2, +1.0), "ArrowLeft": (2, +1.0),
    "KeyE": (2, -1.0), "ArrowRight": (2, -1.0),
}  # fmt: skip


def main():
    checkpoint = args.checkpoint or newest_checkpoint()
    print(f"checkpoint: {checkpoint}")

    # ---- environment: the training task, with the random commands switched off ----
    cfg = resolve_presets(UnitreeGo2FlatEnvCfg(), selected=("newton_mjwarp",))
    cfg.play_mode()  # evaluation settings: no random pushes
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e6  # no time-out: drive as long as you like
    command = cfg.commands.base_velocity
    command.heading_command = False  # we give the yaw rate directly
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1.0e6, 1.0e6)  # never sample a random command
    agent_cfg = UnitreeGo2FlatPPORunnerCfg()

    with launch_simulation(cfg, {"visualizer": ["newton_gl"], "visualizer_explicit": True}):
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent_cfg.clip_actions)
        base = env.unwrapped
        try:
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=base.device)
            cmd_term = base.command_manager.get_term("base_velocity")
            robot = base.scene["robot"]

            # ---- keyboard: remember which keys are held ----
            held, state = set(), {"follow": True, "reset": False}

            def on_key(code, pressed):
                if pressed and code == "KeyC":
                    state["follow"] = not state["follow"]
                elif pressed and code == "KeyR":
                    state["reset"] = True
                (held.add if pressed else held.discard)(code)

            subscriptions = []
            for viz in base.sim.visualizers:
                if viz.key_event_source is not None:
                    viz.set_camera_keys_enabled(False)  # WASD/QE/arrows drive the robot, not the camera
                    subscriptions.append(viz.key_event_source.add_key_listener(on_key, held.clear))
            if not subscriptions:
                print("WARNING: the viewer reports no keys; the robot will only stand.")

            print(__doc__.split("How it works")[0])
            cmd = torch.zeros(3, device=base.device)
            obs = env.get_observations()
            step, next_time = 0, time.perf_counter()
            rt_start, rt_steps, rt_factor = time.perf_counter(), 0, 1.0  # real-time factor measurement
            while base.sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
                # target command from the held keys, smoothed so the robot isn't jerked around
                target = torch.zeros(3, device=base.device)
                keys = held | set(args.hold)  # --hold keys stay pressed even without window focus
                if "Space" not in keys:
                    for code in keys:
                        if code in KEYMAP:
                            i, value = KEYMAP[code]
                            target[i] += value
                cmd += 0.15 * (target - cmd)
                cmd_term.vel_command_b[:] = cmd  # <- the command the policy observes

                if state["reset"]:
                    state["reset"] = False
                    obs, _ = env.reset()

                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                rt_steps += 1
                if rt_steps == 50:  # every 50 steps: simulated time / wall-clock time (1.00 = real time)
                    now = time.perf_counter()
                    rt_factor = rt_steps * base.step_dt / (now - rt_start)
                    rt_start, rt_steps = now, 0

                pos = robot.data.root_pos_w.torch[0]
                if state["follow"]:
                    for viz in base.sim.visualizers:
                        viz.set_camera_view(
                            eye=(float(pos[0]) - 2.0, float(pos[1]) - 2.0, 1.2), target=(float(pos[0]), float(pos[1]), 0.3)
                        )
                vel = robot.data.root_lin_vel_b.torch[0]
                print(
                    f"\rcommand vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}   "
                    f"actual vx={vel[0]:+.2f} vy={vel[1]:+.2f}   speed {rt_factor:.2f}x real time   ",
                    end="",
                    flush=True,
                )

                # run in real time: one policy step = step_dt seconds (0.02 s)
                next_time += base.step_dt
                time.sleep(max(0.0, next_time - time.perf_counter()))
                if time.perf_counter() - next_time > 0.5:  # fell far behind (e.g. window dragged): resync
                    next_time = time.perf_counter()
                step += 1
            print()
            for s in subscriptions:
                s.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
