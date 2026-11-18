# Your first RL policy in Isaac Lab

<p align="center"><img src="media/go2_walk.gif" width="400" alt="A Unitree Go2 walking, trained in this tutorial"></p>

A hands-on, single-notebook introduction to robot learning with **[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0**
on the **Newton** physics engine. No Isaac Sim needed, runs on a laptop GPU (tested on an RTX 5070 Laptop, 8 GB).
Made to help those who have some basic knowledge of RL to start writing, training, debugging and explaining their own RL tasks.

## What you'll learn

| Part | Topic |
|---|---|
| 0 | Setup |
| 1,2 | RL refresher: MDPs, value functions, PPO recap |
| 3 | Isaac Lab basics: scenes, assets, joints, the simulation loop, batched GPU tensors |
| 4 | Your first task (cartpole, Direct workflow): observations, actions, rewards, terminations, resets; PPO training; reading training curves; a reward-hacking experiment |
| 5 | Manager-based workflow: train a **Unitree Go2** to walk; reward terms; a **local optimum** and how to diagnose it |
| 6 | The terminal workflow (`train`, `play`, TensorBoard) |
| 7 | Recap and test questions |

Everything runs in the notebook: videos are rendered with a simulated camera and shown under the cells.
Total compute is about 20 minutes (most of it the Go2 training); the rest is reading.

## Requirements

- **Linux** (tested: Ubuntu 24.04), or **Windows 11 with WSL2** (tested: Ubuntu 24.04 on WSL2) → see [docs/windows_wsl_setup.md](docs/windows_wsl_setup.md)
- an **NVIDIA GPU** with a recent driver (8 GB VRAM is enough)
- ~10 GB of disk space

## Quickstart

```bash
git clone https://github.com/andresdiazaguilar/isaaclab-newton-tutorials
cd isaaclab-newton-tutorials
./install.sh          # uv + Isaac Lab + Python environment
./start_jupyter.sh    # then open the printed http://localhost:8888/... link and 01_first_policy.ipynb
```

`install.sh` clones Isaac Lab into `./IsaacLab` (or reuses `../IsaacLab`, or `$ISAACLAB_DIR`) at the commit this
tutorial was tested with, because Isaac Lab's main branch changes quickly.

## Drive a trained robot with your keyboard

A simple Go2 policy trained with this notebook (1000 PPO iterations, 1 MB) is included, so you can try this immediately:

```bash
./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint checkpoints/go2_flat_model_999.pt
```

After notebook 1, drive the Go2 with your own policy  instead with `--checkpoint logs/go2_flat`.
Click into the viewer window, then: **W/S** forward/backward, **A/D** sideways, **Q/E** turn left/right, **Space** stop,
**R** reset, **C** toggle the follow camera.

`drive_policy.py` works for any registered task and RSL-RL checkpoint (`--task`, `--num_envs`, `--speed`, ...;
see the top of the file). 

## Repository layout

```
01_first_policy.ipynb   the notebook
install.sh              installs everything (safe to re-run)
start_jupyter.sh        JupyterLab with Isaac Lab's Python environment
run.sh                  run any command with Isaac Lab's environment, e.g. ./run.sh isaaclab list_envs
scripts/                drive_policy.py, drive_go2.py, env.sh (shared settings)
checkpoints/            a trained Go2 policy
docs/                   Windows/WSL setup and troubleshooting
logs/, videos/          created when you run things (not tracked by git)
```

## Troubleshooting

- **The viewer window runs in slow motion (WSL).** OpenGL falls back to a CPU renderer (`llvmpipe`) by default.
  `run.sh` and the scripts set `GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` to draw on the GPU;
  set them yourself for other OpenGL programs. Details in [docs/windows_wsl_setup.md](docs/windows_wsl_setup.md).
- **The first simulation cell takes long.** NVIDIA Warp compiles its GPU kernels the first time; later runs use a cache.
- **The Go2 stands still after training.** That's a local optimum, discussed in Part 5: train longer (the notebook uses 1000 iterations).
- **`size mismatch` when loading a checkpoint.** The checkpoint comes from a different task: the network's input size
  depends on the task's observations (e.g. the rough-terrain task adds 187 height-scan values).

## Tested with

Isaac Lab `3e7e613` (main, 2026-11-16, v3.0.0), Newton/MJWarp physics, RSL-RL 5.5.1, PyTorch with CUDA,
Ubuntu 24.04 on WSL2, NVIDIA RTX 5070 Laptop GPU (driver 621.24).

## About this repository

I made this repository to guide myself through learning Isaac Lab while being able to run it locally (given a beefy enough laptop), and I'm sharing it in case it helps others starting out the same way. The material draws on:

- Isaac Lab's official Newton notebooks (`notebooks/explore.ipynb` and `notebooks/training.ipynb` in the
  [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab))
- Isaac Lab's own task code, in particular the cartpole Direct task and the Go2 velocity-tracking task, which
  the notebook's cartpole environment and Go2 sections are based on,
- the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl),
- the PPO and GAE papers ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347);
  [Schulman et al., 2015](https://arxiv.org/abs/1506.02438)) for the theory parts, and
  [Rudin et al., 2021](https://arxiv.org/abs/2109.11978) (*Learning to Walk in Minutes*), the origin of RSL-RL
  and of Isaac Lab's locomotion tasks.

The explanations, experiments (reward hacking, the standing-still local optimum), helper scripts and installation
were written for this tutorial and tested on the setup listed above. It was built with the assistance of Claude Code.

## License

[BSD-3-Clause](LICENSE). This repository only uses Isaac Lab and RSL-RL; it doesn't contain their code.
