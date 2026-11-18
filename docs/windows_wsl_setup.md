# Windows setup: WSL2 + Ubuntu 24.04

Isaac Lab without Isaac Sim is easiest on Linux. On Windows, WSL2 gives you a real Ubuntu that uses your
NVIDIA GPU, with GUI windows (WSLg) and Jupyter in your Windows browser. These are the steps and the
problems we actually ran into.

## 1. Before you start

- **Windows 11** and an **NVIDIA GPU**.
- Install or update the normal **NVIDIA Windows driver**. Do **not** install an NVIDIA driver inside Ubuntu:
  WSL passes the Windows driver through.
- Hardware virtualization must be enabled in the BIOS/UEFI (usually on by default on recent laptops).

Why not native Windows? Isaac Lab's Python tooling (uv) downloads unsigned binaries. With Windows
**Smart App Control** on, they are blocked (error `0xc0e90002`), and turning Smart App Control off is a
security trade-off you may not want to make. WSL avoids the issue entirely.

## 2. Install WSL and Ubuntu 24.04

In **PowerShell as administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Then **restart** Windows (use *Restart*, not *Shut down*: with Fast Startup, *Shut down* doesn't finish
installing Windows features).

After the restart, open **Ubuntu 24.04** from the Start menu and create your Linux user name and password.

Use this command rather than the Microsoft Store's generic "Ubuntu" app: that one installs the newest Ubuntu
release, which this tutorial wasn't tested with.

### If it fails

- `WslRegisterDistribution failed with error: 0x80370114`, or `wsl --install` keeps saying the distribution
  isn't registered: the **Virtual Machine Platform** feature hasn't finished installing. Windows can defer
  this across a restart, so **restart once more**. If it still fails, enable it explicitly (PowerShell as admin),
  then restart:
  ```powershell
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
  dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
  ```
- `wsl --status` and `wsl -l -v` show what is installed; the distribution should be listed with VERSION 2.

## 3. Inside Ubuntu

```bash
sudo apt update && sudo apt install -y git curl
nvidia-smi            # should list your GPU; if not, update the Windows NVIDIA driver
```

Clone this repository **inside the Linux file system** (e.g. in `~`), not under `/mnt/c/...`: accessing
Windows files from Linux is much slower. Then follow the Quickstart in the [README](../README.md).

## 4. Working with WSL

- **Jupyter:** `./start_jupyter.sh` prints a `http://localhost:8888/...` link; open it in your Windows browser.
  (VS Code with the *WSL* extension also works.)
- **Files:** in Windows Explorer, your Ubuntu files are under `\\wsl$\Ubuntu-24.04\home\<your user>`.
- **Viewer windows** (e.g. `drive_policy.py`, `isaaclab play --viz newton_gl`) open as normal Windows windows.
  By default WSL draws OpenGL on the CPU (`llvmpipe`), which runs the simulation at ~0.16× real time.
  `run.sh` and the scripts set these variables so it uses the NVIDIA GPU instead:
  ```bash
  export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
  ```
  To use them for everything, add that line to `~/.bashrc`. (`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` matters on
  laptops that also have integrated graphics; without it, the integrated GPU may be picked.)
