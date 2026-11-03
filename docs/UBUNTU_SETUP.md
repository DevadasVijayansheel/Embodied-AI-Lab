# Ubuntu setup and reproduction

## Development baseline

The project documentation records Python 3.11, Isaac Lab 2.3.2, and RSL-RL 3.1.2. The original machine's standalone Isaac Sim VERSION file reads `5.1.0-rc.19+release.26219.9c81211b.gl`. This is the development baseline, not a newly tested installation matrix. The source package metadata permits additional Python versions; use 3.11 to reproduce this work.

Use an Ubuntu release and NVIDIA GPU/driver supported by the selected Isaac Sim release. GPU memory requirements depend on scene complexity, number of environments and cameras. The documented CPU tests are diagnostic tests, not representative training workloads.

## Install the simulator stack

1. Install Git, Miniconda, and the NVIDIA driver appropriate for your simulator release. Confirm GPU access with `nvidia-smi`.
2. Follow the version-specific [Isaac Sim installation documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html). Do not copy a Conda environment directory from another computer.
3. Install Isaac Lab **v2.3.2** outside this repository using its [versioned installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html). Use its documented Conda/Python 3.11 workflow and simulator integration. A standalone simulator and a pip-installed simulator have different runtime setup steps.
4. Install Lab's RSL-RL integration and its dependencies using the version-specific Lab instructions. The recorded RSL-RL version is 3.1.2. Confirm package compatibility before substituting newer versions.

Project setup declares `psutil`; its build configuration requires `setuptools`, `wheel`, and `toml`. Isaac Lab provides the broader simulation, PyTorch, Gymnasium and RL dependencies. `pip install -e` for this project alone does not install that full stack.

## Clone and install the project

```bash
git clone https://github.com/DevadasVijayansheel/Embodied-AI-Lab.git
cd Embodied-AI-Lab/mobile_manipulation
conda activate env_isaaclab
# Standalone Isaac Sim only: use your installation's actual path.
source /absolute/path/to/isaacsim/setup_conda_env.sh
python -m pip install -e source/mobile_manipulation
python -m mobile_manipulation.validate
```

The environment name is conventional. Activate the environment containing Isaac Lab. Use the runtime setup matching your simulator installation method. The lightweight validator checks source/config structure and registrations, not physical task success.

## Supply robot assets separately

USD assets are excluded from Git, including the SO-101 robot. Copy the original robot asset plus all referenced meshes, textures and nested USD files, preserving relative paths. Resolve absolute references to the old machine or export a self-contained asset.

Restore the file to this project-relative path:

```text
source/mobile_manipulation/mobile_manipulation/robots/usd/so101_gt.usd
```

Alternatively:

```bash
export MOBILE_MANIPULATION_SO101_GT_USD=/absolute/path/to/so101_gt.usd
```

The SO-101 Lift task also references the shared Isaac Lab DexCube asset; ensure network asset access or an appropriate cache. Perception and iRobot robot variants remain incomplete and their assets are not provided. Review asset redistribution rights before sharing any models.

## Validate and experiment

From `Embodied-AI-Lab/mobile_manipulation/`, with the environment active and assets restored:

```bash
python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180
python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180
```

Native Kit shutdown has crashed/hung on the original installation. The existing `--skip-kit-cleanup` flag skips normal teardown for diagnostics; it does not fix shutdown. See the preserved project reports for limitations.

The existing training entry point is:

```bash
python scripts/rsl_rl/train.py --task SO101-GT-Reach-v0 --num_envs 1 --headless
```

Choose environment counts and devices to match your hardware. This command is an entry point, not evidence of a successful complete training run. Logs and checkpoints remain local and ignored. Successful physical grasp/lift is still unproven.

## Record an environment without publishing private paths

On the original machine, optionally capture inventories outside the repository:

```bash
conda env export -n env_isaaclab --no-builds > /tmp/embodied-ai-environment.yml
conda run -n env_isaaclab python -m pip freeze > /tmp/embodied-ai-pip-freeze.txt
```

Review editable paths, package URLs and credentials before sharing. These inventories are not portable lockfiles; reinstall editable Isaac Lab and project packages at their new paths. Record the Isaac Lab revision, simulator version, GPU/driver, and external extensions. A fresh Ubuntu installation has not been tested as part of preparing this repository.

## Git workflow

The parent repository uses `main` and the GitHub remote above. The project directory is not a nested repository or submodule. Review staged files before committing; preserve checkpoints and datasets outside Git. Pull before editing on a second computer and push reviewed source changes from the first. Do not run the historical standalone remote commands in the preserved project guide.
