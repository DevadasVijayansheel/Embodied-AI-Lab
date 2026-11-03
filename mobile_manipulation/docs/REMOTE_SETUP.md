# Remote repository and environment setup

## Upload the existing repository

This repository already has Git history. Its current branch is `master`, with
uncommitted work and no configured remote. Do not initialize it again.

Create an empty repository on GitHub named `mobile_manipulation`. Choose private
if appropriate. Do not initialize a README, .gitignore or license on GitHub.
Replace YOUR_USERNAME below. The SSH URL requires GitHub SSH authentication;
you can instead use HTTPS with a credential manager or personal access token.

```bash
cd /home/intrinsic/mobile_manipulation
git status --short
git add README.md .gitignore docs scripts source tools
git diff --cached --stat
git diff --cached
# Review staged files and existing history for private information before publishing.
git commit -m "Save mobile manipulation project and setup documentation"
git remote add origin git@github.com:YOUR_USERNAME/mobile_manipulation.git
git push -u origin master
```

Keep `master` as the branch name; renaming is unnecessary. Never commit tokens
or passwords. These commands are instructions; this guide does not configure a
remote, commit, or push automatically.

[Official GitHub instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## Environment required on the other machine

This is an external Isaac Lab project. Install Isaac Sim and Isaac Lab separately
using their version-specific upstream installation instructions, including the
Lab RL dependencies. Installing this project's Python package alone is insufficient.

Development versions recorded in this repository:

- Python 3.11, Isaac Lab 2.3.2.
- RSL-RL 3.1.2, recorded in the PPO smoke-test report.
- Current local Isaac Sim VERSION: `5.1.0-rc.19+release.26219.9c81211b.gl`.
- Conda environment name on this machine: `env_isaaclab`.

These identify the original setup; a fresh-machine installation has not been
validated. Use a supported NVIDIA GPU/driver for GPU simulation and training.
Avoid upgrading the stack while reproducing an existing experiment.

Record the environment on the original machine before moving:

```bash
conda env export -n env_isaaclab --no-builds > /tmp/mobile_manipulation-environment.yml
conda run -n env_isaaclab python -m pip freeze > /tmp/mobile_manipulation-pip-freeze.txt
```

Transfer these inventories separately. They are not verified portable installers:
review local editable paths, private package URLs and platform dependencies.
Reinstall editable Isaac Lab packages from their new locations. Also record the
Isaac Lab Git revision, GPU driver, and separately installed simulator extensions.

## Assets and training results need separate transfer

The .gitignore excludes all USD files. Git clone will not include the existing
robot asset:

```text
source/mobile_manipulation/mobile_manipulation/robots/usd/so101_gt.usd
```

Copy this asset with all referenced meshes, textures and USD dependencies,
preserving relative paths. Repair any absolute references to the original
machine, or export a self-contained asset. Restore it at the path above or set:

```bash
export MOBILE_MANIPULATION_SO101_GT_USD=/absolute/path/to/so101_gt.usd
```

Lift also needs the shared Isaac Lab DexCube asset and working asset access/cache.
Perception/iRobot assets and experiments remain incomplete.
Logs, outputs, datasets, videos and checkpoints are also ignored. Transfer the
needed `logs/rsl_rl/` directories separately if resuming trained policies.

## Clone, install and check

After installing Git, configuring GitHub authentication, and setting up Isaac Lab:

```bash
git clone git@github.com:YOUR_USERNAME/mobile_manipulation.git
cd mobile_manipulation
conda activate env_isaaclab
# For standalone Isaac Sim: replace this with its actual installation path.
source /absolute/path/to/isaacsim/setup_conda_env.sh
python -m pip install -e source/mobile_manipulation
python -m mobile_manipulation.validate
```

Activate whichever environment contains Isaac Lab; its name need not match the
original. Use runtime setup appropriate to your Isaac Sim installation method.
Lightweight validation checks registrations/source; it does not run physics.
Restore the robot asset, then try:

```bash
python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180
```

If native shutdown crashes/hangs as on the original installation, the existing
`--skip-kit-cleanup` diagnostic flag skips normal teardown; it does not fix that
problem. See the main README for runtime limitations. Successful physical grasp
and lift remain unproven.

## Daily work between machines

On the machine with your changes, review, commit and push:

```bash
git status
git add path/to/changed_file
git commit -m "Describe the change"
git push
```

On the other machine, with a clean working tree, run `git pull --ff-only` before
editing. If both machines have independent work, commit or stash local changes
and resolve diverging history deliberately; do not overwrite work with force push.
