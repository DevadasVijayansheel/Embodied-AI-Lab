# Mobile manipulation RL research

External Isaac Lab **2.3.2**, Manager-Based, single-agent project using Python 3.11 and RSL-RL PPO.

SO-101 GT **Reach and Lift now have complete environment configurations** and passed
single-instance CPU reset/step tests. Reach also passed a scripted target-reaching test.
A successful physical grasp and lift has not yet been demonstrated. The remaining
perception/iRobot experiments are registered scaffolds. The original cart-pole template is preserved.

Reach/Lift use task-local MDP packages and generic parent/SO-101 joint-position
configurations. Lift uses the shared DexCube USD at scale 0.5 and an enabled verified-TCP frame marker. The robot and cube are ground-supported with no table. Both tasks
passed normal headless scene creation and 180 CPU physics steps without material
bypasses. A physical grasp/lift trial did not succeed. See the
[ground/MDP report](docs/SO101_GT_GROUND_MDP.md) and [current Lift reference configuration](docs/SO101_GT_LIFT_REFERENCE_CFG.md).

Reach and Lift both map RSL-RL policy and critic inputs to their policy observation group. All four zero/random agent combinations passed 100 CPU steps without training; see [agent/PPO input validation](docs/SO101_GT_AGENTS.md).

## GitHub and another machine

See [remote and environment setup](docs/REMOTE_SETUP.md) for pushing this repository,
cloning it, and restoring the simulator environment and robot assets.

## Install and discovery

```bash
conda activate env_isaaclab
cd /home/intrinsic/mobile_manipulation
python -m pip install -e source/mobile_manipulation
python -m mobile_manipulation.validate
```

Without installation, use `PYTHONPATH=source/mobile_manipulation python -m mobile_manipulation.validate`.
No packages were installed or Conda environments modified during implementation.
The preserved `scripts/list_envs.py` only lists IDs containing `Template-`; the lightweight
validation module lists all eleven project registrations and checks source syntax/entry-point classes.

## SO-101 GT single-instance physics tests

The current local Isaac Sim installation needs its runtime setup before starting Python:

```bash
conda activate env_isaaclab
source /home/intrinsic/isaacsim/setup_conda_env.sh
cd /home/intrinsic/mobile_manipulation
export PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH"
python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180 --skip-kit-cleanup
python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180 --skip-kit-cleanup
```

Remove `--headless` to view the tests. `--config-only` imports and instantiates configs
without creating a scene. `--report /tmp/report.json` records test results.
`--skip-kit-cleanup` is an explicit diagnostic option: it flushes results and exits the
process without normal Kit teardown because this installation previously crashed/hung in native shutdown.
It does not fix that shutdown problem. Without the option, normal cleanup is attempted.

The preserved zero-agent script also launches each environment with one instance:

```bash
python scripts/zero_agent.py --task SO101-GT-Reach-v0 --num_envs 1 --device cpu
python scripts/zero_agent.py --task SO101-GT-Lift-v0 --num_envs 1 --device cpu
```

Zero actions hold the five arm joints at the saved pose; Lift's zero gripper action closes the jaw.
See [docs/SO101_GT_TASKS.md](docs/SO101_GT_TASKS.md) for observations, actions, reward definitions,
verified geometry, runtime results and outstanding issues.

## Registrations and checkpoints

| Task | Environment config | PPO experiment name | Status |
| --- | --- | --- | --- |
| SO101-GT-Reach-v0 | so101_gt.reach.joint_pos_env_cfg:JointPosEnvCfg | so101_gt_reach | Runtime smoke test passed |
| SO101-GT-Lift-v0 | so101_gt.lift.joint_pos_env_cfg:JointPosEnvCfg | so101_gt_lift | Runtime smoke test passed; physical lift unproven |
| SO101-Perception-Reach-v0 | so101_perception.reach.reach_env_cfg:EnvCfg | so101_perception_reach | Incomplete scaffold |
| SO101-Perception-Lift-v0 | so101_perception.lift.lift_env_cfg:EnvCfg | so101_perception_lift | Incomplete scaffold |
| iRobot-GT-Navigate-v0 | irobot_gt.navigate.navigate_env_cfg:EnvCfg | irobot_gt_navigate | Incomplete scaffold |
| iRobot-GT-Reach-v0 | irobot_gt.reach.reach_env_cfg:EnvCfg | irobot_gt_reach | Incomplete scaffold |
| iRobot-GT-Lift-v0 | irobot_gt.lift.lift_env_cfg:EnvCfg | irobot_gt_lift | Incomplete scaffold |
| iRobot-Perception-Navigate-v0 | irobot_perception.navigate.navigate_env_cfg:EnvCfg | irobot_perception_navigate | Incomplete scaffold |
| iRobot-Perception-Reach-v0 | irobot_perception.reach.reach_env_cfg:EnvCfg | irobot_perception_reach | Incomplete scaffold |
| iRobot-Perception-Lift-v0 | irobot_perception.lift.lift_env_cfg:EnvCfg | irobot_perception_lift | Incomplete scaffold |

Config module prefixes in the table are `mobile_manipulation.tasks.manager_based.`.
The template ID remains `Template-Mobile-Manipulation-v0`. Default checkpoint directories remain
`logs/rsl_rl/<experiment_name>/<run>/`.

## Assets and modular structure

`robots/so101_gt_cfg.py` exports `so101_gt_cfg`; its USD is the user's existing
`robots/usd/so101_gt.usd`. Its actuators and saved Inspector joint positions are unchanged.
A `robot_usd_path` config override or `MOBILE_MANIPULATION_SO101_GT_USD` environment variable
can select another local file, but its hierarchy/joint/frame compatibility must be verified.

The missing future assets remain `so101_perception.usd`, `irobot_gt.usd`, and
`irobot_perception.usd`, with matching `MOBILE_MANIPULATION_<VARIANT>_USD` overrides.
No placeholder USD files were generated. The camera configuration factories live only
in perception skills and have no PPO dependencies. Perception and iRobot task code is unchanged.

Verified SO-101 joint/TCP/ground-placement properties live in `robots/so101_gt_cfg.py`.
Each GT skill owns its scene and MDP implementation; task-level shared modules were removed.
Each GT skill owns its environment, observation, reward, event and termination configuration.
Inspected official Isaac Lab MDP functions are reused through each skill's MDP package.
Local TCP observations, reach kernels, command metrics and success predicates handle the
verified offset where body-origin functions would have different semantics.
The generic `common/` scaffold remains the base for incomplete experiments and for the
unchanged template-derived PPO defaults.

## Original template documentation


# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/mobile_manipulation

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/mobile_manipulation/mobile_manipulation/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/mobile_manipulation"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
