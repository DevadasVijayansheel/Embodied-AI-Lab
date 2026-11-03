# Embodied AI Lab

A research workspace for learning how robots perceive, act, and interact with the physical world. The current project develops simulated mobile manipulation tasks with Isaac Lab and reinforcement learning.

## Research scope

Embodied AI studies intelligence through an agent's interaction with an environment: observations inform actions, and actions change what the agent observes next.

- **Reinforcement learning (RL)** trains policies through rewards and interaction. The current project uses RSL-RL PPO configurations for robot reach and lift tasks.
- **Imitation learning (IL)** learns behavior from demonstrations. It is a future research direction here; demonstration collection and IL training are not implemented.
- **Vision-language-action (VLA)** models connect visual observations and language instructions to robot actions. VLA integration is a future direction, not an available feature.
- **Isaac Sim and Isaac Lab** provide the simulation runtime and robot learning environment framework used by the current project.

## Current project: mobile manipulation

[mobile_manipulation/](mobile_manipulation/) is an external Isaac Lab Python project with robot configurations, environment definitions, MDP components, agent scripts, and PPO training entry points.

SO-101 ground-truth Reach and Lift configurations have passed documented single-instance CPU physics smoke tests. Reach also passed a scripted target-reaching test. A successful physical grasp and lift has not been demonstrated. Perception and iRobot variants are incomplete scaffolds. The original cart-pole template is preserved. These results do not establish policy convergence or real-world deployment readiness.

```text
Embodied-AI-Lab/
├── README.md
├── docs/UBUNTU_SETUP.md
└── mobile_manipulation/
    ├── README.md
    ├── docs/
    ├── scripts/
    ├── source/mobile_manipulation/
    └── tools/
```

## Getting started

Follow [Ubuntu installation and reproduction](docs/UBUNTU_SETUP.md) for software versions, simulator setup, editable installation, robot assets, and validation commands. Read the preserved [project README](mobile_manipulation/README.md) for task details and validation reports.

Run project commands from `mobile_manipulation/`, not the repository root. The preserved [original remote setup guide](mobile_manipulation/docs/REMOTE_SETUP.md) describes the former standalone repository; its old remote/branch instructions are historical. Use the parent setup guide for this repository.

## Repository contents and reproducibility

Only source, configuration, scripts, and documentation are included. Model checkpoints, weights, training outputs, datasets, caches, environments, and robot USD assets are excluded. Supply robot assets separately with all their dependencies. Referenced generated JSON validation reports are not shipped; narrative validation reports remain in the project documentation.

This is a source snapshot of an existing local project, stored as an ordinary directory rather than a submodule. Original Git history remains in the original local repository and a separate local backup; it is not imported into this parent's public history. No simulator installation or trained policy is bundled.

Existing source headers and upstream attribution are retained. A repository-wide license for newly authored material has not been selected; review applicable source and asset licenses before redistribution.
