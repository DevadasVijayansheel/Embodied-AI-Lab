# SO-101 GT Reach and Lift

Historical task/table report. The current ground-only task and MDP organization is
documented in [SO101_GT_GROUND_MDP.md](SO101_GT_GROUND_MDP.md).

Both registered tasks use `isaaclab.envs:ManagerBasedRLEnv` with their local
`joint_pos_env_cfg:JointPosEnvCfg`. These specialize reusable `ReachEnvCfg` / `LiftEnvCfg` parents through
`super().__post_init__()` and explicit robot, TCP, action and command assignments. `EnvCfg` aliases
remain for older class imports. No Gym IDs or PPO configuration files changed.

## Verified asset and geometry

The user's USD, robot configuration, actuator gains, initial joint positions and
zero initial velocities are unchanged. `so101_gt_cfg.replace(...)` supplies each scene's
robot, with launch-time file-existence validation and optional local path override.

Runtime confirmed these six joints:
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
Bodies are `base_link`, `shoulder_link`, `upper_arm_link`, `lower_arm_link`, `wrist_link`,
`gripper_link`, `moving_jaw`, `gripper_frame_link`. The saved articulation root is
`base_link`; Isaac Lab's fixed-base spawn creates a world anchor and moves the root
API to the parent, so the articulation configuration retains automatic root discovery.

The grasp frame is tracked on `gripper_link` using the actual fixed
`gripper_frame_link` transform: translation approximately
(-0.007900012, -0.000218121, -0.098127412) meters, with its measured quaternion.
The sensor source prim is `Robot/base_link`; target prim is `Robot/gripper_link`.
Both tasks track this TCP through `FrameTransformerCfg`. Commands retain the physical
`gripper_link` body name for official command initialization, while custom command
metrics, Reach observations/rewards/success and Lift approach observations/rewards
use the tracked TCP. Lift goal rewards continue to track the cube, as intended.

The fixed support is a local 0.65 x 0.50 x 0.04 meter collision cuboid. Its top is
-0.07870018194 meters, matching the lowest saved base collision geometry. This
preserves the saved negative base height without sinking the base into a z=0 plane.
The ground plane is at z=-0.4 meters, below the table. Lift uses an 18 mm, 10 gram
cube whose bottom resets 0.5 mm above the tabletop. Gravity settles it onto the
surface. Support, object mass, friction and task thresholds are initial benchmark
choices, not robot actuator changes. No online or external-repository assets are needed.

## Reach

The fixed TCP target is (0.1903325521395122, -0.032209839850018206,
0.03367581183852951) meters in the robot-root frame. It is computed from USD
joint local frames at the saved pose plus 12 degrees of shoulder_pan, including
the fixed gripper_frame_joint. This replaces the old body-origin target to preserve
the intended joint posture when measuring the TCP. The constant-action seed alone
misses the unchanged 8 mm threshold under gravity; the diagnostic now applies
TCP Jacobian feedback after 30 steps. This feedback exists only in the test script.
Target orientation is unused by the task; command orientation metrics are diagnostic only.

Policy observation dimension: **23**:

- Six relative joint positions and six relative joint velocities.
- Three target coordinates and three TCP coordinates in the robot-root frame.
- Five previous arm actions.

Actions: five ordered joint-position targets, with the saved pose as offset and
scale 0.5 radians/action unit. Processed targets are clipped to inspected USD hard
limits. The gripper is not a Reach action; reset sets its saved position and target.

Rewards: negative Euclidean reaching distance (-1), tanh reaching reward (1,
length scale 0.04 m), action-rate penalty (-0.001), action-magnitude penalty (-0.0001).
Success is distance <8 mm; timeout is 10 seconds. No orientation objective or joint
reset randomization is applied.

## Lift

Policy observation dimension: **27**:

- Six relative joint positions and six relative joint velocities.
- Three cube coordinates, three grasp-frame coordinates and three goal coordinates,
  all expressed in the robot-root frame.
- Six previous actions.

Actions: the same five clipped arm position targets plus one binary gripper action:
non-negative opens to 0.5 rad; negative closes to 0.0 rad. This is the
verified installed BinaryJointPositionActionCfg API; the previous test incorrectly
assumed zero closed the gripper. Both values are inside
the USD gripper limits and are retained from the inspected existing lift controller.
The robot still resets to the exact saved closed-jaw pose; opening happens through actions.

The cube goal is 7 cm above its reset center. Rewards reuse official Isaac Lab lift
MDP functions: reaching the cube (1, tanh scale 0.04 m), cube center lifted 5 cm above
the tabletop resting center (10), elevated goal tracking (5, tanh scale 0.04 m),
action-rate (-0.001) and action-magnitude (-0.0001) penalties.

Success requires the elevated-height threshold, distance <15 mm to the goal, and
linear cube speed <0.1 m/s. A cube center falling 5 cm below the tabletop terminates
as failure. Timeout is 10 seconds. Robot and object resets restore defaults and joint
targets using the official `reset_scene_to_default(reset_joint_targets=True)` event;
there is no reset randomization.

## Original validation results and limitations

Isaac Lab version: 2.3.2; existing Conda Python: 3.11; single environment; CPU physics;
physics dt 1/120 s, control decimation 4 (30 Hz).

- All project Python sources passed syntax/source checks; all eleven Gym registrations remain.
- Both task configurations instantiate, validate without environment MISSING fields, and resolve
  scene entities, commands, actions, observations, events, rewards and terminations at runtime.
- Both ran 180 steps with finite policy observations, robot/object state and rewards.
- Reach completed 15 successful scripted reaches, including automatic resets.
- Lift's cube settled at z approximately -0.06970022 m, matching the tabletop plus its half-height.
  The gripper open/close phases and grasp-frame sensor executed without runtime errors.
- Repeated manual resets reproduced the Inspector joint positions, zero velocities and saved base pose.
- Lift height, goal-success, drop-failure and timeout predicates were checked with injected
  diagnostic states. Those checks do not constitute a physical grasp or successful lift.
- A physical grasp/lift, robust grasp geometry, learning convergence and multi-environment GPU
  behavior remain unvalidated. The tests use one environment as requested.
- The USD still reports an unresolved `/visuals/gripper_frame_link` visual reference. Its near-zero
  mass/zero inertia frame body remains authored as-is. No additional joint/body mismatch was found.
- Existing native Kit teardown crashes/hangs remain unresolved. The diagnostic runner's optional
  `--skip-kit-cleanup` flushes reports then exits the process to avoid teardown; it does not repair Kit.
- At the time of these single-instance tests, PPO configs were preserved. The inherited runner config had MISSING `obs_groups`.
  Training requires an explicit policy/critic mapping such as
  `{"policy": ["policy"], "critic": ["policy"]}` in the agent config separately; this task explicitly preserves those files.
  No runner was instantiated and no PPO training was performed.

Reach's subsequent PPO mapping correction and bounded parallel-test results are documented
in [SO101_GT_REACH_PPO_SMOKE.md](SO101_GT_REACH_PPO_SMOKE.md). Lift's agent file remains unchanged.

Runtime JSON reports from the implementation session are in
`/tmp/so101_reach_runtime.json` and `/tmp/so101_lift_runtime.json` (temporary artifacts).
The tests hash the USD and verify that it stays unchanged.

## Current refactor validation

See [SO101_GT_REFACTOR.md](SO101_GT_REFACTOR.md) for the TCP architecture, current
single-instance physics results and the diagnostic-only material bypass needed
for the existing native scene-material stall. No PPO files were changed in this refactor.

## Commands

```bash
conda activate env_isaaclab
source /home/intrinsic/isaacsim/setup_conda_env.sh
cd /home/intrinsic/mobile_manipulation
export PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH"
python -m mobile_manipulation.validate
python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180 --skip-kit-cleanup
python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180 --skip-kit-cleanup
```

Remove `--headless` for a visible single-instance test. Add `--config-only` for imports
without simulation, or `--report /tmp/report.json` to save results. Use normal Kit
cleanup by omitting `--skip-kit-cleanup`, with the known teardown risk above.

## Files changed for task completion

- Added `tasks/manager_based/so101_gt/shared.py` for inspected joint limits, support geometry,
  frame constants and common environment setup.
- Updated `so101_gt/reach/reach_env_cfg.py`, `reach/joint_pos_env_cfg.py`, and `reach/__init__.py`.
- Updated `so101_gt/lift/lift_env_cfg.py`, `lift/joint_pos_env_cfg.py`, and `lift/__init__.py`.
- Updated `mdp/observations.py`, `mdp/rewards.py`, `mdp/terminations.py`, `mdp/events.py` in each skill.
- Updated `mobile_manipulation/validate.py` to keep CPU-only discovery working with official Gym entry points.
- Added `tools/test_so101_gt.py`; updated `tools/validate_so101_gt.py` and
  `tools/validate_isaaclab_configs.py` for the completed configuration classes.
- Updated `README.md`, added this report, and appended the current task status to `SO101_GT_ASSET.md`.

Robot configuration, robot USD, PPO files, template task, original scripts, perception tasks
and iRobot tasks were not modified by this completion work.
