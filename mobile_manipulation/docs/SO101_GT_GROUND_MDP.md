> Lift reward, observation and simulation settings below are historical. The latest pasted-code adaptation is documented in [SO101_GT_LIFT_REFERENCE_CFG.md](SO101_GT_LIFT_REFERENCE_CFG.md). Ground placement and verified TCP remain valid.

# SO-101 GT ground-only scene and task-local MDP

This report supersedes the earlier table/shared-module architecture reports.
The configuration and MDP organization was compared with the reference
[Reach parent](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/reach/reach_env_cfg.py),
[Lift parent](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/lift/lift_env_cfg.py)
and [Lift MDP rewards](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/lift/mdp/rewards.py),
and the installed Isaac Lab 2.3.2 implementation. No external repository imports
were introduced and no external robot/TCP/scene values were copied.

## Architecture and MDP locations

`so101_gt/shared.py` and `so101_gt/tcp.py` were removed after their functionality
was migrated, dependent imports eliminated and the specialized configs validated.
Obsolete compiled-module caches were also removed. No replacement shared task module
was created. Robot joint names/limits, verified TCP offset and ground-root placement
metadata now live in `robots/so101_gt_cfg.py`.

Both tasks retain their generic parent, `JointPosEnvCfg` specialization and
`JointPosEnvCfg_PLAY`. The parents own scene/managers/timing; the subclasses assign
the local robot, frame transformer, actions and commands through `super().__post_init__()`.
All subclass MISSING fields are filled. Both PLAY classes validate and default to
one environment with observation corruption disabled.

| File | Functions/classes used or exported |
| --- | --- |
| `reach/mdp/observations.py` | Local `target_position`, `end_effector_position`, `TcpPoseCommand`, `TcpPoseCommandCfg`; official `joint_pos_rel`, `joint_vel_rel`, `last_action` |
| `reach/mdp/rewards.py` | Local TCP `position_command_error`, `position_command_error_tanh`; official `action_rate_l2`, `action_l2`; retained unused `joint_vel_l2` export |
| `reach/mdp/terminations.py` | Local `reached_target`; official `time_out` |
| `reach/mdp/events.py` | Official `reset_scene_to_default` with reset_joint_targets=True supplied by the parent event |
| `lift/mdp/observations.py` | Task-local target/TCP observation functions and TCP command classes; official joint/action observations; reference-local `object_position_in_robot_root_frame` |
| `lift/mdp/rewards.py` | Reference-local `object_ee_distance`, `object_is_lifted`, `object_goal_distance`, `object_ee_distance_and_lifted`; official `action_rate_l2`, `action_l2` |
| `lift/mdp/terminations.py` | Local `lifted_at_goal`, composing reference-local `object_reached_goal` with height and speed; official `root_height_below_minimum`, `time_out` |
| `lift/mdp/events.py` | Official `reset_scene_to_default` with reset_joint_targets=True |

Official functions were inspected: joint observations subtract configured defaults;
object observation transforms the cube into robot-root coordinates; approach reward
uses `ee_frame.target_pos_w`; lift reward checks the cube center's world z;
goal reward compares the cube to a transformed root-frame goal and gates on height;
reset restores default states and requested joint targets. These semantics match
the existing tasks. The Lift kernels are kept locally to follow the requested reference MDP layout.

Small TCP observation/command duplication between tasks is intentional: neither task
imports the other's MDP. Reach reward kernels remain local because official
body-origin reach kernels cannot represent this fixed TCP offset.

## Ground placement

The table is absent from both scene definitions and spawned stages. The Isaac Lab
ground plane at **z=0** is the sole support surface. Ground retains friction coefficients 1.0 and zero restitution. The shared DexCube uses its authored collision/material settings and 0.216 kg mass, with the requested rigid-body property overrides.

Four enabled base collision geometries were inspected through their actual mesh
vertices, including transforms. Lowest saved world vertex:
`-0.07870017894732786 m`, on `base_so101_v2/node_STL_BINARY_/mesh`.
Root-relative lowest z is `-0.0024002582691891317 m`.

```python
saved_root = (-2.123415470123291e-7, 1.3396129361353815e-7, -0.07629992067813873)
ground_root = (-2.123415470123291e-7, 1.3396129361353815e-7, 0.0024002582691891317)
vertical_shift = 0.07870017894732786

ground_z = 0.0
cube_size = (0.03, 0.03, 0.03)
cube_reset = (0.2, 0.0, 0.015)
```

Only root translation changed. Its orientation, saved initial joint angles,
zero initial joint velocities and all actuator configurations were preserved.
The robot USD is unchanged (SHA256
`af007581382c395859b9b77c5ad7d17938e293f701744399215ab529b6173aef`).
The shared DexCube resets at ground contact, with its center at approximately
0.015 m. No replacement platform or support mesh was introduced.

The Reach target stays in robot-root coordinates:
`(0.1903325521395122, -0.032209839850018206, 0.03367581183852951)`.
Moving the root translates its world position consistently; the saved angular pose
plus 12-degree shoulder-pan target and existing diagnostic TCP feedback are retained.

Lift thresholds now reference the ground:

- Lift reward/goal-reward gate: cube center **z > 0.065 m** (50 mm above its resting center).
- Elevated goal: `(0.2, 0.0, 0.085)` in world coordinates,
  70 mm above reset center; subclass converts it to robot-root coordinates.
- Success: existing goal distance <0.015 m, z >0.065 m and cube linear speed <0.1 m/s.
- Dropped-object failure: center **z < -0.05 m**; ordinary ground contact is valid.
- Timeout: existing 10 seconds. Reward weights and 0.04 m tanh scales are unchanged.

Before the DexCube change, joint-limit-constrained USD FK/IK checks found the old cube and goal positions kinematically
reachable, with residuals 3.10e-6 m and 7.77e-15 m respectively. This does not prove
a collision-free grasp trajectory or orientation solution.

## TCP

Both robot-specific subclasses explicitly configure `FrameTransformerCfg`:
source `Robot/base_link`, target physical body `Robot/gripper_link`, offset target
named `gripper_frame_link`. The verified translation and WXYZ quaternion remain:

```python
pos = (-0.00790001197275339, -0.00021812137186685843, -0.09812741222130128)
rot = (1.2606114104306037e-06, -3.0179257313699397e-09,
       0.9999999999992054, 4.439806178955286e-10)
```

Observations, Reach rewards/success, Lift approach reward and command metrics use
the TCP. Lift's goal-tracking reward continues to track the cube.
The `gripper` joint still controls the moving jaw: negative action closes to 0 rad;
non-negative opens to 0.5 rad.

## Shared DexCube and TCP marker

The SO-101 Lift subclass now supplies `RigidObjectCfg` with `UsdFileCfg` pointing to
`ISAAC_NUCLEUS_DIR/Props/Blocks/DexCube/dex_cube_instanceable.usd`, scale `(0.5, 0.5, 0.5)`,
identity orientation and reset position `(0.2, 0.0, 0.015)`. The generic scene's object is
`MISSING` until the subclass supplies it. The asset's verified 60 mm visual/collision
edge becomes 30 mm. Its authored mass remains 0.216 kg; no mass override is supplied.
Solver iterations 16/1, velocity limits 1000, depenetration limit 5 and gravity enabled
follow the user's snippet. This intentionally changes the former 18 mm, 10 g cube.

`FRAME_MARKER_CFG` is copied with frame scale `(0.05, 0.05, 0.05)`,
path `/Visuals/SO101GTLift/FrameTransformer` and `debug_vis=True`. Both frame and
connecting-line prototypes are retained, as required by the local FrameTransformer
callback. The connecting-line preview material is authored directly with supported
USD APIs to avoid the previously observed Kit Sdr query stall; its properties are preserved.
The verified `base_link`, `gripper_link`, TCP translation and WXYZ rotation are retained.
No reference-repository link names or unverified offset is used. The frame USD and
DexCube asset/dependencies must be accessible through the configured shared asset root.

The former procedural cube material helper was removed. Native Kit teardown issues
remain: diagnostics flush reports then use `--skip-kit-cleanup` to exit.

## Physics validation

Existing env_isaaclab, Python 3.11, Isaac Lab 2.3.2, CPU physics, one instance/task.

| Check | Reach | Lift |
| --- | --- | --- |
| Normal scene creation with no material bypass | Passed | Passed |
| Reset/step test | 180 steps | 180 steps |
| Policy/action dimensions | 23 / 5 | 27 / 6 |
| Finite actions/observations/rewards/states | Passed | Passed |
| Deterministic saved-angle, zero-velocity resets | Passed | Passed |
| Base collision bottom vs ground | max 2.03e-8 m discrepancy | max 2.03e-8 m discrepancy |
| TCP vs actual fixed-frame body | max 4.62e-8 m | max 4.61e-8 m |
| TCP quaternion, configured observation/reward and command metric checks | Passed | Passed |
| Behavior | 34 success terminations | Shared cube settles near 0.015 m; no drop failures; gripper opening/closing verified |
| Native articulation/joint/physics errors | None captured | None captured |

The runtime stage was checked for absence of a Table prim. Lift predicates were
also checked with controlled injected states, then the scene was reset before any
physical trial. All eleven Gym registrations remain; the two GT IDs still target
their respective `joint_pos_env_cfg:JointPosEnvCfg`. Both specialized and PLAY
configs validated after removal of the shared modules; syntax/source checks passed.

With the former procedural cube, a separate **240-step physical approach/close/raise trial did not achieve a grasp
or lift**. No object pose was written during that trial. Minimum TCP-to-cube distance
was 0.015874 m; cube maximum center height was 0.011814 m, returning to 0.009 m;
zero success or drop terminations. Grasp orientation/contact control and a demonstrated
physical lift remain research work, rather than validated functionality.

The authored missing `/visuals/gripper_frame_link` visual reference and tiny-mass,
zero-inertia frame body remain unchanged. Their fixed transform and runtime tracking
were verified; no corresponding physics errors were captured.
PPO files/hyperparameters were untouched. Lift still has pre-existing MISSING
`obs_groups`; no PPO training or training-readiness claim is made.

Results: [validation/so101_gt_ground_mdp.json](validation/so101_gt_ground_mdp.json).
Raw reports: `/tmp/so101_reach_ground.json`, `/tmp/so101_lift_ground.json`,
`/tmp/so101_ground_grasp_trial.json`.

## Reproduce

```bash
conda activate env_isaaclab
cd /home/intrinsic/mobile_manipulation
source /home/intrinsic/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH"
python -m mobile_manipulation.validate
timeout --signal=KILL 60s python -u tools/validate_so101_gt.py --headless --device cpu --skip-kit-cleanup
timeout --signal=KILL 60s python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180 --skip-kit-cleanup --report /tmp/so101_reach_ground.json
timeout --signal=KILL 60s python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180 --skip-kit-cleanup --report /tmp/so101_lift_ground.json
timeout --signal=KILL 60s python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 240 --grasp-lift-test --skip-kit-cleanup --report /tmp/so101_ground_grasp_trial.json
```

## Files changed

- Removed `so101_gt/shared.py` and `so101_gt/tcp.py`.
- Updated both parent/subclass environment configurations and task-local MDP packages.
- Updated `robots/so101_gt_cfg.py` with relocated robot constants and justified ground root position.
- Updated validation/physics tools, README, historical-document banners; added this report and results.
- Gym registration files, USD, original scripts, PPO files and other experiment groups were preserved.

### Reference Lift MDP alignment

The Lift object observation, four object reward functions (including `object_ee_distance_and_lifted`), and `object_reached_goal` are now local implementations from the reference Lift MDP, retaining its BSD attribution and formulas. Robot type annotations use `Articulation`. The frame sensor supplies the verified USD TCP; these functions never select a gripper body origin or load a robot asset. Existing TCP observations/command metrics, deterministic reset helpers, action penalties and elevated/settled success wrapper remain. The combined reward is available but is not added to the configured reward terms, preserving existing weights and behavior.

Validation: `docs/validation/so101_gt_lift_reference_mdp.json` records the successful 180-step single-instance CPU physics test after this port. Policy shape is `(1, 27)`, action dimension is 6. No training was started; this is not proof of successful grasping. The existing Lift PPO `obs_groups` omission remains outside this MDP change.

The shared-cube/marker rerun is recorded in `docs/validation/so101_gt_lift_dexcube.json`; older reports describe the former procedural cube. A successful grasp/lift of the shared DexCube has not been demonstrated.
