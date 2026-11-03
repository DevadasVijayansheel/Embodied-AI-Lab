# SO-101 GT task architecture and TCP validation

Historical task/table report. The current ground-only task and MDP organization is
documented in [SO101_GT_GROUND_MDP.md](SO101_GT_GROUND_MDP.md).

The configuration hierarchy follows the `super().__post_init__()` specialization
pattern in the [reference Reach configuration](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/reach/joint_pos_env_cfg.py).
No external repository imports or robot/frame constants were introduced.

## Responsibilities

| Configuration | Responsibilities |
| --- | --- |
| `ReachEnvCfg` | Generic task managers, TCP-based observations/reach kernels/success, deterministic reset event, timeout, support scene, simulation timing; placeholders for robot, TCP sensor, arm action, command body and target ranges |
| `LiftEnvCfg` | Generic task managers, tested cube/support geometry, object/TCP/goal observations, existing approach/lift/object-goal rewards and drop/success predicates; the same robot/control placeholders plus gripper action |
| Reach `JointPosEnvCfg` | Calls parent; assigns local `so101_gt_cfg`, validates USD path, configures TCP sensor, five ordered/clipped joint-position actions and verified TCP target |
| Lift `JointPosEnvCfg` | Calls parent; assigns the same robot/TCP/arm control, binary `gripper` action and existing cube-relative elevated goal |
| Each `JointPosEnvCfg_PLAY` | Calls parent; one environment, spacing 2.5 m, policy corruption disabled |

The training defaults remain one environment to preserve the tested configuration.
PLAY also uses one environment, the smallest valid scene; it is not numerically
smaller than the already single-instance training default. Neither parent is
directly launchable: its deliberate `MISSING` fields are filled by the subclasses.
Both specialized configs and both PLAY configs passed `.validate()` with no
unresolved environment fields.

`shared.py` holds inspected joint/geometry constants, the TCP-sensor factory, generic
support-scene placeholders and timing defaults. `tcp.py` contains the shared TCP
observation/reach kernels and a pose-command subclass that retains official sampling
but computes tracking metrics and optional markers at the sensor TCP.
Lift's cube-to-goal reward still tracks the cube, not the TCP; replacing it would
change the lifting task. Lift's approach reward uses the TCP, as it did before.

## Verified fixed TCP

USD hierarchy:

```text
/so101_new_calib/gripper_link
    -- gripper_frame_joint (fixed) --> /so101_new_calib/gripper_frame_link
```

Authored `gripper_link` to `gripper_frame_link` transform, meters and quaternion **wxyz**:

```python
pos = (-0.00790001197275339, -0.00021812137186685843, -0.09812741222130128)
rot = (1.2606114104306037e-06, -3.0179257313699397e-09,
       0.9999999999992054, 4.439806178955286e-10)
```

The fixed joint's local frames independently produce:

```python
pos = (-0.007899999618530273, -0.0002181210002163425, -0.09812740236520767)
rot = (1.2301777332966709e-06, 0.0, 0.9999999999992434, 0.0)
```

The matrices agree within 1e-6, consistent with USD float precision. Both tasks use
the already authored/previously verified Lift offset. The sensor source is
`Robot/base_link`; its target rigid body is `Robot/gripper_link`, named
`gripper_frame_link` with the full translation and rotation offset. The actuated
`gripper` joint controls the moving jaw and is distinct from these frames.

The unresolved visual reference is an authored reference from
`/so101_new_calib/gripper_frame_link/visuals` to absent `/visuals/gripper_frame_link`.
It does not remove the fixed joint, rigid body or verified transform. No USD changes
were made. Its tiny mass and zero authored diagonal inertia remain unchanged;
no articulation/joint/physics errors were captured in the completed tests.

## Intentional TCP consistency correction

Reach previously measured the `gripper_link` origin. Its target now represents the
TCP at the same saved joint pose plus 12 degrees of shoulder-pan motion:

```python
REACH_TARGET = (0.1903325521395122, -0.032209839850018206, 0.03367581183852951)
```

The target was calculated from actual USD joint frames and saved angular states,
including `gripper_frame_joint`; validation independently recomputes it.
Observations, both reach rewards, success and command metrics all use the sensor TCP.
Reward weights, tanh scale 0.04 m, success threshold 8 mm and timeout remain unchanged.

The original constant +12-degree action seed settled 10.7 mm away from the TCP
target under physics. The test controller therefore adds damped TCP Jacobian
feedback after 30 steps, integrating position-target corrections to compensate
gravity deflection. This exists only in the diagnostic, not in environment control,
robot gains, reset pose or task logic. Thus the exact old constant-action sequence
does not pass the TCP success threshold; the intended target-reaching behavior does.

The old Lift diagnostic incorrectly assumed zero closed the binary gripper.
Inspection of the installed API confirmed **negative closes; non-negative opens**.
The test now sends -1 during the close phase. Configuration values remain open
0.5 rad and closed 0.0 rad.

## Runtime results

Existing `env_isaaclab`, Isaac Lab 2.3.2, CPU physics, one environment, 180 steps/task.
Both actual Gym registrations still resolve to `joint_pos_env_cfg:JointPosEnvCfg`.

| Check | Reach | Lift |
| --- | --- | --- |
| Observation/action dimensions | 23 / 5 | 27 / 6 |
| Finite observations, rewards and states | Passed | Passed |
| Inspector pose/zero-velocity repeated resets | Passed | Passed |
| TCP sensor vs actual fixed-frame body | max 4.62e-8 m | max 4.49e-8 m |
| TCP quaternion/observation/reward/command-metric checks | Passed | Passed |
| Task behavior | 34 success terminations; minimum pre-reset distance 0.002376 m | Cube supported at approximately -0.06970022 m; gripper opens to 0.499883 rad and closes to 0.000024917 rad |
| Native articulation/joint/physics errors | None captured | None captured |

Lift's lift-height, goal-success, drop and timeout predicates were also checked with
injected diagnostic states. A real grasp and successful physical cube lift were not
demonstrated. Detailed summary: [validation/so101_gt_refactor.json](validation/so101_gt_refactor.json).
Raw trajectory reports are `/tmp/so101_reach_refactor_physics.json` and
`/tmp/so101_lift_refactor_physics.json`.

**Validation limitation:** Default Reach scene creation again stalled and was stopped
by an external 90-second timeout before physics started. This is consistent with
the previously captured native shader-registry traceback. Completed physics tests
used `--skip-procedural-materials` to skip only procedural table/cube visual-material
creation in the diagnostic config. Collision geometry, physics materials, mass,
robot USD, scene entities and task logic were retained. Neither production task
configuration was changed to bypass materials. Default rendered scene creation
remains unvalidated; the material stall is unresolved. Native Kit teardown was
bypassed with `--skip-kit-cleanup` after reports were flushed, as in earlier tests.

## Reproduce checks

```bash
conda activate env_isaaclab
cd /home/intrinsic/mobile_manipulation
source /home/intrinsic/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH"
python -m mobile_manipulation.validate
timeout --signal=KILL 90s python -u tools/validate_so101_gt.py --headless --device cpu --skip-kit-cleanup
timeout --signal=KILL 90s python -u tools/test_so101_gt.py --task SO101-GT-Reach-v0 --headless --device cpu --steps 180 --skip-procedural-materials --skip-kit-cleanup --report /tmp/so101_reach_refactor_physics.json
timeout --signal=KILL 90s python -u tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180 --skip-procedural-materials --skip-kit-cleanup --report /tmp/so101_lift_refactor_physics.json
```

Omit `--skip-procedural-materials` to test complete scene creation, subject to the
native stall. No PPO training was started, and PPO configuration files are unchanged
by this refactor. Lift's pre-existing missing PPO `obs_groups` remains.

## Files changed

- Both skills' generic environment and joint-position configuration files.
- Reach MDP observations/rewards/terminations and Lift MDP observations to share TCP logic.
- `so101_gt/shared.py`; added `so101_gt/tcp.py`.
- `tools/test_so101_gt.py` and `tools/validate_so101_gt.py` for TCP/physics validation.
- README and task documentation; added this report and validation summary.

Gym registration files, robot configuration/asset, cube/table physical settings,
Lift rewards/terminations/events and PPO files were preserved.
