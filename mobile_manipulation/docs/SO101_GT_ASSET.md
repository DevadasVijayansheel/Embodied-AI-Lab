# SO-101 GT USD migration

The existing configuration was found in
`/home/intrinsic/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`.
That source file was inspected and left unchanged. Its configuration was ported into
`source/mobile_manipulation/mobile_manipulation/robots/so101_gt_cfg.py`, exporting
`so101_gt_cfg = ArticulationCfg(...)`.

## Loading and scene use

`UsdFileCfg.usd_path` is the absolute string derived from
`Path(__file__).resolve().parent / "usd" / "so101_gt.usd"`.
The GT reach/lift environment configs use `make_so101_gt_cfg(...)`, which copies the
exported configuration and validates file existence. The factory also supports
`MOBILE_MANIPULATION_SO101_GT_USD` and an explicit `usd_path` override.

After Kit starts, this scene configuration operation is valid and was tested:

```python
from mobile_manipulation.robots.so101_gt_cfg import so101_gt_cfg
robot = so101_gt_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

The saved USD has one enabled articulation root at `/so101_new_calib/base_link`,
but no world-fixed joint. `fix_root_link=True` preserves the original URDF's
`fix_base=True` in the spawned stage. Isaac Lab 2.3.2 moves the root API to the
parent when creating that fixed joint, so automatic root discovery is retained.
No changes were saved to the USD.

## Preserved settings

| Joint | Stiffness | Damping |
| --- | ---: | ---: |
| shoulder_pan | 200 | 80 |
| shoulder_lift | 170 | 65 |
| elbow_flex | 120 | 45 |
| wrist_flex | 80 | 30 |
| wrist_roll | 50 | 20 |
| gripper | 60 | 20 |

Arm group expressions remain `shoulder_.*`, `elbow_flex`, `wrist_.*`; the gripper
expression remains `gripper`. Arm effort limit is 1.9, gripper effort limit is 2.5,
and both velocity limits are 1.5. Joint velocities remain `{".*": 0.0}`.
At the user's subsequent request, the initial base pose and joint positions now
match the saved USD state, rather than the original URDF configuration. Angular
positions are converted from degrees to radians: shoulder_pan -0.0000132608,
shoulder_lift -1.7452578793, elbow_flex 1.5864393356, wrist_flex 0.9372432864,
wrist_roll -0.0000000771, gripper -0.0000000435. The saved base translation is
approximately (0, 0, -0.07629992) meters, with identity orientation. Use an
appropriate scene support height when testing this authored pose.
Soft-limit factor 0.9, gravity enabled, maximum depenetration velocity 5.0, solver
iterations 8/0 and contact sensors disabled are preserved. An AST comparison
confirmed all actuator settings still match the original exactly.

Self-collisions are disabled as requested: the original and USD enabled them
without a documented reason in the inspected configuration. URDF-only cylinder
conversion and converter joint-drive settings are not applicable to UsdFileCfg.
USD drives are retained; implicit actuators apply the preserved configured gains
and effort/velocity limits to the simulation at runtime.

## USD findings and differences

All six controlled joint names match, with no renaming. Initial positions are
inside the USD hard limits:

| Joint | USD angular limits (degrees) |
| --- | --- |
| shoulder_pan | -110 to 110 |
| shoulder_lift | -100 to 100 |
| elbow_flex | -96.8299 to 96.8299 |
| wrist_flex | -94.9998 to 94.9998 |
| wrist_roll | -157.2110 to 162.7893 |
| gripper | -10 to 100 |

The extra `gripper_frame_joint` is fixed and does not need an actuator.
All six controlled joints have enabled angular force drives. USD stiffness is 70
for each; damping is 0.3 except shoulder_lift 0.4 and elbow_flex 0.5; maxForce is 10.
USD maximum joint velocity is approximately 572.958 degrees/s (10 rad/s).
These differ from the preserved actuator settings and are overridden at runtime,
not rewritten in the asset. Saved joint states are used for initialization, rather
than drive targets. In particular, elbow_flex's saved state is 90.896278 degrees,
while its drive target is 96.8 degrees. Reset/first-step transients require a physics test.
USD solver iterations are 32/1; the original configuration's 8/0 are preserved.

The stage is Z-up, meter units, with eight enabled non-kinematic rigid bodies,
gravity not disabled, and 34 collision prims including instance proxies.
Physics connectivity is present. Stability and collision suitability are untested.
`gripper_frame_link` has near-zero mass and zero diagonal inertia; review PhysX
warnings during the first test. Its visual references `/visuals/gripper_frame_link`,
which does not exist, producing an unresolved-reference warning. The asset was
not repaired, and its collision meshes and joint-drive schemas were not changed.

## Validation and readiness

Syntax checks and actual configuration imports/instantiation succeeded using
Python 3.11 in `env_isaaclab`, after starting Kit. Joint mappings, actuator group
coverage, initial hard-limit checks, articulation schema, gravity and scene
`replace()` operation passed. The asset SHA-256 was unchanged during validation.
Actual fixed-base spawning, physics stepping and stability have not been tested.
The Kit validator encountered a native segmentation fault during shutdown after
printing its PASS result; an earlier run stalled at shutdown and was stopped.
This cleanup issue remains unresolved and is not a clean end-to-end process pass.

To reproduce from the project root:

```bash
conda activate env_isaaclab
source /home/intrinsic/isaacsim/setup_conda_env.sh
cd /home/intrinsic/mobile_manipulation
PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH" python -u tools/validate_so101_gt.py --headless --device cpu
```

The robot configuration is ready for a supervised physics smoke test, with the
asset warnings above. The RL reach/lift tasks still lack task-specific geometry,
actions, observations, commands, rewards and success/failure logic. Their existing
incomplete-task launch guards and registered IDs remain intact. Perception and
iRobot configs, rewards and PPO settings were not modified. No training was run.

The latest initial-state change was checked with the installed USD libraries in
`env_isaaclab` without launching Kit: all six saved positions and the base world
pose match, positions are inside hard limits, actuator settings and zero joint
velocities are preserved, syntax passes, and the asset checksum is unchanged.
Kit imports were not repeated for this numeric initial-state change.


## Subsequent GT task completion

Reach and Lift now have task-specific environment configurations and passed single-instance
CPU physics reset/step tests. See [SO101_GT_TASKS.md](SO101_GT_TASKS.md) for the current
scene, frame, action, reward and validation details. The earlier incomplete-task status above
is historical. The robot configuration and saved USD remain unchanged by task completion.
