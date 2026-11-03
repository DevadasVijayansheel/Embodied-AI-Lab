> Update: Lift PPO observation mapping and four bounded zero/random-agent checks are now complete. See [SO101_GT_AGENTS.md](SO101_GT_AGENTS.md); prior missing-obs_groups notes below describe the earlier state.

# Lift configuration adapted from the pasted implementation

`lift/lift_env_cfg.py` now follows the supplied reference parent implementation with local `.mdp` imports, reusable scene placeholders, aliased manager terms, a reward curriculum, and its simulation settings. `joint_pos_env_cfg.py` defines `So101GtLiftCubeEnvCfg` and `So101GtLiftCubeEnvCfg_PLAY` and continues to supply the own-USD SO-101, verified TCP, five joint-position actions, binary gripper and shared DexCube. Gym IDs and PPO configurations are unchanged.

## Ground placement

There is no table. Ground Z is 0. The 30 mm cube center is `(0.2, 0, 0.015)`, placing its bottom at ground Z. The robot root remains at the independently verified ground height `0.0024002582691891317` m; its collision geometry's root-relative minimum Z is `-0.0024002582691891317` m. The robot is fixed by `fix_root_link=True`, rather than by a joint to the movable cube. Robot and cube therefore share a support plane, not an origin height. Saved joint positions, initial velocities, actuator gains and USD contents are unchanged.

## Reference settings adopted

- Approach reward: weight 1, tanh scale 0.05.
- Lift reward: weight 15, center height above 0.025 m (10 mm above resting center).
- Coarse goal tracking: weight 16, scale 0.3, gated above 0.025 m.
- Fine goal tracking: weight 5, scale 0.05, same gate.
- Action-rate and joint-velocity penalties: initial weights -0.0001; reference curriculum changes each to -0.1 after step 10000.
- Simulation timestep 0.01 s, decimation 2 (50 Hz control), episode 5 s; reference PhysX capacities, bounce threshold 0.01 and friction correlation distance 0.00625.
- Pose-command resampling every 5 s, reference full 7-value pose-command observation, reference observation-group initialization and light settings.

## Adaptations retained

The observation also retains the required ground-truth TCP position: 6 joint positions + 6 velocities + 3 cube coordinates + 3 TCP coordinates + 7 pose-command values + 6 actions = **31 policy observations**, with **6 actions**. This changes the former 27-value observation layout; old checkpoints are not compatible with this layout.

The USD-specific subclass retains the tested fixed goal 70 mm above cube reset center, rather than the reference's unrelated randomized target region. TCP-aware command metrics and sensor markers remain; command debug visualization is disabled to avoid redundant marker assets. Reset remains deterministic, restoring saved robot/cube states and clearing action targets, instead of enabling the reference cube randomization. The tested elevated/settled success termination is retained alongside timeout and reference `object_dropping`. Default parallel environment count remains 1 rather than 4096. PLAY disables policy corruption as before. These changes preserve the earlier robot/TCP/reset constraints while adopting the supplied implementation.

## Validation

`docs/validation/so101_gt_lift_pasted_cfg.json` records a successful single-instance CPU run of 180 physics steps with configuration/import/registration validation, 31 policy observations, 6 actions, finite observations/rewards/state, deterministic saved-pose resets, grounded fixed base, TCP tracking, shared cube collision geometry, marker callback and gripper operation. The initial diagnostic failed on the old termination name `object_dropped`; the diagnostic was updated to the reference name `object_dropping` and rerun successfully.

Known warnings: the robot's existing unresolved gripper-frame visual reference, and a Fabric frame-marker point-instancer prototype warning. No native articulation/joint/physics errors were captured. A successful physical grasp/lift is not demonstrated. Lift PPO still has the existing missing `obs_groups`; PPO settings were not changed and training was not started. Diagnostic exit still uses `--skip-kit-cleanup` for the previously observed native Kit teardown issue.

```bash
conda activate env_isaaclab
source /home/intrinsic/isaacsim/setup_conda_env.sh
python tools/test_so101_gt.py --task SO101-GT-Lift-v0 --headless --device cpu --steps 180 --skip-kit-cleanup --report /tmp/so101_lift_test.json
```

The Lift Gym registration targets `joint_pos_env_cfg:So101GtLiftCubeEnvCfg`. Legacy `JointPosEnvCfg` and `JointPosEnvCfg_PLAY` names remain aliases for compatibility. Imports follow the supplied style: local `mdp`, explicit `FrameTransformerCfg`/`OffsetCfg`, `RigidBodyPropertiesCfg` and `UsdFileCfg`; no external robot package is imported. This naming/import-only change preserves environment behavior.

Editor import compatibility: `.vscode/settings.json` now selects the existing env_isaaclab interpreter by default and includes project, local Isaac Lab and installed Isaac Sim analysis roots (following the installed simulator settings). Lift MDP explicitly re-exports the action and pose-command configuration classes used by its parent/subclass. If VS Code already selected a different interpreter, select env_isaaclab manually and reload the window. Runtime imports/configuration checks pass; editor highlighting itself cannot be verified from this session.

Final compatibility audit: all ten Lift modules passed syntax checks and the registered renamed configuration completed 180 single-instance CPU physics steps. Additional diagnostic checks cover `object_reached_goal`, exact-goal reward, the exported combined reach/lift reward, and both curriculum terms at step 10000 and 10001 (the installed API uses a strict greater-than threshold); test counter and reward weights are restored before physics stepping. Report: `docs/validation/so101_gt_lift_compatibility.json`. Only the diagnostic and this report/documentation were changed during the audit; task/PPO behavior was preserved.
