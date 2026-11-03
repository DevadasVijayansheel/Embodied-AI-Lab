# Reach PPO compatibility and bounded diagnostic

Installed versions checked: Isaac Lab 2.3.2 and RSL-RL 3.1.2, using the existing
`env_isaaclab` environment. No packages were installed or changed.

Only Reach's `agents/rsl_rl_ppo_cfg.py` was changed to add:

```python
obs_groups = {"policy": ["policy"], "critic": ["policy"]}
```

Both actor and critic consume the existing ground-truth `policy` observation group;
no separate privileged observations are required. Source PPO policy, optimizer,
rollout length, save interval and iteration defaults remain unchanged. Lift, USD,
robot gains, environment configurations and original training scripts were not modified.

Configuration instantiation and validation passed. RSL-RL's installed
`resolve_obs_groups` accepted the mapping with a synthetic observation TensorDict.
Expected dimensions are 23 actor observations, 23 critic observations and 5 actions.
These dimensions were not verified in a running parallel environment.

Two CPU smoke attempts were made, each bounded to two PPO iterations:

| Attempt | Environments | Physics replication | Result |
| --- | --- | --- | --- |
| First | 8 | Enabled | Scene creation stalled; process terminated before PPO construction |
| Second | 2 | Disabled for diagnostic only | Scene creation stalled; externally force-terminated |

The second attempt's 45-second Python traceback located the wait in
`Sdr.Registry().GetShaderNodeByIdentifierAndType`, called by Isaac Sim's
`CreateShaderFromNode` command during Isaac Lab preview-surface creation for a cuboid.
The precise cause of this native material-registry stall is unresolved. No native
error was captured before the stall. The existing unresolved
`/visuals/gripper_frame_link` USD visual-reference warning remains.

Neither attempt reached PPO rollouts: finite actions/rewards/losses, automatic resets,
parallel articulation/physics behavior, TensorBoard logging and checkpoint creation
remain unverified. Parameter YAML files and diagnostic reports were created; no
TensorBoard event files or model checkpoints were created.

Reports:

- `logs/rsl_rl/so101_gt_reach/2026-10-11_21-44-42_479105_ppo_smoke/smoke_report.json`
- `logs/rsl_rl/so101_gt_reach/2026-10-11_21-49-06_345165_ppo_smoke/smoke_report.json`
- `/tmp/so101_reach_ppo_preflight.json` (configuration-only validation)

The new diagnostic checks dimensions, finite tensors and losses, fixed-base state,
saved-pose resets, time-limit resets, native errors, TensorBoard scalars, checkpoint
contents and unchanged USD hash if scene creation completes. It has a 90-second Python watchdog and runtime-only iteration bounds. The watchdog
could not interrupt the native stall, so use the external timeout below. `--skip-kit-cleanup` flushes reports and
exits without the already problematic native Kit teardown; it does not repair Kit.

Reproduce configuration validation:

```bash
conda activate env_isaaclab
cd /home/intrinsic/mobile_manipulation
source /home/intrinsic/isaacsim/setup_conda_env.sh
export PYTHONPATH="$PWD/source/mobile_manipulation:$PYTHONPATH"
python -u tools/smoke_train_so101_reach.py --validate-only --headless --device cpu --skip-kit-cleanup
```

After resolving the material-registry stall, reproduce the smallest bounded smoke test:

```bash
timeout --signal=KILL 90s python -u tools/smoke_train_so101_reach.py --num_envs 2 --iterations 2 --disable-physics-replication --headless --device cpu --skip-kit-cleanup
```

Do not treat the configuration-only pass as runtime or training readiness.
