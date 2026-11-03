# SO-101 GT agent and PPO input validation

## Lift observation history

For N environments, policy terms in declaration order are `joint_pos` (N,6), `joint_vel` (N,6), `object_position` (N,3), `end_effector_position` (N,3), `target_object_position` (N,7) and `actions` (N,6). Concatenation is (N,31).

The previous local layout contained a 3-value XYZ target, making 27 values. The current full pose command contains XYZ plus WXYZ quaternion, adding four. The reference SO-ARM101 Lift observation list omits the explicit TCP observation; with these six joints/actions its total is 28. The local additional three TCP coordinates make 31. No observation term was altered during this work.

Reference: https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/lift/lift_env_cfg.py

## PPO mapping

Lift now defines `obs_groups = {"policy": ["policy"], "critic": ["policy"]}`, matching Reach. There is one environment observation group called policy; critic is an RSL-RL input-set name selecting that same group, not a newly created environment group.

For each task the actual RSL-RL wrapper observations are passed to `resolve_obs_groups` and `OnPolicyRunner(log_dir=None)`. Actor/critic observations, actor output and value output are checked with the installed RSL-RL APIs. The runner is constructed only for inference compatibility; `learn`, backward and optimizer steps are never called. PPO hyperparameters are unchanged.

## Scripts and validation

New `scripts/so101_gt/zero_agent.py` and `random_agent.py` support both Reach and Lift through `--task`. Their shared `agent_runner.py` follows the installed official Isaac Lab `scripts/environments/zero_agent.py` and `random_agent.py` flow: AppLauncher, parse_env_cfg, gym.make, reset and tensor actions. The originals in this external project's scripts directory are untouched.

Action shape comes from the actual Gym space and is checked against the RSL wrapper; observation terms and actor/critic dimensions come from the runtime managers/tensors. No observation/action dimensions are hardcoded. Random actions follow the official uniform [-1,1) example; zero actions follow its torch.zeros example. Default run length is 100 steps. Scripts additionally support reports, finite-value/native-error checks and deterministic reset checks.

All four task/agent combinations passed 100 steps each, one environment, CPU, headless, Fabric disabled. Each had three successful explicit resets, finite observations, actions, rewards and network outputs; zero captured native errors. Reach actor/critic inputs: (1,23), actions (1,5). Lift actor/critic inputs: (1,31), actions (1,6). Reports record actual observation terms and reward extrema. No automatic episode resets occurred during these short runs; explicit resets were tested instead. No grasp success or trained policy is claimed.

Existing USD visual/Fabric and Kit teardown warnings remain separate limitations. The diagnostic `--skip-kit-cleanup` exits after flushing the report to avoid the previously observed native teardown problem. This is confined to the new scripts; neither environment/MDP nor Isaac Sim was modified. Hash checks verified the protected Lift configuration/MDP and robot configuration/USD were unchanged.

## Reproduce all four checks

```bash
cd /home/intrinsic/mobile_manipulation
conda activate env_isaaclab
source /home/intrinsic/isaacsim/setup_conda_env.sh
for task in SO101-GT-Reach-v0 SO101-GT-Lift-v0; do
  for agent in zero random; do
    python scripts/so101_gt/${agent}_agent.py --task "$task" --num_envs 1 --steps 100 --headless --device cpu --disable_fabric --skip-kit-cleanup --report "/tmp/${task}_${agent}.json"
  done
done
```

Remove `--headless` for GUI runs. Results: `docs/validation/so101_gt_agents_100_steps.json`.
