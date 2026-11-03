"""Bounded Reach PPO diagnostic: validate first, train briefly, check logs/checkpoints.

Default: 8 CPU environments, 2 PPO iterations, unchanged rollout/optimizer settings.
Only SO101-GT-Reach-v0 is supported. Native errors stop the test before further steps.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8, choices=range(2, 17))
parser.add_argument('--iterations', type=int, default=2, choices=range(1, 6))
parser.add_argument('--validate-only', action='store_true')
parser.add_argument('--disable-physics-replication', action='store_true', help='Use independently spawned USD clones for this diagnostic run only')
parser.add_argument('--skip-kit-cleanup', action='store_true', help='Flush results and exit without known-problematic native Kit teardown')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

import carb
import carb.logging

native_errors = []
native_warnings = []
def collect_native(source, level, filename, line, message):
    entry = {'source': source, 'message': message}
    if level >= carb.logging.LEVEL_ERROR:
        native_errors.append(entry)
    elif level == carb.logging.LEVEL_WARN:
        native_warnings.append(entry)

logging_interface = carb.logging.acquire_logging()
logger_handle = logging_interface.add_logger(collect_native)
launcher = None
vec_env = None
runner = None
status = 1
report = {'task': 'SO101-GT-Reach-v0', 'native_errors': native_errors, 'native_warnings': native_warnings}
log_dir = None
phase = 'startup'
import faulthandler
import threading
# Keep even native scene-construction stalls bounded, without starting long work.
def timeout_test():
    report.update(exit_status=124, failure='90-second diagnostic deadline exceeded', phase=phase)
    path = log_dir / 'smoke_report.json' if log_dir else Path('/tmp/so101_reach_ppo_preflight.json')
    path.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('SMOKE TIMEOUT:', phase, file=sys.stderr, flush=True)
    os._exit(124)
watchdog = threading.Timer(90.0, timeout_test)
watchdog.daemon = True
watchdog.start()
faulthandler.dump_traceback_later(45.0)

def check_native():
    if native_errors:
        raise RuntimeError('Native runtime error; smoke test stopped: ' + json.dumps(native_errors))

try:
    launcher = AppLauncher(args)
    check_native()
    import gymnasium as gym
    import torch
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils import resolve_obs_groups
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from mobile_manipulation.robots.so101_gt_cfg import SO101_GT_USD_PATH
    from mobile_manipulation.tasks.manager_based.common.ppo_cfg import ResearchPPORunnerCfg
    import mobile_manipulation.tasks

    version = importlib.metadata.version('rsl-rl-lib')
    digest = hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest()
    env_cfg = parse_env_cfg(report['task'], device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(report['task'], 'rsl_rl_cfg_entry_point')
    if args.disable_physics_replication:
        env_cfg.scene.replicate_physics = False
    env_cfg.validate()
    agent_cfg.validate()
    baseline = ResearchPPORunnerCfg()
    assert agent_cfg.policy.to_dict() == baseline.policy.to_dict()
    assert agent_cfg.algorithm.to_dict() == baseline.algorithm.to_dict()
    for field in ('num_steps_per_env', 'save_interval', 'max_iterations'):
        assert getattr(agent_cfg, field) == getattr(baseline, field)
    mapping = {'policy': ['policy'], 'critic': ['policy']}
    assert agent_cfg.obs_groups == mapping
    synthetic_obs = TensorDict({'policy': torch.zeros(args.num_envs, 23)}, batch_size=[args.num_envs])
    assert resolve_obs_groups(synthetic_obs, agent_cfg.obs_groups.copy(), ['critic']) == mapping
    report.update(isaaclab_version=Path('/home/intrinsic/IsaacLab/VERSION').read_text().strip(),
                  rsl_rl_version=version, obs_groups=mapping, configuration='passed',
                  num_envs=args.num_envs, policy_dim=23, critic_dim=23, action_dim=5,
                  hyperparameters_unchanged=True, usd_sha256=digest)
    print('PREFLIGHT PASS:', json.dumps(report, default=str), flush=True)
    if args.validate_only:
        status = 0
    else:
        # Runtime-only bounds and device override; source PPO hyperparameters stay intact.
        agent_cfg.device = args.device
        agent_cfg.max_iterations = args.iterations
        agent_cfg.run_name = 'ppo_smoke'
        assert agent_cfg.logger == 'tensorboard'
        env_cfg.seed = agent_cfg.seed
        torch.manual_seed(agent_cfg.seed)
        timestamp = datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d_%H-%M-%S_%f')
        log_dir = Path('logs/rsl_rl') / agent_cfg.experiment_name / (timestamp + '_ppo_smoke')
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True)
        env_cfg.log_dir = str(log_dir)
        dump_yaml(str(log_dir / 'params/env.yaml'), env_cfg)
        dump_yaml(str(log_dir / 'params/agent.yaml'), agent_cfg)
        report['log_dir'] = str(log_dir)
        print('SMOKE LOG DIRECTORY:', log_dir, flush=True)

        phase = 'scene construction'
        report['physics_replication'] = env_cfg.scene.replicate_physics
        env = gym.make(report['task'], cfg=env_cfg)
        check_native()
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base = vec_env.unwrapped
        robot = base.scene['robot']
        expected_joints = torch.tensor([env_cfg.scene.robot.init_state.joint_pos[n] for n in robot.joint_names],
                                      device=base.device).expand(args.num_envs, -1)
        expected_roots = base.scene.env_origins + torch.tensor(env_cfg.scene.robot.init_state.pos, device=base.device)
        def check_reset():
            assert torch.allclose(robot.data.joint_pos, expected_joints, atol=1e-5)
            assert torch.allclose(robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel), atol=1e-6)
            assert torch.allclose(robot.data.root_pos_w, expected_roots, atol=1e-5)
        for _ in range(2):
            vec_env.reset()
            check_reset()
        obs = vec_env.get_observations()
        assert tuple(obs['policy'].shape) == (args.num_envs, 23)
        assert vec_env.num_actions == 5
        assert resolve_obs_groups(obs, agent_cfg.obs_groups.copy(), ['critic']) == mapping
        assert torch.isfinite(obs['policy']).all()
        check_native()
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
        assert runner.alg.policy.get_actor_obs(obs).shape[-1] == 23
        assert runner.alg.policy.get_critic_obs(obs).shape[-1] == 23
        print('ACTUAL GROUP/DIMENSION PASS: actor=23, critic=23, actions=5', flush=True)

        stats = {'transitions': 0, 'automatic_resets': 0, 'timeout_resets': 0,
                 'reward_min': float('inf'), 'reward_max': -float('inf'), 'action_abs_max': 0.0, 'losses': []}
        step_impl = vec_env.step
        def checked_step(actions):
            check_native()
            assert actions.shape == (args.num_envs, 5)
            assert torch.isfinite(actions).all(), 'Non-finite action'
            output = step_impl(actions)
            check_native()
            current_obs, rewards, dones, extras = output
            assert torch.isfinite(current_obs['policy']).all(), 'Non-finite observation'
            assert torch.isfinite(rewards).all(), 'Non-finite reward'
            assert torch.isfinite(robot.data.root_state_w).all(), 'Non-finite root state'
            assert torch.isfinite(robot.data.joint_pos).all(), 'Non-finite joint state'
            assert torch.allclose(robot.data.root_pos_w, expected_roots, atol=1e-4), 'Fixed-base root drift'
            reset_ids = dones.bool()
            if reset_ids.any():
                assert torch.allclose(robot.data.joint_pos[reset_ids], expected_joints[reset_ids], atol=1e-5)
                assert torch.allclose(robot.data.joint_vel[reset_ids], torch.zeros_like(robot.data.joint_vel[reset_ids]), atol=1e-6)
            stats['transitions'] += args.num_envs
            stats['automatic_resets'] += int(dones.sum())
            stats['timeout_resets'] += int(extras['time_outs'].sum())
            stats['reward_min'] = min(stats['reward_min'], rewards.min().item())
            stats['reward_max'] = max(stats['reward_max'], rewards.max().item())
            stats['action_abs_max'] = max(stats['action_abs_max'], actions.abs().max().item())
            return output
        vec_env.step = checked_step
        update_impl = runner.alg.update
        def checked_update():
            check_native()
            losses = update_impl()
            assert all(math.isfinite(float(value)) for value in losses.values()), 'Non-finite loss'
            for parameter in runner.alg.policy.parameters():
                assert torch.isfinite(parameter).all(), 'Non-finite policy parameter'
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), 'Non-finite gradient'
            stats['losses'].append({key: float(value) for key, value in losses.items()})
            return losses
        runner.alg.update = checked_update
        # Exercise time-limit automatic reset during rollout without changing episode duration.
        base.episode_length_buf[:2] = base.max_episode_length - 1
        phase = 'PPO rollout/update'
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        check_native()
        assert stats['transitions'] == args.num_envs * agent_cfg.num_steps_per_env * args.iterations
        assert stats['timeout_resets'] >= 2
        assert len(stats['losses']) == args.iterations
        runner.writer.flush()
        runner.writer.close()
        checkpoints = sorted(log_dir.glob('model_*.pt'))
        assert checkpoints, 'No checkpoint created'
        for checkpoint in checkpoints:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['model_state_dict']
            assert all(torch.isfinite(tensor).all() for tensor in state['model_state_dict'].values())
        accumulator = EventAccumulator(str(log_dir)).Reload()
        scalar_tags = accumulator.Tags()['scalars']
        assert any(tag.startswith('Loss/') for tag in scalar_tags), 'No loss TensorBoard scalars'
        assert 'Train/mean_reward' in scalar_tags, 'No reward TensorBoard scalars'
        for tag in scalar_tags:
            assert all(math.isfinite(event.value) for event in accumulator.Scalars(tag)), 'Non-finite logged scalar'
        assert hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest() == digest
        vec_env.reset()
        check_reset()
        vec_env.close()
        vec_env = None
        check_native()
        report.update(runtime='passed', iterations=args.iterations, stats=stats,
                      checkpoints=[str(path) for path in checkpoints], tensorboard_tags=scalar_tags,
                      event_files=[str(path) for path in log_dir.glob('events.out.tfevents.*')])
        status = 0
        print('PPO SMOKE PASS:', json.dumps(report, indent=2), flush=True)
except Exception as exc:
    report['failure'] = repr(exc)
    import traceback
    traceback.print_exc()
finally:
    watchdog.cancel()
    faulthandler.cancel_dump_traceback_later()
    report['exit_status'] = status
    if log_dir:
        (log_dir / 'smoke_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    else:
        Path('/tmp/so101_reach_ppo_preflight.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    logging_interface.remove_logger(logger_handle)
    if args.skip_kit_cleanup:
        os._exit(status)
    if launcher is not None:
        launcher.app.close(wait_for_replicator=False)
    sys.exit(status)
