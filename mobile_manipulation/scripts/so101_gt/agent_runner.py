# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared official-example launch/reset/step flow; validates PPO without learning."""
import argparse
import json
import os
from pathlib import Path


def run(agent_kind):
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['SO101-GT-Reach-v0', 'SO101-GT-Lift-v0'])
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--disable_fabric', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--skip-kit-cleanup', action='store_true', help='Diagnostic exit for the known native Kit teardown issue')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error('steps and num_envs must be positive')
    launcher = AppLauncher(args)
    import carb.logging
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import mobile_manipulation.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils import resolve_obs_groups
    native_errors = []
    logging_interface = carb.logging.acquire_logging()
    def logger(source, level, filename, line, message):
        if level >= carb.logging.LEVEL_ERROR:
            native_errors.append({'source': source, 'message': message})
    logger_handle = logging_interface.add_logger(logger)
    report = {'task': args.task, 'agent': agent_kind, 'requested_steps': args.steps,
              'steps': 0, 'device': args.device, 'fabric': not args.disable_fabric}
    status = 1
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            use_fabric=not args.disable_fabric)
        cfg.validate()
        agent_cfg = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
        agent_cfg.validate()
        torch.manual_seed(agent_cfg.seed)
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        def check_reset():
            vec.reset()
            robot = base.scene['robot']
            expected = torch.tensor([cfg.scene.robot.init_state.joint_pos[n] for n in robot.joint_names], device=base.device)
            assert torch.allclose(robot.data.joint_pos, expected.expand_as(robot.data.joint_pos), atol=1e-5)
            assert torch.allclose(robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel), atol=1e-6)
            root = base.scene.env_origins + torch.tensor(cfg.scene.robot.init_state.pos, device=base.device)
            assert torch.allclose(robot.data.root_pos_w, root, atol=1e-5)
        check_reset()
        obs = vec.get_observations()
        mapping = resolve_obs_groups(obs, {k: v.copy() for k, v in agent_cfg.obs_groups.items()}, ['critic'])
        runner = OnPolicyRunner(vec, agent_cfg.to_dict(), log_dir=None, device=args.device)
        policy = runner.alg.policy
        actor_input = policy.get_actor_obs(obs)
        critic_input = policy.get_critic_obs(obs)
        expected_actor = sum(obs[g].shape[-1] for g in mapping['policy'])
        expected_critic = sum(obs[g].shape[-1] for g in mapping['critic'])
        assert actor_input.shape == (args.num_envs, expected_actor)
        assert critic_input.shape == (args.num_envs, expected_critic)
        action_shape = tuple(env.action_space.shape)
        assert action_shape == (args.num_envs, vec.num_actions)
        terms = {}
        for name, dim in zip(base.observation_manager.active_terms['policy'],
                             base.observation_manager.group_obs_term_dim['policy']):
            terms[name] = [args.num_envs, *dim]
        report.update(observation_terms=terms, observation_groups={k: list(v.shape) for k, v in obs.items()},
                      obs_groups=mapping, actor_input_shape=list(actor_input.shape),
                      critic_input_shape=list(critic_input.shape), action_shape=list(action_shape))
        def check(values):
            for value in values.values():
                assert torch.isfinite(value).all(), 'Non-finite observation'
            assert not native_errors, native_errors
            robot = base.scene['robot']
            assert torch.isfinite(robot.data.root_state_w).all()
            assert torch.isfinite(robot.data.joint_pos).all()
            if cfg.skill == 'lift':
                assert torch.isfinite(base.scene['object'].data.root_state_w).all()
        reward_min, reward_max = float('inf'), -float('inf')
        reset_count = 1
        automatic_resets = 0
        with torch.inference_mode():
            assert policy.act_inference(obs).shape == action_shape
            assert torch.isfinite(policy.act_inference(obs)).all()
            assert torch.isfinite(policy.evaluate(obs)).all()
            check(obs)
            for step in range(args.steps):
                if not launcher.app.is_running():
                    raise RuntimeError('Simulator closed before requested steps completed')
                actions = (torch.zeros(action_shape, device=base.device) if agent_kind == 'zero'
                           else 2 * torch.rand(action_shape, device=base.device) - 1)
                assert torch.isfinite(actions).all()
                obs, rewards, dones, extras = vec.step(actions)
                check(obs)
                assert torch.isfinite(rewards).all(), 'Non-finite reward'
                reward_min = min(reward_min, rewards.min().item())
                reward_max = max(reward_max, rewards.max().item())
                automatic_resets += int(dones.sum().item())
                report['steps'] = step + 1
                if step + 1 == args.steps // 2:
                    check_reset()
                    reset_count += 1
            check_reset()
            reset_count += 1
        report.update(status='passed', explicit_resets=reset_count, automatic_resets=automatic_resets,
                      reward_min=reward_min, reward_max=reward_max, native_errors=native_errors,
                      training_started=False)
        status = 0
    except Exception as exc:
        report.update(status='failed', error=repr(exc), native_errors=native_errors)
        import traceback
        traceback.print_exc()
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        logging_interface.remove_logger(logger_handle)
        if args.skip_kit_cleanup:
            os._exit(status)
        if env is not None:
            env.close()
        launcher.app.close()
    raise SystemExit(status)
