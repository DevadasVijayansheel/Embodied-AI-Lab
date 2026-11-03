"""Single-instance SO-101 GT configuration/reset/step test. Never trains a policy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=['SO101-GT-Reach-v0', 'SO101-GT-Lift-v0'], required=True)
parser.add_argument('--steps', type=int, default=180)
parser.add_argument('--config-only', action='store_true')
parser.add_argument('--grasp-lift-test', action='store_true', help='Lift only: bounded open/approach/close/raise physical trial without object teleportation')
parser.add_argument('--skip-procedural-materials', action='store_true',
                    help='Physics diagnostic only: skip cube visual material creation to isolate the native shader-registry stall')
parser.add_argument('--report', type=Path)
parser.add_argument('--skip-kit-cleanup', action='store_true', help='Diagnostic process exit after reports are flushed; avoids the known native Kit teardown crash')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.grasp_lift_test and args.task != "SO101-GT-Lift-v0":
    parser.error("--grasp-lift-test requires SO101-GT-Lift-v0")
import faulthandler
faulthandler.dump_traceback_later(30.0)
import carb.logging
native_errors = []
logging_interface = carb.logging.acquire_logging()
def native_logger(source, level, filename, line, message):
    if level >= carb.logging.LEVEL_ERROR:
        native_errors.append({"source": source, "message": message})
logger_handle = logging_interface.add_logger(native_logger)
launcher = AppLauncher(args)
status = 1
try:
    import gymnasium as gym
    import torch
    import mobile_manipulation.tasks
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from mobile_manipulation.robots.so101_gt_cfg import so101_gt_cfg, SO101_GT_USD_PATH
    from mobile_manipulation.robots.so101_gt_cfg import ARM_JOINTS, BASE_COLLISION_MIN_Z
    from mobile_manipulation.tasks.manager_based.so101_gt.lift.lift_env_cfg import GROUND_HEIGHT, CUBE_SIZE, DROP_HEIGHT

    digest = hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 42
    if args.skip_procedural_materials:
        if cfg.skill == 'lift':
            cfg.scene.object.spawn.visual_material = None
    cfg.validate()
    agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    from dataclasses import MISSING
    # Preserve the user's existing PPO config, reporting its pre-existing missing mapping.
    ppo_missing = []
    if isinstance(agent.obs_groups, type(MISSING)):
        ppo_missing.append('obs_groups')
        print('PPO WARNING: existing configuration has MISSING obs_groups; runner validation/training is blocked.', flush=True)
    else:
        agent.validate()
    assert cfg.scene.robot.init_state == so101_gt_cfg.init_state
    assert cfg.scene.robot.actuators == so101_gt_cfg.actuators
    result = {'task': args.task, 'configuration': 'passed', 'registration': gym.spec(args.task).kwargs,
              'experiment_name': agent.experiment_name, 'ppo_missing': ppo_missing, 'steps': 0,
              'procedural_visual_materials_skipped': args.skip_procedural_materials}
    print('CONFIG PASS:', args.task, flush=True)
    if not args.config_only:
        assert not native_errors, native_errors
        env = gym.make(args.task, cfg=cfg)
        assert not native_errors, native_errors
        base = env.unwrapped
        if cfg.skill == 'lift':
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = base.scene.stage
            obj_prim = stage.GetPrimAtPath('/World/envs/env_0/Object')
            assert obj_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            collision = stage.GetPrimAtPath('/World/envs/env_0/Object/collisions/collisions')
            assert collision.HasAPI(UsdPhysics.CollisionAPI)
            cube = UsdGeom.Cube(collision)
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(collision)
            edge = cube.GetSizeAttr().Get() * transform.ExtractRotationMatrix().GetRow(0).GetLength()
            assert abs(edge - CUBE_SIZE) < 1e-6
            assert stage.GetPrimAtPath('/World/envs/env_0/Object/visuals/visuals').IsValid()
            sensor = base.scene['ee_frame']
            sensor._debug_vis_callback(None)
            marker = stage.GetPrimAtPath(sensor.cfg.visualizer_cfg.prim_path)
            assert marker.IsValid()
            assert len(sensor.cfg.visualizer_cfg.markers) == 2
            result['tcp_marker_verified'] = True
            result['shared_cube_verified'] = True
            result['cube_collision_edge_m'] = edge
            result['cube_authored_mass_kg'] = UsdPhysics.MassAPI(obj_prim).GetMassAttr().Get()
        with torch.inference_mode():
            obs, info = env.reset()
            robot = base.scene['robot']
            expected = torch.tensor([cfg.scene.robot.init_state.joint_pos[n] for n in robot.joint_names], device=base.device)
            print('BODY NAMES', robot.body_names, flush=True)
            print('JOINT NAMES', robot.joint_names, flush=True)
            print('RESET ROOT', robot.data.root_pos_w.cpu().tolist(), 'EXPECTED', cfg.scene.robot.init_state.pos, flush=True)
            print('RESET JOINTS', robot.data.joint_pos.cpu().tolist(), flush=True)
            assert torch.allclose(robot.data.joint_pos[0], expected, atol=1e-5)
            expected_root = torch.tensor(cfg.scene.robot.init_state.pos, device=base.device)
            assert torch.allclose(robot.data.root_pos_w[0], expected_root, atol=1e-5)
            assert torch.allclose(robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel), atol=1e-6)
            # Verify predicates independently of a trained behavior.
            from isaaclab.envs.mdp import time_out
            saved_steps = base.episode_length_buf.clone()
            base.episode_length_buf[:] = base.max_episode_length
            assert time_out(base).all()
            base.episode_length_buf[:] = saved_steps
            if cfg.skill == 'lift':
                from mobile_manipulation.tasks.manager_based.so101_gt.lift import mdp as lift_mdp
                from mobile_manipulation.tasks.manager_based.so101_gt.lift.lift_env_cfg import LIFT_HEIGHT
                from isaaclab.utils.math import combine_frame_transforms
                from isaaclab.managers import SceneEntityCfg
                obj = base.scene['object']
                goal, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                    base.command_manager.get_command('object_pose')[:, :3])
                pose = obj.data.root_pose_w.clone()
                pose[:, :3] = goal
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
                assert lift_mdp.object_is_lifted(base, minimal_height=LIFT_HEIGHT).all()
                assert lift_mdp.lifted_at_goal(base, minimum_height=LIFT_HEIGHT).all()
                assert lift_mdp.object_reached_goal(base).all()
                assert torch.allclose(lift_mdp.object_goal_distance(base, std=0.05,
                    minimal_height=LIFT_HEIGHT, command_name='object_pose'), torch.ones(1, device=base.device))
                combined = lift_mdp.object_ee_distance_and_lifted(base, std=0.05, minimal_height=LIFT_HEIGHT)
                assert torch.allclose(combined, lift_mdp.object_ee_distance(base, std=0.05)
                                      * lift_mdp.object_is_lifted(base, minimal_height=LIFT_HEIGHT))
                # Exercise the curriculum boundary without advancing simulation/training.
                import copy
                saved_counter = base.common_step_counter
                for name in ('action_rate', 'joint_vel'):
                    original = copy.deepcopy(base.reward_manager.get_term_cfg(name))
                    curriculum_cfg = getattr(cfg.curriculum, name)
                    curriculum = lift_mdp.modify_reward_weight(curriculum_cfg, base)
                    try:
                        base.common_step_counter = 10000
                        assert curriculum(base, [0], **curriculum_cfg.params) == original.weight
                        base.common_step_counter = 10001
                        assert curriculum(base, [0], **curriculum_cfg.params) == -0.1
                    finally:
                        base.common_step_counter = saved_counter
                        base.reward_manager.set_term_cfg(name, original)
                result['additional_mdp_and_curriculum_checks'] = 'passed'

                pose[:, 2] = GROUND_HEIGHT - 0.06
                obj.write_root_pose_to_sim(pose)
                assert lift_mdp.root_height_below_minimum(base, minimum_height=DROP_HEIGHT,
                    asset_cfg=SceneEntityCfg('object')).all()
                obs, info = env.reset()
                print('Lift goal/lift/drop predicates: PASS (injected test states, not a grasp demonstration)', flush=True)
            arm = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
            assert len(arm) == 5
            from mobile_manipulation.robots.so101_gt_cfg import GRASP_OFFSET, GRASP_ROTATION
            import importlib
            task_mdp = importlib.import_module(
                f'mobile_manipulation.tasks.manager_based.so101_gt.{cfg.skill}.mdp')
            end_effector_position = task_mdp.end_effector_position
            def position_command_error(env, command_name):
                return torch.linalg.vector_norm(end_effector_position(env) -
                    env.command_manager.get_command(command_name)[:, :3], dim=-1)
            from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, quat_error_magnitude
            gripper_id = robot.find_bodies('gripper_link')[0][0]
            tcp_id = robot.find_bodies('gripper_frame_link')[0][0]
            tcp_errors = []
            base_ground_errors = []
            gripper_positions = []
            gripper_joint = robot.find_joints('gripper')[0][0]
            offset_pos = torch.tensor(GRASP_OFFSET, device=base.device).expand(1, -1)
            offset_rot = torch.tensor(GRASP_ROTATION, device=base.device).expand(1, -1)
            def check(observation):
                assert not native_errors, native_errors
                assert torch.isfinite(observation['policy']).all()
                assert observation['policy'].shape == (1, 23 if cfg.skill == 'reach' else 31)
                assert torch.isfinite(robot.data.root_state_w).all()
                assert torch.allclose(robot.data.root_pos_w[0], expected_root, atol=1e-5)
                ground_error = abs(robot.data.root_pos_w[0, 2].item() + BASE_COLLISION_MIN_Z - GROUND_HEIGHT)
                base_ground_errors.append(ground_error)
                assert ground_error < 1e-5
                assert not base.scene.stage.GetPrimAtPath('/World/envs/env_0/Table').IsValid()
                assert torch.isfinite(robot.data.joint_pos).all()
                sensor = base.scene['ee_frame'].data
                tcp_pos, tcp_rot = combine_frame_transforms(
                    robot.data.body_pos_w[:, gripper_id], robot.data.body_quat_w[:, gripper_id], offset_pos, offset_rot)
                assert torch.allclose(sensor.target_pos_w[:, 0], tcp_pos, atol=1e-6)
                assert quat_error_magnitude(sensor.target_quat_w[:, 0], tcp_rot).max() < 1e-3
                error = torch.linalg.vector_norm(tcp_pos - robot.data.body_pos_w[:, tcp_id], dim=-1).max().item()
                tcp_errors.append(error)
                assert error < 1e-4, 'Sensor TCP differs from actual fixed frame body'
                assert quat_error_magnitude(sensor.target_quat_w[:, 0], robot.data.body_quat_w[:, tcp_id]).max() < 1e-3
                root_tcp, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, tcp_pos)
                assert torch.allclose(end_effector_position(base), root_tcp, atol=1e-6)
                # Verify the configured policy TCP slot, not just the helper function.
                tcp_slice = slice(15, 18)  # joints (12), target/object position (3), TCP (3)
                assert torch.allclose(observation['policy'][:, tcp_slice], root_tcp, atol=1e-6)
                if cfg.skill == 'reach':
                    target = base.command_manager.get_command('ee_pose')[:, :3]
                    distance = torch.linalg.vector_norm(root_tcp - target, dim=-1)
                    assert torch.allclose(position_command_error(base, 'ee_pose'), distance, atol=1e-6)
                    reward_cfg = base.reward_manager.get_term_cfg('reaching')
                    assert torch.allclose(reward_cfg.func(base, **reward_cfg.params), 1 - torch.tanh(distance / 0.04), atol=1e-6)
                else:
                    distance = torch.linalg.vector_norm(base.scene['object'].data.root_pos_w - tcp_pos, dim=-1)
                    reward_cfg = base.reward_manager.get_term_cfg('reaching_object')
                    assert torch.allclose(reward_cfg.func(base, **reward_cfg.params), 1 - torch.tanh(distance / reward_cfg.params["std"]), atol=1e-6)
            check(obs)
            action_count = 5 if cfg.skill == 'reach' else 6
            assert base.action_manager.total_action_dim == action_count
            successes = 0
            root_history = []
            heights = []
            best_reach_error = float('inf')
            approach_errors = []
            physical_drop_count = 0
            trial_target = base.scene['object'].data.root_pos_w.clone() if args.grasp_lift_test else None
            close_step = 100 if args.grasp_lift_test else args.steps // 2
            from isaaclab.utils.math import skew_symmetric_matrix
            def feedback_action(target_w):
                sensor = base.scene['ee_frame'].data
                error = target_w - sensor.target_pos_w[:, 0]
                jacobian = robot.root_physx_view.get_jacobians()[:, gripper_id - 1, :, arm]
                offset_w = sensor.target_pos_w[:, 0] - robot.data.body_pos_w[:, gripper_id]
                tcp_jacobian = jacobian[:, :3] - skew_symmetric_matrix(offset_w) @ jacobian[:, 3:]
                regularizer = 0.0001 * torch.eye(3, device=base.device).expand(1, -1, -1)
                delta = tcp_jacobian.transpose(1, 2) @ torch.linalg.solve(
                    tcp_jacobian @ tcp_jacobian.transpose(1, 2) + regularizer, error.unsqueeze(-1))
                previous_targets = base.action_manager.get_term('arm_action').processed_actions
                desired = previous_targets + 0.2 * delta.squeeze(-1).clamp(-0.1, 0.1)
                return (desired - expected[arm]) / 0.5
            for step in range(args.steps):
                action = torch.zeros((1, action_count), device=base.device)
                if cfg.skill == 'reach':
                    # Known reachable target: shoulder_pan +12 degrees, other saved joints held.
                    action[:, 0] = torch.pi / 15 / 0.5
                    # Retain the +12-degree seed, then compensate physics deflection
                    # at the TCP using a damped Jacobian controller (test-only).
                    if step >= 30:
                        target_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                            base.command_manager.get_command('ee_pose')[:, :3])
                        action = feedback_action(target_w)
                elif step < close_step:
                    action[:, -1] = 1.0  # non-negative opens in installed Isaac Lab
                else:
                    action[:, -1] = -1.0  # negative closes; zero also opens
                if args.grasp_lift_test and step >= 30:
                    target_w = trial_target.clone()
                    if step >= 160:
                        target_w[:, 2] += 0.07 * min((step - 160) / 60, 1.0)
                    action[:, :5] = feedback_action(target_w)
                assert torch.isfinite(action).all()
                obs, reward, terminated, truncated, info = env.step(action)
                check(obs)
                assert torch.isfinite(reward).all()
                command_name = 'ee_pose' if cfg.skill == 'reach' else 'object_pose'
                command = base.command_manager.get_term(command_name)
                assert torch.allclose(command.metrics['position_error'],
                    position_command_error(base, command_name), atol=1e-6)
                if cfg.skill == 'reach':
                    # Reward manager retains the pre-reset distance for successful steps.
                    terms = dict(base.reward_manager.get_active_iterable_terms(0))
                    step_distance = -terms['distance'][0]  # unchanged weight = -1
                    best_reach_error = min(best_reach_error, step_distance)
                    if base.termination_manager.get_term('success')[0]:
                        assert step_distance < 0.008
                root_history.append(robot.data.root_pos_w[0].cpu().tolist())
                if cfg.skill == 'lift':
                    obj = base.scene['object']
                    assert torch.isfinite(obj.data.root_state_w).all()
                    heights.append(obj.data.root_pos_w[0, 2].item())
                    approach_errors.append(torch.linalg.vector_norm(obj.data.root_pos_w - base.scene['ee_frame'].data.target_pos_w[:, 0], dim=-1).item())
                    physical_drop_count += int(base.termination_manager.get_term('object_dropping')[0])
                    gripper_positions.append(robot.data.joint_pos[0, gripper_joint].item())
                    gripper_action = base.action_manager.get_term('gripper_action')
                    expected_target = 0.5 if step < close_step else 0.0
                    assert torch.allclose(gripper_action.processed_actions, torch.full_like(gripper_action.processed_actions, expected_target))
                    assert torch.isfinite(base.scene['ee_frame'].data.target_pos_w).all()
                successes += int(base.termination_manager.get_term('success')[0])
            print('FINAL ROOT', robot.data.root_pos_w.cpu().tolist(), flush=True)
            if cfg.skill == 'reach':
                print('FINAL TCP REACH ERROR', position_command_error(base, 'ee_pose').cpu().tolist(), flush=True)
            if cfg.skill == 'reach' and args.steps >= 120:
                assert successes > 0, f'Scripted TCP target was not achieved; best post-step error {best_reach_error}'
            if gripper_positions and args.steps >= 120 and not args.grasp_lift_test:
                assert max(gripper_positions) > 0.1, 'Gripper did not physically open'
                assert gripper_positions[-1] < max(gripper_positions) - 0.05, 'Gripper did not physically close'
            if heights:
                if not args.grasp_lift_test:
                    assert abs(heights[-1] - (GROUND_HEIGHT + CUBE_SIZE / 2)) < 0.001
                    assert physical_drop_count == 0
                print('CUBE HEIGHT RANGE', min(heights), max(heights), 'GROUND HEIGHT', GROUND_HEIGHT, flush=True)
            result.update(runtime='passed', steps=args.steps, policy_shape=list(obs['policy'].shape),
                          action_dim=action_count, successes=successes, root_positions=root_history,
                          cube_heights=heights, tcp_max_error_m=max(tcp_errors),
                          gripper_positions=gripper_positions, native_errors=native_errors,
                          ground_height=GROUND_HEIGHT, robot_root_position=cfg.scene.robot.init_state.pos,
                          base_ground_max_error_m=max(base_ground_errors))
            if cfg.skill == 'reach':
                result['best_reach_error_m'] = best_reach_error
                result['scripted_controller'] = 'shoulder_pan +12 degree seed, then damped TCP Jacobian feedback after 30 steps'
            if args.grasp_lift_test:
                result['physical_trial'] = {'controller': 'open, TCP approach, close, raise 7 cm',
                    'object_teleported_during_trial': False, 'drop_count': physical_drop_count,
                    'cube_max_height_m': max(heights), 'cube_final_height_m': heights[-1],
                    'minimum_tcp_cube_distance_m': min(approach_errors),
                    'success_terminations': successes, 'grasp_demonstrated': False}
            # A reset must reproduce the exact Inspector joint state each time.
            for _ in range(2):
                env.reset()
                assert torch.allclose(robot.data.joint_pos[0], expected, atol=1e-5)
                assert torch.allclose(robot.data.root_pos_w[0], expected_root, atol=1e-5)
                assert torch.allclose(robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel), atol=1e-6)
        env.close()
        assert not native_errors, native_errors
        print('RUNTIME PASS:', args.task, flush=True)
    assert hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest() == digest
    result['usd_sha256'] = digest
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
    status = 0
except Exception as exc:
    import traceback
    traceback.print_exc()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'task': args.task, 'runtime': 'failed',
            'failure': repr(exc), 'native_errors': native_errors,
            'procedural_visual_materials_skipped': args.skip_procedural_materials}, indent=2) + '\n')
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    faulthandler.cancel_dump_traceback_later()
    logging_interface.remove_logger(logger_handle)
    # Normal Kit cleanup has crashed on this installation. Do not mask a failed check.
    if args.skip_kit_cleanup:
        import os
        os._exit(status)
    launcher.app.close(wait_for_replicator=False)
    sys.exit(status)
