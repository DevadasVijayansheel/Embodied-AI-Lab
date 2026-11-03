"""TCP distance/tanh reach kernels and inspected official action regularizers.

The official body-origin reach kernel cannot represent the fixed TCP offset.
These kernels retain the same L2 distance and 1-tanh(distance/std) reward scales.
Official action_rate_l2 penalizes change in raw actions; action_l2 their squared norm.
"""
import torch
from isaaclab.envs.mdp import action_rate_l2, action_l2, joint_vel_l2
from isaaclab.managers import SceneEntityCfg
from .observations import end_effector_position


def position_command_error(env, command_name, robot_cfg=SceneEntityCfg("robot"),
                           ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Official reach distance kernel evaluated at the sensor TCP, not a body origin."""
    target = env.command_manager.get_command(command_name)[:, :3]
    return torch.linalg.vector_norm(end_effector_position(env, robot_cfg, ee_frame_cfg) - target, dim=-1)


def position_command_error_tanh(env, command_name, std, robot_cfg=SceneEntityCfg("robot"),
                                ee_frame_cfg=SceneEntityCfg("ee_frame")):
    return 1.0 - torch.tanh(position_command_error(env, command_name, robot_cfg, ee_frame_cfg) / std)


__all__ = ["action_rate_l2", "action_l2", "joint_vel_l2", "position_command_error", "position_command_error_tanh"]
