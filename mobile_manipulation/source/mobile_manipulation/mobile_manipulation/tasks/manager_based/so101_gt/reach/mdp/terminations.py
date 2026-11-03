"""Position-only reach success; orientation is unconstrained."""
from isaaclab.envs.mdp import time_out
from .rewards import position_command_error


def reached_target(env, command_name, robot_cfg, ee_frame_cfg, threshold=0.008):
    return position_command_error(env, command_name, robot_cfg, ee_frame_cfg) < threshold

__all__ = ["time_out", "reached_target"]
