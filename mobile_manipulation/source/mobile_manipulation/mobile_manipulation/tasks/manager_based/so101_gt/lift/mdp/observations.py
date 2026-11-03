# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


# Local additions: observations and command metrics at the USD-verified TCP.


def target_position(env, command_name="object_pose"):
    """Position component of the root-frame pose command; orientation is not observed."""
    return env.command_manager.get_command(command_name)[:, :3]


def end_effector_position(env, robot_cfg=SceneEntityCfg("robot"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """TCP position in the robot root frame, independent of sensor source offsets."""
    robot = env.scene[robot_cfg.name]
    tcp = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    pos, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, tcp)
    return pos


class TcpPoseCommand(UniformPoseCommand):
    """Retain official pose sampling while measuring tracking at the offset TCP."""

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            self.pose_command_b[:, :3], self.pose_command_b[:, 3:])
        frame = self._env.scene[self.cfg.frame_name].data
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:],
            frame.target_pos_w[:, 0, :], frame.target_quat_w[:, 0, :])
        self.metrics["position_error"] = torch.linalg.vector_norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.linalg.vector_norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        frame = self._env.scene[self.cfg.frame_name].data
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.current_pose_visualizer.visualize(frame.target_pos_w[:, 0], frame.target_quat_w[:, 0])


@configclass
class TcpPoseCommandCfg(UniformPoseCommandCfg):
    class_type = TcpPoseCommand
    frame_name: str = "ee_frame"


__all__ = ["joint_pos_rel", "joint_vel_rel", "last_action", "target_position",
           "end_effector_position", "object_position_in_robot_root_frame", "TcpPoseCommandCfg"]
