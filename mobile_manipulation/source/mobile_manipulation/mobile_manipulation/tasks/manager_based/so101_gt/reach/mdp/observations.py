"""Ground-truth robot-root target/TCP observations and TCP command metrics.

Official joint_pos_rel/joint_vel_rel subtract the configured defaults for all six
joints; last_action reports the manager's raw actions. No noise or pose randomization.
"""
from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms
import torch


def target_position(env, command_name="ee_pose"):
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
           "end_effector_position", "TcpPoseCommandCfg"]
