"""SO-101 GT joint-position reach specialization."""
from isaaclab.envs.mdp import JointPositionActionCfg, UniformPoseCommandCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from mobile_manipulation.robots.so101_gt_cfg import (
    ROBOT_ASSET, so101_gt_cfg, ARM_JOINTS, ARM_LIMITS, GRASP_OFFSET, GRASP_ROTATION,
)
from .reach_env_cfg import ReachEnvCfg

# Root-frame TCP target at saved angular states plus 12-degree shoulder-pan motion.
REACH_TARGET = (0.1903325521395122, -0.032209839850018206, 0.03367581183852951)


@configclass
class JointPosEnvCfg(ReachEnvCfg):
    robot_variant = "so101_gt"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = so101_gt_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = str(ROBOT_ASSET.resolve(self.robot_usd_path))
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=False,
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link", name="gripper_frame_link",
                offset=OffsetCfg(pos=GRASP_OFFSET, rot=GRASP_ROTATION),
            )],
        )
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS.copy(),
            preserve_order=True,
            scale=0.5,
            use_default_offset=True,
            clip=ARM_LIMITS.copy(),
        )
        self.commands.ee_pose.body_name = "gripper_link"
        self.commands.ee_pose.ranges = UniformPoseCommandCfg.Ranges(
            pos_x=(REACH_TARGET[0], REACH_TARGET[0]),
            pos_y=(REACH_TARGET[1], REACH_TARGET[1]),
            pos_z=(REACH_TARGET[2], REACH_TARGET[2]),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0))


@configclass
class JointPosEnvCfg_PLAY(JointPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Training configurations retain the previously tested single environment.
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
