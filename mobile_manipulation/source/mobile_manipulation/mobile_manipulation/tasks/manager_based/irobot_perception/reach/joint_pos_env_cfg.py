"""Joint position variant: provide actual controlled joints and scale before use."""
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass
from .reach_env_cfg import EnvCfg

@configclass
class JointPosEnvCfg(EnvCfg):
    def configure_joint_position_control(self, *, joint_names, scale):
        if not joint_names:
            raise ValueError("Supply verified joint names from the robot articulation")
        self.actions.joint_position = JointPositionActionCfg(
            asset_name="robot", joint_names=list(joint_names), scale=scale, use_default_offset=True
        )
