"""iRobot-GT-Navigate-v0: registered but incomplete; launch guard reports missing inputs."""
from isaaclab.utils import configclass
from ...common.env_cfg import ResearchEnvCfg

@configclass
class EnvCfg(ResearchEnvCfg):
    robot_variant = "irobot_gt"
    skill = "navigate"
