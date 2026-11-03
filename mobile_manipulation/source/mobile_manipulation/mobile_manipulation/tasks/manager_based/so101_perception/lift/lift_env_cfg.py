"""SO101-Perception-Lift-v0: registered but incomplete; launch guard reports missing inputs."""
from isaaclab.utils import configclass
from ...common.env_cfg import ResearchEnvCfg

@configclass
class EnvCfg(ResearchEnvCfg):
    robot_variant = "so101_perception"
    skill = "lift"
