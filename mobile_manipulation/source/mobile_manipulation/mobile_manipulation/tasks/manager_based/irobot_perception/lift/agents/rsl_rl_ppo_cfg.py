"""Separate checkpoint namespace; PPO settings are provisional."""
from isaaclab.utils import configclass
from ....common.ppo_cfg import ResearchPPORunnerCfg

@configclass
class PPORunnerCfg(ResearchPPORunnerCfg):
    experiment_name = "irobot_perception_lift"
