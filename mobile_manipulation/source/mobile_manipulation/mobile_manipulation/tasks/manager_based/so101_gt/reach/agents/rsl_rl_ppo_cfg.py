"""Separate checkpoint namespace; PPO settings are provisional."""
from isaaclab.utils import configclass
from ....common.ppo_cfg import ResearchPPORunnerCfg

@configclass
class PPORunnerCfg(ResearchPPORunnerCfg):
    experiment_name = "so101_gt_reach"
    # RSL-RL 3.x observation sets both consume the environment's policy group.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
