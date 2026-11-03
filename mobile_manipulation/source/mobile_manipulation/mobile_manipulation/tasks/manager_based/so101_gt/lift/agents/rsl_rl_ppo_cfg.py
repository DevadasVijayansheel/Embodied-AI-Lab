"""Separate checkpoint namespace; PPO settings are provisional."""
from isaaclab.utils import configclass
from ....common.ppo_cfg import ResearchPPORunnerCfg

@configclass
class PPORunnerCfg(ResearchPPORunnerCfg):
    experiment_name = "so101_gt_lift"
    # Match the validated Reach mapping: both networks consume the policy group.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
