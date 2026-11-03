"""Register iRobot-Perception-Navigate-v0; configuration imports remain lazy."""
import gymnasium as gym

gym.register(
    id="iRobot-Perception-Navigate-v0",
    entry_point="mobile_manipulation.tasks.manager_based.common.env:make_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigate_env_cfg:EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
