"""Launch guard evaluated before Isaac Lab allocates a scene."""
def make_env(cfg, **kwargs):
    cfg.check_ready()
    from isaaclab.envs import ManagerBasedRLEnv
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)
