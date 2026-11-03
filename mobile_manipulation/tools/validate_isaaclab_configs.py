"""Validate real configuration imports after Kit starts; creates no environments or training runs."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

try:
    import gymnasium as gym
    import importlib
    import mobile_manipulation

    names = set()
    for spec in gym.registry.values():
        if not str(spec.kwargs.get('env_cfg_entry_point', '')).startswith('mobile_manipulation.'):
            continue
        configs = {}
        for key in ('env_cfg_entry_point', 'rsl_rl_cfg_entry_point'):
            module, cls = spec.kwargs[key].split(':')
            configs[key] = getattr(importlib.import_module(module), cls)()
        cfg = configs['env_cfg_entry_point']
        cfg.validate()
        ppo = configs['rsl_rl_cfg_entry_point']
        ppo.validate()
        assert ppo.experiment_name not in names
        names.add(ppo.experiment_name)
        if not spec.id.startswith('Template-'):
            package = spec.kwargs['env_cfg_entry_point'].split(':')[0].rsplit('.', 1)[0]
            importlib.import_module(package + '.mdp')
            if cfg.skill != 'navigate':
                joint_cfg = importlib.import_module(package + '.joint_pos_env_cfg').JointPosEnvCfg()
                assert joint_cfg.robot_variant == cfg.robot_variant
            if 'perception' in cfg.robot_variant:
                importlib.import_module(package + '.zed_camera_cfg')
            try:
                if hasattr(cfg, 'check_ready'):
                    cfg.check_ready()
            except RuntimeError as exc:
                print(f'{spec.id}: imports OK; launch blocked: {exc}')
            else:
                print(f'{spec.id}: imports OK; readiness guard passed; simulation still requires a smoke test')
        else:
            print(f'{spec.id}: imports OK')
finally:
    app.close()
