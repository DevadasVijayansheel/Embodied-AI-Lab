"""CPU-only discovery and source validation. This does not validate simulation readiness."""
import argparse
import ast
import importlib
from pathlib import Path
import gymnasium as gym
from .robots.assets import RobotAssetSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-assets', action='store_true', help='Fail when required local USDs are absent')
    args = parser.parse_args()
    root = Path(__file__).parent
    specs = [s for s in gym.registry.values() if str(s.kwargs.get('env_cfg_entry_point', '')).startswith('mobile_manipulation.')]
    assert len(specs) == 11, f'Expected ten research tasks plus template, found {len(specs)}'
    experiments = set()
    for spec in specs:
        for key in ('env_cfg_entry_point', 'rsl_rl_cfg_entry_point'):
            module, name = spec.kwargs[key].split(':')
            file = root.parent.joinpath(*module.split('.')).with_suffix('.py')
            tree = ast.parse(file.read_text(), filename=str(file))
            assert any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body), (file, name)
            if key == 'rsl_rl_cfg_entry_point':
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
                value = next(ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'experiment_name' for t in node.targets))
                assert value not in experiments, f'Duplicate experiment_name: {value}'
                experiments.add(value)
        if str(spec.entry_point).startswith('mobile_manipulation.'):
            module, name = spec.entry_point.split(':')
            assert callable(getattr(importlib.import_module(module), name))
        print(f'{spec.id}: {spec.kwargs["env_cfg_entry_point"]}')
    files = list(root.rglob('*.py'))
    for file in files:
        compile(file.read_text(), str(file), 'exec')
    print(f'Validated {len(specs)} actual Gym registrations, distinct experiment names and {len(files)} Python sources.')
    print('Isaac Lab configuration imports and simulation readiness are NOT validated by this command.')
    if args.check_assets:
        errors = []
        for variant in ('so101_gt', 'so101_perception', 'irobot_gt', 'irobot_perception'):
            try:
                print(RobotAssetSpec(variant).resolve())
            except FileNotFoundError as exc:
                errors.append(str(exc))
        if errors:
            parser.exit(1, '\n'.join(errors) + '\n')


if __name__ == '__main__':
    main()
