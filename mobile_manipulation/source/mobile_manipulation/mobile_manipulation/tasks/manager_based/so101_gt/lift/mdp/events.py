"""Deterministic cube and robot reset, including position targets."""
# Inspected official reset restores root/joint defaults and zero default velocities.
# The event passes reset_joint_targets=True, preventing stale controller targets.
from isaaclab.envs.mdp import reset_scene_to_default
__all__ = ["reset_scene_to_default"]
