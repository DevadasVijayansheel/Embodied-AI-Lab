"""Incomplete research scaffold, using Isaac Lab v2.3.2 manager configuration types."""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg,
                               RewardTermCfg, TerminationTermCfg, EventTermCfg)
from isaaclab.envs import mdp
from isaaclab.utils import configclass
from mobile_manipulation.robots.assets import RobotAssetSpec

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Proprioception only. Task targets and perception features are still missing.
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        concatenate_terms = True
        enable_corruption = False
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    # Generic regularizer; deliberately no fabricated task reward.
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class ActionsCfg:
    # Set only after inspecting the actual articulation and controller requirements.
    pass

@configclass
class ResearchEnvCfg(ManagerBasedRLEnvCfg):
    robot_variant: str = ""
    skill: str = ""
    robot_usd_path: str | None = None
    # Clear items individually only after implementing and validating each requirement.
    incomplete: tuple[str, ...] = (
        "robot articulation, prim path, joint/body mappings, actuators and initial state",
        "scene geometry, task targets/objects and reset distributions",
        "task observations, commands and coordinate frames",
        "controller actions and limits",
        "task reward, success and failure conditions",
        "simulation smoke test and observation/action shape validation",
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    decimation = 2
    episode_length_s = 10.0

    def check_ready(self):
        issues = list(self.incomplete)
        try:
            path = RobotAssetSpec(self.robot_variant).resolve(self.robot_usd_path)
        except FileNotFoundError as exc:
            issues.insert(0, str(exc))
            path = None
        robot = getattr(self.scene, "robot", None)
        if robot is None:
            issues.append("scene.robot is not configured")
        elif path is not None and getattr(getattr(robot, "spawn", None), "usd_path", None) != str(path):
            issues.append("scene.robot.spawn.usd_path must match the resolved robot USD")
        if not any(value is not None for value in vars(self.actions).values()):
            issues.append("actions contains no control terms")
        if "perception" in self.robot_variant:
            if getattr(self.scene, "zed_camera", None) is None:
                issues.append("scene.zed_camera and calibrated perception pipeline are missing")
        if issues:
            raise RuntimeError(f"Incomplete {self.robot_variant}/{self.skill} environment:\n- " + "\n- ".join(issues))
