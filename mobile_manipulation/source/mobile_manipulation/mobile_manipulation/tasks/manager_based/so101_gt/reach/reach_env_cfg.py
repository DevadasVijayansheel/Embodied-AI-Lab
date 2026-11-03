"""Reusable position-only reach task; subclasses supply the robot, TCP and control."""
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg, RewardTermCfg,
    TerminationTermCfg, EventTermCfg, SceneEntityCfg)
from isaaclab.utils import configclass
from . import mdp

GROUND_HEIGHT = 0.0


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Ground-supported scene; the subclass supplies robot and TCP sensor."""
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_HEIGHT)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))



@configclass
class CommandsCfg:
    ee_pose = mdp.TcpPoseCommandCfg(asset_name="robot", body_name=MISSING,
        resampling_time_range=(10.0, 10.0), debug_vis=False, ranges=MISSING)

@configclass
class ActionsCfg:
    arm_action: JointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        target_position = ObservationTermCfg(func=mdp.target_position)
        end_effector_position = ObservationTermCfg(func=mdp.end_effector_position,
            params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")})
        actions = ObservationTermCfg(func=mdp.last_action)
        concatenate_terms = True
        enable_corruption = False
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    distance = RewardTermCfg(func=mdp.position_command_error, weight=-1.0,
        params={"command_name": "ee_pose", "robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")})
    reaching = RewardTermCfg(func=mdp.position_command_error_tanh, weight=1.0,
        params={"command_name": "ee_pose", "std": 0.04,
                "robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")})
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001)
    action_magnitude = RewardTermCfg(func=mdp.action_l2, weight=-0.0001)

@configclass
class EventsCfg:
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=mdp.reached_target,
        params={"command_name": "ee_pose", "threshold": 0.008,
                "robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")})

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    robot_usd_path: str | None = None
    skill = "reach"
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    decimation = 4
    episode_length_s = 10.0

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (0.65, 0.65, 0.40)
        self.viewer.lookat = (0.12, 0.0, 0.08)

# Legacy alias for the generic parent; Gym launches the specialized joint-position class.
EnvCfg = ReachEnvCfg
