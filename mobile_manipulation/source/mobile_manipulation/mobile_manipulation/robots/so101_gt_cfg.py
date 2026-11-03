import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Port the verified settings from isaac_so_arm101/robots/trs_so101/so_arm101.py.
# Import-time configuration is portable; asset existence is checked at launch.
from .assets import RobotAssetSpec

SO101_GT_USD_PATH = Path(__file__).resolve().parent / "usd" / "so101_gt.usd"
ROBOT_ASSET = RobotAssetSpec("so101_gt")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# USD hard limits, converted from degrees. Clip processed position targets.
ARM_LIMITS = {name: (math.radians(lo), math.radians(hi)) for name, lo, hi in (
    ("shoulder_pan", -109.99987030029297, 109.99987030029297),
    ("shoulder_lift", -100.00003814697266, 100.00003814697266),
    ("elbow_flex", -96.82986450195312, 96.82986450195312),
    ("wrist_flex", -94.99983215332031, 94.99983215332031),
    ("wrist_roll", -157.2110137939453, 162.78932189941406),
)}

GRASP_OFFSET = (-0.00790001197275339, -0.00021812137186685843, -0.09812741222130128)
GRASP_ROTATION = (0.0000012606114104306037, -3.0179257313699397e-9, 0.9999999999992054, 4.439806178955286e-10)


# Measured from enabled base collision mesh vertices in the authored USD.
# Root-relative lowest base collision z; place that surface on a z=0 ground.
BASE_COLLISION_MIN_Z = -0.0024002582691891317
SAVED_ROOT_POSITION = (-2.123415470123291e-7, 1.3396129361353815e-7, -0.07629992067813873)
GROUND_ROOT_POSITION = (SAVED_ROOT_POSITION[0], SAVED_ROOT_POSITION[1], -BASE_COLLISION_MIN_Z)

##
# Configuration
##

so101_gt_cfg = ArticulationCfg(
    # Auto-discover the root: fix_root_link moves the USD root API from
    # base_link to its parent in the spawned stage (Isaac Lab 2.3.2).
    articulation_root_prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_GT_USD_PATH),
        activate_contact_sensors=False,  # Preserve the original sensor setting.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The original enabled self-collisions without a documented reason.
            enabled_self_collisions=False,
            # The inspected USD has no world-fixed joint; preserve URDF fix_base=True.
            # Isaac Lab applies this to the spawned stage, not the saved USD file.
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Ground placement: authored base collision bottom at z=0.
        # Only root translation changes; saved angular states remain below.
        pos=GROUND_ROOT_POSITION,
        rot=(1.0, 0.0, 0.0, 0.0),
        # Authored state:angular:physics:position, not drive targets.
        # USD angular state uses degrees; Isaac Lab joint_pos uses radians.
        joint_pos={
            "shoulder_pan": math.radians(-0.0007597862277179956),
            "shoulder_lift": math.radians(-99.99591064453125),
            "elbow_flex": math.radians(90.89627838134766),
            "wrist_flex": math.radians(53.7000846862793),
            "wrist_roll": math.radians(-4.415210241859313e-6),
            "gripper": math.radians(-2.4914072582760127e-6),
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


def make_so101_gt_cfg(*, prim_path, usd_path=None, **overrides):
    """Copy the preserved config, checking USD existence without modifying the file.

    Optional kwargs replace ArticulationCfg fields explicitly. The default and
    environment-variable USD overrides use the same resolver as the launch guard.
    """
    cfg = so101_gt_cfg.replace(prim_path=prim_path, **overrides)
    cfg.spawn.usd_path = str(ROBOT_ASSET.resolve(usd_path))
    return cfg
