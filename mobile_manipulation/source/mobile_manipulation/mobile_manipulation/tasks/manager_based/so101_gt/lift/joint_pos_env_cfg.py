"""SO-101 GT joint-position arm and binary-gripper lift specialization."""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Sdf, UsdShade
from . import mdp
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from mobile_manipulation.robots.so101_gt_cfg import (
    ROBOT_ASSET, so101_gt_cfg, ARM_JOINTS, ARM_LIMITS, GRASP_OFFSET, GRASP_ROTATION,
)
from .lift_env_cfg import LiftEnvCfg, CUBE_POSITION


def spawn_marker_material(prim_path, cfg):
    """Preserve the marker material while avoiding the known Kit Sdr query stall."""
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Material already exists: {prim_path}")
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, value_type, value in (
        ("diffuseColor", Sdf.ValueTypeNames.Color3f, cfg.diffuse_color),
        ("emissiveColor", Sdf.ValueTypeNames.Color3f, cfg.emissive_color),
        ("roughness", Sdf.ValueTypeNames.Float, cfg.roughness),
        ("metallic", Sdf.ValueTypeNames.Float, cfg.metallic),
        ("opacity", Sdf.ValueTypeNames.Float, cfg.opacity),
    ):
        shader.CreateInput(name, value_type).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    material.CreateDisplacementOutput().ConnectToSource(shader.CreateOutput("displacement", Sdf.ValueTypeNames.Float))
    return shader.GetPrim()


@configclass
class So101GtLiftCubeEnvCfg(LiftEnvCfg):
    robot_variant = "so101_gt"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = so101_gt_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = str(ROBOT_ASSET.resolve(self.robot_usd_path))
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POSITION, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0, max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0, disable_gravity=False,
                ),
            ),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        # FrameTransformer requires both frame and connecting-line marker prototypes.
        marker_cfg.markers["connecting_line"].visual_material.func = spawn_marker_material
        marker_cfg.prim_path = "/Visuals/SO101GTLift/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link", name="gripper_frame_link",
                offset=OffsetCfg(pos=GRASP_OFFSET, rot=GRASP_ROTATION),
            )],
        )
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS.copy(),
            preserve_order=True,
            scale=0.5,
            use_default_offset=True,
            clip=ARM_LIMITS.copy(),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )
        # Preserve the existing cube goal: 7 cm above reset center in robot-root coordinates.
        goal = tuple(CUBE_POSITION[i] - self.scene.robot.init_state.pos[i]
                     + (0.07 if i == 2 else 0.0) for i in range(3))
        self.commands.object_pose.body_name = "gripper_link"
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(goal[0], goal[0]), pos_y=(goal[1], goal[1]), pos_z=(goal[2], goal[2]),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0))


@configclass
class So101GtLiftCubeEnvCfg_PLAY(So101GtLiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# Compatibility names for existing tools and configurations.
JointPosEnvCfg = So101GtLiftCubeEnvCfg
JointPosEnvCfg_PLAY = So101GtLiftCubeEnvCfg_PLAY
