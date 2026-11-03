"""Import and inspect SO-101 GT using Kit; no scene construction or training."""
import argparse
import hashlib
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--skip-kit-cleanup', action='store_true', help='Diagnostic exit after validation; avoids the known native Kit teardown crash')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
status = 1
try:
    import math
    import re
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom, Gf
    from mobile_manipulation.robots.so101_gt_cfg import so101_gt_cfg, SO101_GT_USD_PATH
    from mobile_manipulation.tasks.manager_based.so101_gt.reach.joint_pos_env_cfg import JointPosEnvCfg as ReachCfg, JointPosEnvCfg_PLAY as ReachPlayCfg
    from mobile_manipulation.tasks.manager_based.so101_gt.lift.joint_pos_env_cfg import So101GtLiftCubeEnvCfg as LiftCfg, So101GtLiftCubeEnvCfg_PLAY as LiftPlayCfg

    before = hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest()
    cfg = so101_gt_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.validate()
    assert cfg.spawn.usd_path == str(SO101_GT_USD_PATH)
    assert cfg.spawn.articulation_props.fix_root_link is True
    assert cfg.spawn.articulation_props.enabled_self_collisions is False
    for cls in (ReachCfg, LiftCfg, ReachPlayCfg, LiftPlayCfg):
        env_cfg = cls()
        env_cfg.validate()
        assert env_cfg.scene.robot.prim_path == cfg.prim_path
        assert env_cfg.scene.robot.actuators == cfg.actuators
        print(f"{cls.__module__}: imports and robot assignment OK")
    stage = Usd.Stage.Open(str(SO101_GT_USD_PATH))
    root = stage.GetDefaultPrim()
    from mobile_manipulation.robots.so101_gt_cfg import GRASP_OFFSET, GRASP_ROTATION, BASE_COLLISION_MIN_Z, SAVED_ROOT_POSITION
    from mobile_manipulation.tasks.manager_based.so101_gt.reach.joint_pos_env_cfg import REACH_TARGET
    cache = UsdGeom.XformCache()
    gripper = stage.GetPrimAtPath(str(root.GetPath()) + '/gripper_link')
    tcp = stage.GetPrimAtPath(str(root.GetPath()) + '/gripper_frame_link')
    relative = cache.GetLocalToWorldTransform(tcp) * cache.GetLocalToWorldTransform(gripper).GetInverse()
    transform = Gf.Transform(relative)
    quat = transform.GetRotation().GetQuat()
    position = tuple(transform.GetTranslation())
    rotation = (quat.GetReal(), *quat.GetImaginary())
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(position, GRASP_OFFSET))
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(rotation, GRASP_ROTATION))
    fixed = next(UsdPhysics.FixedJoint(p) for p in stage.Traverse()
                 if p.IsA(UsdPhysics.FixedJoint) and p.GetName() == 'gripper_frame_joint')
    assert fixed.GetBody0Rel().GetTargets() == [gripper.GetPath()]
    assert fixed.GetBody1Rel().GetTargets() == [tcp.GetPath()]
    def joint_frame(pos, rot):
        value = Gf.Matrix4d(1)
        value.SetRotate(Gf.Quatd(rot.GetReal(), Gf.Vec3d(rot.GetImaginary())))
        value.SetTranslateOnly(Gf.Vec3d(pos))
        return value
    fixed_offset = joint_frame(fixed.GetLocalPos1Attr().Get(), fixed.GetLocalRot1Attr().Get()).GetInverse() * joint_frame(fixed.GetLocalPos0Attr().Get(), fixed.GetLocalRot0Attr().Get())
    assert max(abs(relative[i][j] - fixed_offset[i][j]) for i in range(4) for j in range(4)) < 1e-6
    frames = {'base_link': Gf.Matrix4d(1)}
    for name in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper_frame_joint'):
        prim = next(p for p in stage.Traverse() if p.GetName() == name)
        joint = UsdPhysics.Joint(prim)
        local0 = joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        local1 = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        motion = Gf.Matrix4d(1)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            direction = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0), 'Z': Gf.Vec3d(0, 0, 1)}[axis]
            angle = prim.GetAttribute('state:angular:physics:position').Get() + (12 if name == 'shoulder_pan' else 0)
            motion.SetRotate(Gf.Rotation(direction, angle))
        parent = joint.GetBody0Rel().GetTargets()[0].name
        child = joint.GetBody1Rel().GetTargets()[0].name
        frames[child] = local1.GetInverse() * motion * local0 * frames[parent]
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(frames['gripper_frame_link'].ExtractTranslation(), REACH_TARGET))
    # Joint-limit-constrained kinematic check; this does not prove collision-free grasping.
    import numpy as np
    from scipy.optimize import least_squares
    from mobile_manipulation.robots.so101_gt_cfg import ARM_JOINTS, ARM_LIMITS
    joint_data = []
    for name in (*ARM_JOINTS, 'gripper_frame_joint'):
        prim = next(p for p in stage.Traverse() if p.GetName() == name)
        joint = UsdPhysics.Joint(prim)
        local0 = joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        local1 = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        joint_data.append((name, joint.GetBody0Rel().GetTargets()[0].name,
                           joint.GetBody1Rel().GetTargets()[0].name, local0, local1))
    def fk_tcp(q):
        values = dict(zip(ARM_JOINTS, q))
        result = {'base_link': Gf.Matrix4d(1)}
        for name, parent, child, local0, local1 in joint_data:
            motion = Gf.Matrix4d(1)
            if name in values:
                motion.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(values[name])))
            result[child] = local1.GetInverse() * motion * local0 * result[parent]
        return np.array(result['gripper_frame_link'].ExtractTranslation())
    q0 = np.array([cfg.init_state.joint_pos[name] for name in ARM_JOINTS])
    limits = np.array([ARM_LIMITS[name] for name in ARM_JOINTS])
    lift = LiftCfg()
    for label, height in (('cube', lift.scene.object.init_state.pos[2]),
                          ('elevated goal', lift.scene.object.init_state.pos[2] + 0.07)):
        target = np.array(lift.scene.object.init_state.pos) - np.array(cfg.init_state.pos)
        target[2] = height - cfg.init_state.pos[2]
        solution = least_squares(lambda q: fk_tcp(q) - target, q0,
            bounds=(limits[:, 0], limits[:, 1]), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        error = np.linalg.norm(fk_tcp(solution.x) - target)
        assert error < 1e-4, f'{label} outside tested kinematic workspace: {error}'
        print('Kinematic workspace:', label, 'error m', error,
              'joints rad', solution.x.tolist(), flush=True)
    print('VERIFIED TCP translation:', position, 'quaternion wxyz:', rotation, flush=True)
    print('Fixed joint agrees with authored transform within 1e-6; FK target:', REACH_TARGET, flush=True)
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1
    articulation = roots[0]
    assert cfg.articulation_root_prim_path is None, 'Use automatic root discovery after fixed-base spawn' 
    assert articulation.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert PhysxSchema.PhysxArticulationAPI(articulation).GetArticulationEnabledAttr().Get()
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    assert set(joints) == set(cfg.init_state.joint_pos), 'Joint name mismatch'
    covered = set()
    for name, actuator in cfg.actuators.items():
        matched = {joint for joint in joints if any(re.fullmatch(expr, joint) for expr in actuator.joint_names_expr)}
        assert matched and not covered.intersection(matched), 'Empty or overlapping actuator group'
        covered.update(matched)
        print(f"Actuator {name}: {sorted(matched)}")
    assert covered == set(joints)
    for name, prim in joints.items():
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        assert lower <= math.degrees(cfg.init_state.joint_pos[name]) <= upper
        saved_position = prim.GetAttribute('state:angular:physics:position').Get()
        assert saved_position is not None
        assert math.isclose(cfg.init_state.joint_pos[name], math.radians(saved_position), abs_tol=1e-12)
        assert prim.HasAPI(UsdPhysics.DriveAPI, 'angular'), f'No angular drive: {name}'
        drive = UsdPhysics.DriveAPI(prim, 'angular')
        print(f"{name}: limits [{lower:.4f}, {upper:.4f}] degrees; USD drive Kp={drive.GetStiffnessAttr().Get()}, "
              f"Kd={drive.GetDampingAttr().Get()}, maxForce={drive.GetMaxForceAttr().Get()}")
    anchors = [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)
               and (not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
                    or not UsdPhysics.Joint(p).GetBody1Rel().GetTargets())]
    print(f"Articulation root: {articulation.GetPath()}; USD world anchors: {len(anchors)}; spawn fix_root_link=True")
    saved_pose = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(articulation))
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(SAVED_ROOT_POSITION, saved_pose.GetTranslation()))
    base = stage.GetPrimAtPath(str(root.GetPath()) + '/base_link')
    minimum = float('inf')
    mesh_count = 0
    for prim in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        ancestor = prim
        collider = False
        while ancestor and ancestor != base:
            if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                collider |= bool(UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get())
            ancestor = ancestor.GetParent()
        if collider:
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points:
                matrix = cache.GetLocalToWorldTransform(prim)
                minimum = min(minimum, min(matrix.Transform(Gf.Vec3d(p))[2] for p in points))
                mesh_count += 1
    assert mesh_count == 4
    assert math.isclose(minimum - SAVED_ROOT_POSITION[2], BASE_COLLISION_MIN_Z, abs_tol=1e-12)
    assert math.isclose(cfg.init_state.pos[2] + BASE_COLLISION_MIN_Z, 0.0, abs_tol=1e-12)
    assert cfg.init_state.pos[:2] == SAVED_ROOT_POSITION[:2]
    print('Ground placement: collision world minimum', minimum, 'root', cfg.init_state.pos,
          'vertical translation', cfg.init_state.pos[2] - SAVED_ROOT_POSITION[2], flush=True)
    for cls in (ReachCfg, LiftCfg):
        scene = cls().scene
        assert not hasattr(scene, 'table')
        assert scene.ground.init_state.pos[2] == 0.0
    cube = LiftCfg().scene.object
    assert cube.spawn.usd_path.endswith("/Props/Blocks/DexCube/dex_cube_instanceable.usd")
    assert cube.spawn.scale == (0.5, 0.5, 0.5)
    assert cube.init_state.pos == (0.2, 0.0, 0.015)
    quaternion = saved_pose.GetRotation().GetQuat()
    saved_rotation = (quaternion.GetReal(), *quaternion.GetImaginary())
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(cfg.init_state.rot, saved_rotation))
    assert cfg.init_state.joint_vel == {'.*': 0.0}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            assert not PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
    assert hashlib.sha256(SO101_GT_USD_PATH.read_bytes()).hexdigest() == before
    print('PASS: real imports, config validation, replace(), joint mappings, initial limits and gravity. USD unchanged.')
    print('Run tools/test_so101_gt.py for runtime checks; grasp robustness needs a physical lift test.')
    status = 0
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    if args.skip_kit_cleanup:
        import os
        os._exit(status)
    launcher.app.close(wait_for_replicator=False)
