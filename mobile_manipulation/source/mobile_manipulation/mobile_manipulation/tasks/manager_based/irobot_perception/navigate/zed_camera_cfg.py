"""ZED configuration boundary; no assumed prim path, pose or calibration, no PPO dependencies."""
from isaaclab.sensors import CameraCfg

def make_zed_camera_cfg(*, prim_path, width, height, offset, spawn, data_types, update_period):
    """Supply measured calibration via spawn and offset, or spawn=None for an existing calibrated USD camera.

    This is one CameraCfg, not a validated stereo/depth ZED model. Implement the
    stereo pair and perception feature extraction for the chosen experiment.
    """
    if not prim_path or width <= 0 or height <= 0:
        raise ValueError("A verified camera prim path and positive image dimensions are required")
    return CameraCfg(prim_path=prim_path, width=width, height=height, offset=offset,
                     spawn=spawn, data_types=data_types, update_period=update_period)
