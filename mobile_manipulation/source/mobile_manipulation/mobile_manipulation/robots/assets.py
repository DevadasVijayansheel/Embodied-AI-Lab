"""Local USD resolution and explicit articulation construction."""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class RobotAssetSpec:
    variant: str

    @property
    def environment_variable(self):
        return f"MOBILE_MANIPULATION_{self.variant.upper()}_USD"

    def resolve(self, usd_path=None):
        """Resolve at launch, so environment overrides are not captured at import time."""
        raw = usd_path or os.environ.get(self.environment_variable)
        path = Path(raw).expanduser() if raw else Path(__file__).parent / "usd" / f"{self.variant}.usd"
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {self.variant} USD: {path}. Set {self.environment_variable} "
                                    "or supply cfg.robot_usd_path. No placeholder asset is provided.")
        return path

    def articulation(self, *, prim_path, actuators, init_state, usd_path=None, **kwargs):
        """Caller supplies verified prim path, actuators, initial state and physics settings."""
        from isaaclab.assets import ArticulationCfg
        from isaaclab.sim import UsdFileCfg
        return ArticulationCfg(prim_path=prim_path, spawn=UsdFileCfg(usd_path=str(self.resolve(usd_path))),
                               actuators=actuators, init_state=init_state, **kwargs)
