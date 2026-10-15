"""Camera calibration container.

Mirrors `rs_depth_calibration.hpp` / `rs_depth/calibration.py` from the
vendor package: focal length + baseline drive everything, the rest is
carried for future rectification/distortion handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Calibration:
    focal_length_px: float = 388.0
    baseline_m: float = 0.095
    fy: float = 0.0
    ppx: float = 0.0
    ppy: float = 0.0
    width: int = 0
    height: int = 0
    ir_distortion_model: int = 0
    ir_distortion_coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_params(
        cls, focal_length_px: float, baseline_m: float
    ) -> "Calibration":
        return cls(focal_length_px=float(focal_length_px),
                   baseline_m=float(baseline_m))

    @classmethod
    def from_sdk(cls, ir_intrin, ir_extrin,
                 rgb_intrin=None, depth_to_rgb=None) -> "Calibration":
        """Build from librealsense intrinsics/extrinsics (duck-typed).

        Accepts real `pyrealsense2` objects or any objects/structs with the
        same attributes, so this works without the SDK installed.
        """
        get = lambda o, k, d=0: getattr(o, k, d)
        trans = get(ir_extrin, "translation", [0.0])[0]
        coeffs = tuple(get(ir_intrin, "coeffs", (0, 0, 0, 0, 0))[:5])
        return cls(
            focal_length_px=float(get(ir_intrin, "fx")),
            baseline_m=abs(float(trans)),
            fy=float(get(ir_intrin, "fy", 0.0)),
            ppx=float(get(ir_intrin, "ppx", 0.0)),
            ppy=float(get(ir_intrin, "ppy", 0.0)),
            width=int(get(ir_intrin, "width", 0)),
            height=int(get(ir_intrin, "height", 0)),
            ir_distortion_model=int(get(ir_intrin, "model", 0)),
            ir_distortion_coeffs=coeffs + (0.0,) * (5 - len(coeffs)),
        )

    @property
    def min_z_threshold_mm(self) -> int:
        """Auto threshold: F·B/105 (max HW disparity + margin), in mm."""
        return int(self.focal_length_px * self.baseline_m * 1000.0 / 105.0)
