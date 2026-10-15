"""Per-frame metadata containers.

Mirrors `rs_depth_metadata.h` / `rs_depth/_metadata.py`. The CPU prototype
only needs `width`/`height`; sensor fields are carried for API parity and
future exposure/gain-adaptive tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamMetadata:
    frame_number: int = 0
    timestamp_us: float = 0.0
    exposure_us: int = 0
    gain: int = 0
    laser_power: int = 0

    @classmethod
    def from_rs2_frame(cls, frame) -> "StreamMetadata":
        m = cls()
        try:
            get = frame.get_frame_metadata
            import pyrealsense2 as rs  # noqa: F401  (only for enum values)

            def _v(key, default=0):
                try:
                    return frame.get_frame_metadata(key)
                except Exception:
                    return default

            m.frame_number = _v(rs.frame_metadata_value.frame_counter)
            m.exposure_us = _v(rs.frame_metadata_value.actual_exposure)
            m.gain = _v(rs.frame_metadata_value.gain_level)
            m.laser_power = _v(rs.frame_metadata_value.frame_laser_power)
        except Exception:
            pass
        return m


@dataclass
class FrameMetadata:
    ir: StreamMetadata = field(default_factory=StreamMetadata)
    depth: StreamMetadata = field(default_factory=StreamMetadata)
    ir_format: str = "y8"
    width: int = 0
    height: int = 0

    @classmethod
    def from_rs2_frameset(cls, frameset,
                          ir_format: str = "y8") -> "FrameMetadata":
        m = cls(ir_format=ir_format)
        try:
            ir = frameset.get_infrared_frame(1)
            m.ir = StreamMetadata.from_rs2_frame(ir)
            m.width = ir.get_width()
            m.height = ir.get_height()
        except Exception:
            pass
        try:
            d = frameset.get_depth_frame()
            m.depth = StreamMetadata.from_rs2_frame(d)
            if not m.width:
                m.width, m.height = d.get_width(), d.get_height()
        except Exception:
            pass
        return m
