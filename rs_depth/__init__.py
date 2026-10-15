"""rs_depth — API-compatible CPU reimplementation of the close-range backend.

Mirrors the vendor package surface (`Calibration`, `FrameMetadata`,
`DepthRangeImprover`) so callers written against
`librealsense2-enhanced-depth` run unchanged.
"""

from rs_depth.calibration import Calibration
from rs_depth.improver import DepthRangeImprover
from rs_depth.metadata import FrameMetadata, StreamMetadata

__all__ = [
    "Calibration",
    "DepthRangeImprover",
    "FrameMetadata",
    "StreamMetadata",
]

__version__ = "0.1.0"
