"""Offline stats demo mirroring the vendor `range_depth.py` live loop.

Builds a synthetic close-range scene (textured plane HW cannot see),
runs DepthRangeImprover, and prints the HW -> improved recovery line.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_depth import Calibration, DepthRangeImprover

rng = np.random.default_rng(11)
W, H, D = 640, 480, 176  # d=176 @ F*B=36.4 -> z~=207mm, inside band
left = rng.integers(0, 256, (H, W), dtype=np.uint8)
right = np.zeros_like(left)
right[:, :W - D] = left[:, D:]
hw = np.zeros((H, W), dtype=np.uint16)
hw[:, :40] = 2500  # far strip

cal = Calibration.from_params(383.0, 0.095)
imp = DepthRangeImprover(cal, scale_factor=0.5, stereo_z_min_m=0.15)
T = imp.min_z_threshold_mm
out = imp.process(left, right, hw)

n_hw = int(((hw > 0) & (hw < T)).sum())
n_imp = int(((out > 0) & (out < T)).sum())
rescued = int(((hw == 0) & (out > 0) & (out < T)).sum())
print(f"threshold: {T} mm | HW close-range: {n_hw} px -> "
      f"improved: {n_imp} px | rescued: +{rescued} px")
