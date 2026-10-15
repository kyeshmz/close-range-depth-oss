"""DepthRangeImprover — CPU reimplementation of the close-range filter.

Pipeline (mirrors the vendor's documented recipe: accelerated stereo on the
IR pair, merged with hardware depth):

1. Downscale the IR pair by `scale_factor` (default 0.35, vendor default).
2. StereoSGBM over the *near* disparity band only — disparities larger than
   `f·B / stereo_z_min_m`, i.e. depths closer than the physical floor.
3. Disparity → depth (`Z = f·B / d`), 5x5 median, upscale to full res.
4. Merge: take computed depth only inside the close band
   (`z_floor <= z < min_z_threshold`) where HW is missing (`0`) or reports
   far (`>= threshold`); keep HW depth everywhere else.

All depths are uint16 millimetres, like the vendor API.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from rs_depth.calibration import Calibration


class DepthRangeImprover:
    def __init__(
        self,
        calibration: Calibration,
        min_z_threshold_mm: Optional[int] = None,
        scale_factor: float = 0.35,
        crop_region: Optional[tuple] = None,
        stereo_z_min_m: float = 0.12,
        use_median_filter: bool = True,
        fill_holes: bool = True,
        fill_radius: int = 15,
    ):
        if min_z_threshold_mm is None:
            min_z_threshold_mm = calibration.min_z_threshold_mm
        if not 0.0 < scale_factor <= 1.0:
            raise ValueError("scale_factor must be in (0, 1]")
        self._calibration = calibration
        self._min_z_threshold_mm = int(min_z_threshold_mm)
        self._scale_factor = float(scale_factor)
        self._crop_region = crop_region
        self._stereo_z_min_m = float(stereo_z_min_m)
        self._use_median_filter = bool(use_median_filter)
        self._fill_holes = bool(fill_holes)
        self._fill_radius = int(fill_radius)

    @property
    def min_z_threshold_mm(self) -> int:
        return self._min_z_threshold_mm

    @property
    def crop_region(self):
        return self._crop_region

    # -- public API (vendor-compatible) ------------------------------------

    def process(self, ir_left, ir_right=None, depth_mm=None, *,
                calibration: Optional[Calibration] = None,
                metadata=None) -> np.ndarray:
        cal = calibration or self._calibration
        ir_left = np.asanyarray(ir_left)
        ir_right = np.asanyarray(ir_right)
        depth_hw = np.asanyarray(depth_mm)
        self._check_inputs(ir_left, ir_right, depth_hw)

        if self._crop_region is not None:
            x1, y1, x2, y2 = self._crop_region
            out = depth_hw.copy()
            out[y1:y2, x1:x2] = self._improve(
                cal, ir_left[y1:y2, x1:x2],
                ir_right[y1:y2, x1:x2], depth_hw[y1:y2, x1:x2])
            return out
        return self._improve(cal, ir_left, ir_right, depth_hw)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _check_inputs(ir_left, ir_right, depth_hw):
        if ir_left.shape != ir_right.shape or ir_left.ndim != 2:
            raise ValueError("IR pair must be two (H, W) images of equal size")
        if depth_hw.shape != ir_left.shape:
            raise ValueError("depth_mm must match IR resolution")
        if ir_left.dtype != np.uint8 or ir_right.dtype != np.uint8:
            raise ValueError("IR images must be uint8")
        if depth_hw.dtype != np.uint16:
            raise ValueError("depth_mm must be uint16")

    def _improve(self, cal, ir_left, ir_right, depth_hw) -> np.ndarray:
        h, w = ir_left.shape
        s = self._scale_factor
        ws, hs = max(1, int(round(w * s))), max(1, int(round(h * s)))
        f_scaled = cal.focal_length_px * (ws / w)

        d_max = f_scaled * cal.baseline_m / self._stereo_z_min_m
        num_disp = int(math.ceil(d_max / 16.0)) * 16
        if num_disp < 16:
            num_disp = 16
        if num_disp >= ws:
            raise ValueError(
                f"disparity search {num_disp} exceeds scaled width {ws}; "
                "raise scale_factor or stereo_z_min_m")

        if (ws, hs) == (w, h):
            il, ir = ir_left, ir_right
        else:
            il = cv2.resize(ir_left, (ws, hs), interpolation=cv2.INTER_AREA)
            ir = cv2.resize(ir_right, (ws, hs), interpolation=cv2.INTER_AREA)

        stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=5,
            P1=8 * 5 * 5, P2=32 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = stereo.compute(il, ir).astype(np.float32) / 16.0

        valid = disp > 0.0
        z_scaled = np.zeros_like(disp, dtype=np.float32)
        z_scaled[valid] = (f_scaled * cal.baseline_m / disp[valid]) * 1000.0

        if self._use_median_filter:
            z_scaled = cv2.medianBlur(z_scaled, 5)

        if (ws, hs) == (w, h):
            z_full = z_scaled
        else:
            z_full = cv2.resize(z_scaled, (w, h),
                                interpolation=cv2.INTER_NEAREST)

        computed = z_full.astype(np.int32)
        z_floor_mm = int(self._stereo_z_min_m * 1000.0)
        t = self._min_z_threshold_mm
        in_band = (computed >= z_floor_mm) & (computed < t)
        hw_missing_or_far = (depth_hw.astype(np.int32) == 0) | (
            depth_hw.astype(np.int32) >= t)
        out = depth_hw.copy()
        out[in_band & hw_missing_or_far] = computed[
            in_band & hw_missing_or_far].astype(np.uint16)
        if self._fill_holes:
            # Drop speckle: tiny isolated close-valid components seed bad
            # fill, so keep only regions big enough to be surfaces.
            keep = ((out > 0) & (out < t)).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
            big = np.zeros_like(keep, dtype=bool)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= 25:
                    big[lab == i] = True
            out[(out > 0) & (out < t) & ~big] = 0
            out = self._fill(out, t)
        return out

    def _fill(self, out: np.ndarray, t: int) -> np.ndarray:
        # Interpolate textureless holes from neighboring close-band depths
        # (the vendor does this: ~2/3 of its rescued pixels have ~zero
        # local texture, unmeasurable by any matcher). Only fills zeros
        # adjacent to measured close depths; never overwrites HW data.
        close = ((out > 0) & (out < t)).astype(np.uint8)
        k = 2 * self._fill_radius + 1
        near = cv2.dilate(close, np.ones((k, k), np.uint8)) > 0
        region = ((out == 0) & near).astype(np.uint8) * 255
        if not region.any():
            return out
        # Inpaint from close depths only: far HW values would contaminate
        # the interpolation across depth edges, so hide them from the
        # solver (they stay untouched in the output).
        src = np.where(close > 0, out.astype(np.float32), 0.0)
        filled = cv2.inpaint(src, region, 3, cv2.INPAINT_TELEA)
        res = out.copy()
        take = region > 0
        res[take] = np.clip(filled[take], 0, 65535).astype(np.uint16)
        return res
