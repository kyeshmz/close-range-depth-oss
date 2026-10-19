"""CPU/CUDA close-range stereo; all nonzero hardware depths are preserved.

Inputs must be a synchronized, rectified IR pair and aligned uint16 depth
in millimetres. Optional hole filling runs on the CPU for either backend.
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
        stereo_z_min_m: Optional[float] = None,
        use_median_filter: bool = True,
        fill_holes: bool = True,
        fill_radius: int = 15,
        backend: str = "cpu",
        cuda_library_path=None,
    ):
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be 'cpu' or 'cuda'")
        if not 0.0 < scale_factor <= 1.0:
            raise ValueError("scale_factor must be in (0, 1]")
        if fill_radius < 0 or int(fill_radius) != fill_radius:
            raise ValueError("fill_radius must be a nonnegative integer")
        self._calibration = calibration
        self._min_z_threshold_mm = min_z_threshold_mm
        self._scale_factor = float(scale_factor)
        self._crop_region = crop_region
        self._stereo_z_min_m = stereo_z_min_m
        self._use_median_filter = bool(use_median_filter)
        self._fill_holes = bool(fill_holes)
        self._fill_radius = int(fill_radius)
        self.backend = backend
        self._cuda = None
        self._limits(calibration)
        if backend == "cuda":
            from rs_depth._cuda import CudaBackend
            self._cuda = CudaBackend(cuda_library_path)

    @property
    def min_z_threshold_mm(self) -> int:
        return self._limits(self._calibration)[0]

    @property
    def crop_region(self):
        return self._crop_region

    def close(self):
        if self._cuda is not None:
            self._cuda.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _limits(self, cal):
        if not all(math.isfinite(v) and v > 0
                   for v in (cal.focal_length_px, cal.baseline_m)):
            raise ValueError("focal length and baseline must be positive and finite")
        threshold = (cal.min_z_threshold_mm if self._min_z_threshold_mm is None
                     else self._min_z_threshold_mm)
        if not math.isfinite(threshold) or not 1 <= threshold <= 65535:
            raise ValueError("min_z_threshold_mm must be in [1, 65535]")
        # Short-baseline preset, not a guarantee of the camera's physical range.
        floor = self._stereo_z_min_m
        if floor is None:
            floor = 0.02 if cal.baseline_m < 0.03 else 0.12
        if not math.isfinite(floor) or not 0 < floor * 1000 < int(threshold):
            raise ValueError("stereo_z_min_m must be positive and below the threshold")
        return int(threshold), float(floor)

    # -- public API -------------------------------------------------------

    def process(self, ir_left, ir_right=None, depth_mm=None, *,
                calibration: Optional[Calibration] = None,
                metadata=None) -> np.ndarray:
        cal = calibration or self._calibration
        ir_left = np.asanyarray(ir_left)
        ir_right = np.asanyarray(ir_right)
        depth_hw = np.asanyarray(depth_mm)
        self._check_inputs(ir_left, ir_right, depth_hw)
        threshold, floor = self._limits(cal)
        roi = np.ones(depth_hw.shape, dtype=bool)
        if self._crop_region is not None:
            x1, y1, x2, y2 = self._crop_region
            h, w = depth_hw.shape
            if not all(isinstance(v, (int, np.integer)) for v in self._crop_region):
                raise ValueError("crop_region must contain integer coordinates")
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError("crop_region must be nonempty and inside the image")
            roi[:] = False
            roi[y1:y2, x1:x2] = True
        if self.backend == "cuda":
            out = self._cuda.process(
                ir_left, ir_right, depth_hw, cal, threshold, self._scale_factor,
                floor, self._use_median_filter,
                tuple(self._crop_region) if self._crop_region is not None else None)
        else:
            out = self._improve(cal, ir_left, ir_right, depth_hw, threshold, floor, roi)
        return self._postprocess(out, depth_hw, threshold, floor, roi)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _check_inputs(ir_left, ir_right, depth_hw):
        if ir_left.shape != ir_right.shape or ir_left.ndim != 2:
            raise ValueError("IR pair must be two (H, W) images of equal size")
        if not ir_left.size:
            raise ValueError("IR images must be nonempty")
        if depth_hw.shape != ir_left.shape:
            raise ValueError("depth_mm must match IR resolution")
        if ir_left.dtype != np.uint8 or ir_right.dtype != np.uint8:
            raise ValueError("IR images must be uint8")
        if depth_hw.dtype != np.uint16:
            raise ValueError("depth_mm must be uint16")

    def _improve(self, cal, ir_left, ir_right, depth_hw, threshold, floor, roi):
        h, w = ir_left.shape
        s = self._scale_factor
        ws, hs = max(1, int(round(w * s))), max(1, int(round(h * s)))
        f_scaled = cal.focal_length_px * (ws / w)

        d_max = f_scaled * cal.baseline_m / floor
        num_disp = int(math.ceil(d_max / 16.0)) * 16
        if num_disp < 16:
            num_disp = 16
        if num_disp >= ws:
            raise ValueError(
                f"disparity search {num_disp} exceeds scaled width {ws}; "
                "use a wider input image or a larger stereo_z_min_m")

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
        in_band = (computed >= math.ceil(floor * 1000.0)) & (computed < threshold)
        take = in_band & (depth_hw == 0) & roi
        out = depth_hw.copy()
        out[take] = computed[take].astype(np.uint16)
        return out

    def _postprocess(self, out, depth_hw, t, floor, roi):
        if self._fill_holes:
            keep = ((out > 0) & (out < t)).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
            big = np.zeros_like(keep, dtype=bool)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= 25:
                    big[lab == i] = True
            newly_estimated = (depth_hw == 0) & (out > 0) & (out < t)
            out[newly_estimated & ~big] = 0
            out = self._fill(out, t, floor, roi)
        return out

    def _fill(self, out: np.ndarray, t: int, floor: float, roi) -> np.ndarray:
        close = ((out > 0) & (out < t)).astype(np.uint8)
        k = 2 * self._fill_radius + 1
        near = cv2.dilate(close, np.ones((k, k), np.uint8)) > 0
        region = ((out == 0) & near & roi).astype(np.uint8) * 255
        if not region.any():
            return out
        # Inpaint from close depths only: far HW values would contaminate
        # the interpolation across depth edges, so hide them from the
        # solver (they stay untouched in the output).
        src = np.where(close > 0, out.astype(np.float32), 0.0)
        filled = cv2.inpaint(src, region, 3, cv2.INPAINT_TELEA)
        res = out.copy()
        take = ((region > 0) & np.isfinite(filled)
                & (filled >= math.ceil(floor * 1000.0)) & (filled < t))
        res[take] = np.clip(filled[take], 0, 65535).astype(np.uint16)
        return res
