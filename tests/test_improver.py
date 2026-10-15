"""Synthetic tests for DepthRangeImprover — no camera needed."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_depth import Calibration, DepthRangeImprover  # noqa: E402


def test_auto_threshold():
    cal = Calibration.from_params(383.0, 0.095)
    assert cal.min_z_threshold_mm == int(383.0 * 0.095 * 1000.0 / 105.0)
    imp = DepthRangeImprover(cal)
    assert imp.min_z_threshold_mm == cal.min_z_threshold_mm


def test_uniform_ir_keeps_hw_depth():
    # Textureless IR -> SGBM finds nothing valid -> output must equal HW.
    cal = Calibration.from_params(383.0, 0.095)
    imp = DepthRangeImprover(cal, scale_factor=1.0)
    hw = np.full((240, 320), 2000, dtype=np.uint16)
    hw[10:20, 10:20] = 0
    uniform = np.full((240, 320), 128, np.uint8)
    out = imp.process(uniform, uniform, hw)
    assert out.shape == hw.shape and out.dtype == np.uint16
    assert np.array_equal(out, hw), "no stereo evidence, HW must pass through"


def test_close_range_fill():
    # Fronto-parallel textured plane at disparity d: z = F*B/d.
    # F*B = 40, d = 112 -> z ~= 357mm < T ~= 381mm (inside close band).
    rng = np.random.default_rng(7)
    w, h, d = 320, 240, 112
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :w - d] = left[:, d:]
    hw = np.zeros((h, w), dtype=np.uint16)
    hw[:, :20] = 1500  # far strip the merge must preserve

    cal = Calibration.from_params(400.0, 0.10)
    imp = DepthRangeImprover(cal, scale_factor=1.0, stereo_z_min_m=0.25)
    out = imp.process(left, right, hw)

    inner = out[10:h - 10, d + 10:w - 10]
    valid = inner[inner > 0]
    assert valid.size > 0.5 * inner.size, (
        f"only {valid.size}/{inner.size} px recovered")
    med = float(np.median(valid))
    assert abs(med - 357) < 0.15 * 357, f"median depth {med:.0f}mm off 357mm"
    assert np.array_equal(out[:, :20], hw[:, :20]), "far strip must survive"


def test_crop_roundtrip():
    rng = np.random.default_rng(3)
    cal = Calibration.from_params(400.0, 0.10)
    imp = DepthRangeImprover(cal, scale_factor=1.0, stereo_z_min_m=0.25,
                             crop_region=(80, 0, 320, 240))
    w, h, d = 320, 240, 112
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :w - d] = left[:, d:]
    hw = np.zeros((h, w), dtype=np.uint16)
    out = imp.process(left, right, hw)
    assert out.shape == (h, w)
    assert np.array_equal(out[:, :80], hw[:, :80])


if __name__ == "__main__":
    test_auto_threshold()
    print("auto_threshold ok")
    test_uniform_ir_keeps_hw_depth()
    print("uniform_ir ok")
    test_close_range_fill()
    print("close_range_fill ok")
    test_crop_roundtrip()
    print("crop_roundtrip ok")
    print("ALL TESTS PASSED")
