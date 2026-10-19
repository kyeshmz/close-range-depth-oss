"""Synthetic tests for DepthRangeImprover — no camera needed."""

import os
import sys

import numpy as np
import pytest

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
    assert np.count_nonzero(out[:, 180:300]) > 10000


def test_isolated_hardware_depth_survives_speckle_cleanup():
    cal = Calibration.from_params(395.16196, 0.017882943)
    ir = np.full((480, 640), 128, dtype=np.uint8)
    hw = np.full(ir.shape, 2000, dtype=np.uint16)
    hw[240, 320] = 50
    out = DepthRangeImprover(cal).process(ir, ir, hw)
    np.testing.assert_array_equal(out, hw)


def test_all_nonzero_hardware_depths_are_preserved():
    rng = np.random.default_rng(7)
    cal = Calibration.from_params(395.16196, 0.017882943)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-140] = left[:, 140:]
    hw = np.zeros(left.shape, dtype=np.uint16)
    hw[:, 400:450] = 500
    hw[:, 450:500] = 35
    hw[240, 550] = 65535
    for fill in (False, True):
        out = DepthRangeImprover(cal, fill_holes=fill).process(left, right, hw)
        np.testing.assert_array_equal(out[hw != 0], hw[hw != 0])
        assert np.count_nonzero(out[hw == 0]) > 0


def test_d405_default_recovers_depth_below_67mm():
    rng = np.random.default_rng(9)
    cal = Calibration.from_params(395.16196, 0.017882943)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-140] = left[:, 140:]
    hw = np.zeros(left.shape, dtype=np.uint16)
    out = DepthRangeImprover(cal, fill_holes=False).process(left, right, hw)
    valid = out[20:-20, 400:-20]
    valid = valid[valid > 0]
    assert valid.size > 50000
    assert abs(float(np.median(valid)) - 50.5) < 2
    assert np.all((valid >= 20) & (valid < 67))


def test_fill_never_escapes_band_or_crop():
    cal = Calibration.from_params(395.16196, 0.017882943)
    imp = DepthRangeImprover(cal)
    hw = np.zeros((60, 80), dtype=np.uint16)
    hw[20:40, 30:50] = 45
    roi = np.zeros_like(hw, dtype=bool)
    roi[10:50, 25:55] = True
    out = imp._postprocess(hw.copy(), hw, 67, 0.02, roi)
    np.testing.assert_array_equal(out[~roi], hw[~roi])
    np.testing.assert_array_equal(out[hw != 0], hw[hw != 0])
    added = out[(hw == 0) & (out != 0)]
    assert added.size > 0
    assert np.all((added >= 20) & (added < 67))


@pytest.mark.parametrize("floor", [0, -0.02, float("nan"), float("inf"), 0.12])
def test_invalid_d405_floor_is_rejected(floor):
    cal = Calibration.from_params(395.16196, 0.017882943)
    with pytest.raises(ValueError, match="stereo_z_min_m"):
        DepthRangeImprover(cal, stereo_z_min_m=floor)


def test_backend_selection_and_missing_library():
    cal = Calibration.from_params(383.0, 0.095)
    with pytest.raises(ValueError, match="backend"):
        DepthRangeImprover(cal, backend="unknown")
    with pytest.raises(RuntimeError, match="Cannot load CUDA library"):
        DepthRangeImprover(cal, backend="cuda", cuda_library_path="/nonexistent/depth.so")


def test_calibration_override_updates_automatic_limits():
    cal = Calibration.from_params(395.16196, 0.017882943)
    rng = np.random.default_rng(9)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-140] = left[:, 140:]
    hw = np.zeros_like(left, dtype=np.uint16)
    imp = DepthRangeImprover(Calibration.from_params(383.0, 0.095), fill_holes=False)
    out = imp.process(left, right, hw, calibration=cal)
    expected = DepthRangeImprover(cal, fill_holes=False).process(left, right, hw)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("crop", [(-1, 0, 20, 20), (0, 0, 0, 10),
                                  (0, 0, 641, 480), (0.5, 0, 10, 10)])
def test_invalid_crop_is_rejected(crop):
    cal = Calibration.from_params(395.16196, 0.017882943)
    ir = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="crop_region"):
        DepthRangeImprover(cal, crop_region=crop).process(
            ir, ir, np.zeros_like(ir, dtype=np.uint16))


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
