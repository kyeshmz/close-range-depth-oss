"""CUDA-vs-CPU cross-validation — loads build/librs_depth_range.so via ctypes.

Checks: the 4 vendor ABI symbols exist, the .so runs on the RTX 5090, and
CUDA output agrees with the CPU prototype on identical synthetic scenes.
"""

import ctypes
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_depth import Calibration, DepthRangeImprover  # noqa: E402
from rs_depth._cuda import CFrameMetadata as FrameMeta, load_library, last_error

SO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "cuda", "build", "librs_depth_range.so")


def load():
    path = os.environ.get("RS_DEPTH_CUDA_LIBRARY", SO)
    if not os.path.exists(path):
        pytest.skip("Build cuda/build.sh to run GPU tests")
    return load_library(path)


def cuda_process(lib, cal, left, right, hw, scale=0.35):
    meta = FrameMeta(width=left.shape[1], height=left.shape[0])
    h = lib.rs_depth_range_create(float(cal.focal_length_px),
                                  float(cal.baseline_m), 0, float(scale),
                                  -1, -1, -1, -1)
    assert h, lib.rs_depth_range_last_error(None)
    out = np.zeros_like(hw)
    try:
        rc = lib.rs_depth_range_process(
            h, left.ctypes.data, right.ctypes.data,
            hw.ctypes.data, out.ctypes.data, ctypes.byref(meta))
        err = last_error(lib, h)
    finally:
        lib.rs_depth_range_destroy(h)
    assert rc == 0, err
    return out


def scene(seed=7, w=640, h=480, d=183):
    # F*B = 36.385 (D435-like 640x480): d=183 -> z ~= 199mm < T = 346mm.
    # At scale 0.35 the scaled disparity (~64) fits the default search band.
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :w - d] = left[:, d:]
    hw = np.zeros((h, w), dtype=np.uint16)
    hw[:, :20] = 1500
    return left, right, hw


def test_abi_and_fill():
    lib = load()
    cal = Calibration.from_params(383.0, 0.095)  # F*B=36.385, T=346mm
    left, right, hw = scene()
    out = cuda_process(lib, cal, left, right, hw)
    assert out.shape == hw.shape and out.dtype == np.uint16
    inner = out[10:-10, 183 + 10:-10]
    valid = inner[inner > 0]
    assert valid.size > 0.4 * inner.size, f"only {valid.size} px"
    med = float(np.median(valid))
    assert abs(med - 199) < 0.20 * 199, f"median {med:.0f}mm"
    assert np.array_equal(out[:, :20], hw[:, :20])


def test_agrees_with_cpu():
    lib = load()
    cal = Calibration.from_params(383.0, 0.095)
    cpu = DepthRangeImprover(cal, fill_holes=False)
    for seed in (7, 21):
        left, right, hw = scene(seed=seed)
        ref = cpu.process(left, right, hw)
        got = cuda_process(lib, cal, left, right, hw)
        both = (ref > 0) & (got > 0)
        assert both.sum() > 0.3 * ref.size, "too little overlap"
        rel = np.abs(ref[both].astype(float) - got[both].astype(float)
                     ) / np.maximum(ref[both].astype(float), 1.0)
        assert (rel < 0.10).mean() > 0.85, (
            f"median rel-err {np.median(rel):.3f}")
        only_cuda = ((got > 0) & (ref == 0)).sum() / ref.size
        only_cpu = ((ref > 0) & (got == 0)).sum() / ref.size
        assert only_cuda < 0.10 and only_cpu < 0.10, (
            f"validity mismatch cuda-only {only_cuda:.3f} "
            f"cpu-only {only_cpu:.3f}")


def d405_scene():
    cal = Calibration.from_params(395.16196, 0.017882943)
    return cal, scene(seed=9, d=140)


def test_legacy_c_api_recovers_d405_close_depth():
    lib = load()
    cal, (left, right, hw) = d405_scene()
    out = cuda_process(lib, cal, left, right, hw)
    valid = out[20:-20, 400:-20]
    valid = valid[valid > 0]
    assert valid.size > 50000
    assert abs(float(np.median(valid)) - 50.5) < 2
    assert np.all((valid >= 20) & (valid < 67))


@pytest.mark.parametrize("fill", [False, True])
def test_public_cuda_backend_preserves_hardware_and_handles_strides(fill):
    load()
    cal, (left, right, hw) = d405_scene()
    hw[:, 400:450] = 500
    hw[:, 450:500] = 35
    hw[240, 550] = 65535
    left, right, hw = (np.repeat(a, 2, axis=1)[:, ::2] for a in (left, right, hw))
    assert not left.flags.c_contiguous
    with DepthRangeImprover(cal, backend="cuda", fill_holes=fill) as imp:
        out = imp.process(left, right, hw)
        repeated = imp.process(left, right, hw)
        np.testing.assert_array_equal(out, repeated)
        np.testing.assert_array_equal(out[hw != 0], hw[hw != 0])
        added = out[(hw == 0) & (out > 0)]
        assert added.size > 10000
        assert np.all((added >= 20) & (added < 67))


def test_cuda_configuration_and_reallocation():
    load()
    cal, (left, right, hw) = d405_scene()
    with DepthRangeImprover(cal, backend="cuda", stereo_z_min_m=0.04,
                           use_median_filter=False, fill_holes=False) as imp:
        for shape in ((480, 640), (240, 640), (480, 640)):
            h, w = shape
            out = imp.process(left[:h, :w], right[:h, :w], hw[:h, :w])
            assert out.shape == shape
            added = out[(hw[:h, :w] == 0) & (out > 0)]
            assert added.size > 10000
            assert np.all((added >= 40) & (added < 67))


def test_cuda_crop_and_calibration_override():
    load()
    cal, (left, right, hw) = d405_scene()
    roi = np.zeros(hw.shape, dtype=bool)
    roi[100:300, 410:600] = True
    initial = Calibration.from_params(383.0, 0.095)
    with DepthRangeImprover(initial, backend="cuda", crop_region=(410, 100, 600, 300)) as imp:
        out = imp.process(left, right, hw, calibration=cal)
        np.testing.assert_array_equal(out[~roi], hw[~roi])
        assert np.count_nonzero(out[roi]) > 10000


def test_native_invalid_settings_return_errors():
    lib = load()
    assert not lib.rs_depth_range_create(0, 0.02, 67, 0.35, -1, -1, -1, -1)
    assert "invalid calibration" in last_error(lib, None)
    h = lib.rs_depth_range_create(395, 0.01788, 67, 0.35, -1, -1, -1, -1)
    assert h
    try:
        assert lib.rs_depth_range_configure(h, 0.12, 1) == -1
        assert "invalid minimum depth" in last_error(lib, h)
        assert lib.rs_depth_range_configure(h, 0.02, 1) == 0
        assert lib.rs_depth_range_destroy.argtypes == [ctypes.c_void_p]
        assert lib.rs_depth_range_destroy.restype is None
    finally:
        lib.rs_depth_range_destroy(h)


def test_cuda_processing_failure_is_reported():
    load()
    cal = Calibration.from_params(383.0, 0.095)
    ir = np.zeros((120, 160), dtype=np.uint8)
    hw = np.zeros_like(ir, dtype=np.uint16)
    with DepthRangeImprover(cal, backend="cuda") as imp:
        with pytest.raises(RuntimeError, match="disparity search"):
            imp.process(ir, ir, hw)


def test_compare_cli_scores_cuda_instead_of_silently_skipping(tmp_path):
    import csv
    import subprocess
    from test_compare import make_capture

    load()
    data = tmp_path / "capture"
    make_capture(data)
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(SO), "../../validate/compare.py"),
         "--data", str(data), "--out", str(out), "--backend", "both", "--no-fill"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (out / "metrics.csv").open() as fh:
        row = next(csv.DictReader(fh))
    assert row["backend"] == "cpu"
    assert float(row["cuda_rescued_ours"]) > 50000
    assert float(row["cuda_recovered_agree10"]) > 0.9
    assert "backend=cuda" in result.stdout


if __name__ == "__main__":
    test_abi_and_fill()
    print("cuda fill ok")
    test_agrees_with_cpu()
    print("cuda/cpu agreement ok")
    print("ALL CUDA TESTS PASSED")
