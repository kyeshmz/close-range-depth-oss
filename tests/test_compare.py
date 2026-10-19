"""Test compare.py end-to-end on a fake Jetson capture (no camera)."""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rs_depth import Calibration, DepthRangeImprover  # noqa: E402
from validate.compare import metrics


def test_compare_offline(tmp_path=None):
    import tempfile
    data = tempfile.mkdtemp(prefix="fake_jetson_")
    out = tempfile.mkdtemp(prefix="cmp_")
    rng = np.random.default_rng(5)
    cal = Calibration.from_params(383.0, 0.095)
    imp = DepthRangeImprover(cal)
    for i in range(3):
        left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :640 - 183] = left[:, 183:]
        hw = np.zeros((480, 640), dtype=np.uint16)
        vendor = imp.process(left, right, hw)  # stand-in reference
        np.savez_compressed(
            os.path.join(data, f"frame_{i:04d}.npz"),
            ir_left=left, ir_right=right, hw=hw, vendor=vendor,
            fx=np.float32(383.0), baseline_m=np.float32(0.095),
            threshold_mm=np.int32(cal.min_z_threshold_mm),
            t_hw_ms=np.float32(0.1), t_vendor_ms=np.float32(33.0))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "validate",
                                                      "compare.py"),
                        "--data", data, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(out, "metrics.csv"))
    assert len([f for f in os.listdir(out) if f.endswith(".png")]) == 3
    print(r.stdout.strip())


def test_recovered_agreement_excludes_unchanged_hardware():
    hw = np.array([[50, 0, 0]], dtype=np.uint16)
    vendor = np.array([[50, 40, 0]], dtype=np.uint16)
    ours = np.array([[50, 0, 45]], dtype=np.uint16)
    m = metrics(hw, vendor, ours, 67)
    assert m["agree10"] == 1
    assert m["recovered_overlap_px"] == 0
    assert m["rescue_recall_vs_vendor"] == 0
    assert m["rescue_support_precision_vs_vendor"] == 0
    assert np.isnan(m["recovered_agree10"])
    ours[0, 1] = 42
    m = metrics(hw, vendor, ours, 67)
    assert m["recovered_overlap_px"] == 1
    assert m["recovered_median_abs_err_mm"] == 2
    assert m["recovered_agree10"] == 1


def test_empty_capture_directory_fails_clearly(tmp_path):
    data = tmp_path / "empty"
    data.mkdir()
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "validate/compare.py"),
         "--data", str(data), "--out", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "No capture frames found" in result.stderr


def make_capture(directory):
    directory.mkdir()
    rng = np.random.default_rng(9)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-140] = left[:, 140:]
    cal = Calibration.from_params(395.16196, 0.017882943)
    hw = np.zeros_like(left, dtype=np.uint16)
    stand_in = DepthRangeImprover(cal, fill_holes=False).process(left, right, hw)
    np.savez_compressed(
        directory / "frame_0000.npz", ir_left=left, ir_right=right, hw=hw,
        vendor=stand_in, fx=cal.focal_length_px, baseline_m=cal.baseline_m,
        threshold_mm=cal.min_z_threshold_mm, t_vendor_ms=0.0)


def test_requested_cuda_is_not_silently_skipped(tmp_path):
    data = tmp_path / "capture"
    make_capture(data)
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "validate/compare.py"),
         "--data", str(data), "--out", str(tmp_path / "out"),
         "--backend", "both", "--cuda-library", str(tmp_path / "missing.so")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "Cannot load CUDA library" in result.stderr
    assert not (tmp_path / "out/metrics.csv").exists()


if __name__ == "__main__":
    test_compare_offline()
    print("COMPARE TEST PASSED")
