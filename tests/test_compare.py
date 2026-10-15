"""Test compare.py end-to-end on a fake Jetson capture (no camera)."""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rs_depth import Calibration, DepthRangeImprover  # noqa: E402


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


if __name__ == "__main__":
    test_compare_offline()
    print("COMPARE TEST PASSED")
