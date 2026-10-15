"""CUDA-vs-CPU cross-validation — loads build/librs_depth_range.so via ctypes.

Checks: the 4 vendor ABI symbols exist, the .so runs on the RTX 5090, and
CUDA output agrees with the CPU prototype on identical synthetic scenes.
"""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_depth import Calibration, DepthRangeImprover  # noqa: E402

SO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "cuda", "build", "librs_depth_range.so")


class StreamMeta(ctypes.Structure):
    _fields_ = [("frame_number", ctypes.c_int64),
                ("timestamp_us", ctypes.c_double),
                ("arrival_time_us", ctypes.c_double),
                ("sensor_timestamp_us", ctypes.c_double),
                ("backend_timestamp_us", ctypes.c_double),
                ("exposure_us", ctypes.c_int32),
                ("gain", ctypes.c_int32),
                ("auto_exposure", ctypes.c_int32),
                ("laser_power", ctypes.c_int32),
                ("emitter_mode", ctypes.c_int32),
                ("white_balance", ctypes.c_int32),
                ("temperature", ctypes.c_float),
                ("actual_fps", ctypes.c_float),
                ("brightness", ctypes.c_int32),
                ("contrast", ctypes.c_int32),
                ("saturation", ctypes.c_int32)]


class FrameMeta(ctypes.Structure):
    _fields_ = [("ir", StreamMeta), ("depth", StreamMeta),
                ("color", StreamMeta), ("ir_format", ctypes.c_char * 8),
                ("width", ctypes.c_int32), ("height", ctypes.c_int32)]


def load():
    lib = ctypes.CDLL(SO)
    lib.rs_depth_range_create.argtypes = [ctypes.c_float, ctypes.c_float,
                                         ctypes.c_int, ctypes.c_float,
                                         ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int]
    lib.rs_depth_range_create.restype = ctypes.c_void_p
    lib.rs_depth_range_destroy.argtypes = [ctypes.c_void_p]
    lib.rs_depth_range_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.POINTER(FrameMeta)]
    lib.rs_depth_range_process.restype = ctypes.c_int
    lib.rs_depth_range_last_error.argtypes = [ctypes.c_void_p]
    lib.rs_depth_range_last_error.restype = ctypes.c_char_p
    return lib


def cuda_process(lib, cal, left, right, hw, scale=0.35):
    meta = FrameMeta(width=left.shape[1], height=left.shape[0])
    h = lib.rs_depth_range_create(float(cal.focal_length_px),
                                  float(cal.baseline_m), 0, float(scale),
                                  -1, -1, -1, -1)
    assert h, lib.rs_depth_range_last_error(None)
    out = np.zeros_like(hw)
    rc = lib.rs_depth_range_process(
        h, left.ctypes.data, right.ctypes.data,
        hw.ctypes.data, out.ctypes.data, ctypes.byref(meta))
    err = lib.rs_depth_range_last_error(h)
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
    cpu = DepthRangeImprover(cal)  # defaults: scale 0.35, floor 0.12m
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


if __name__ == "__main__":
    test_abi_and_fill()
    print("cuda fill ok")
    test_agrees_with_cpu()
    print("cuda/cpu agreement ok")
    print("ALL CUDA TESTS PASSED")
