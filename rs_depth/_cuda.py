"""Bindings for this project's CUDA library, not the vendor binary."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class CStreamMetadata(ctypes.Structure):
    _fields_ = [
        ("frame_number", ctypes.c_int64),
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
        ("saturation", ctypes.c_int32),
    ]


class CFrameMetadata(ctypes.Structure):
    _fields_ = [
        ("ir", CStreamMetadata),
        ("depth", CStreamMetadata),
        ("color", CStreamMetadata),
        ("ir_format", ctypes.c_char * 8),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


def load_library(path=None):
    if path is None:
        path = os.environ.get("RS_DEPTH_CUDA_LIBRARY")
    if path is None:
        path = Path(__file__).resolve().parents[1] / "cuda/build/librs_depth_range.so"
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(f"Cannot load CUDA library {path}: {exc}") from exc
    try:
        lib.rs_depth_range_create.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.rs_depth_range_create.restype = ctypes.c_void_p
        lib.rs_depth_range_configure.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_int,
        ]
        lib.rs_depth_range_configure.restype = ctypes.c_int
        lib.rs_depth_range_process.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(CFrameMetadata),
        ]
        lib.rs_depth_range_process.restype = ctypes.c_int
        lib.rs_depth_range_last_error.argtypes = [ctypes.c_void_p]
        lib.rs_depth_range_last_error.restype = ctypes.c_char_p
        lib.rs_depth_range_destroy.argtypes = [ctypes.c_void_p]
        lib.rs_depth_range_destroy.restype = None
    except AttributeError as exc:
        raise RuntimeError(
            "CUDA library is outdated or is a vendor binary; rebuild cuda/build.sh"
        ) from exc
    return lib


def last_error(lib, handle):
    return lib.rs_depth_range_last_error(handle).decode("utf-8", errors="replace")


class CudaBackend:
    def __init__(self, library_path=None):
        self._handle = None
        self._config = None
        self._lib = load_library(library_path)

    def close(self):
        if self._handle is not None:
            self._lib.rs_depth_range_destroy(self._handle)
            self._handle = None
            self._config = None

    def __del__(self):
        self.close()

    def process(self, left, right, hw, cal, threshold_mm, scale, floor_m,
                use_median, crop):
        config = (cal.focal_length_px, cal.baseline_m, threshold_mm, scale,
                  floor_m, use_median, crop)
        if config != self._config:
            self.close()
            self._handle = self._lib.rs_depth_range_create(
                cal.focal_length_px, cal.baseline_m, threshold_mm, scale,
                *(crop or (-1, -1, -1, -1)))
            if not self._handle:
                self._handle = None
                raise RuntimeError(last_error(self._lib, None))
            if self._lib.rs_depth_range_configure(
                    self._handle, floor_m, int(use_median)) != 0:
                error = last_error(self._lib, self._handle)
                self.close()
                raise RuntimeError(error)
            self._config = config

        left, right, hw = (np.ascontiguousarray(a) for a in (left, right, hw))
        out = np.empty_like(hw)
        meta = CFrameMetadata(ir_format=b"y8", width=hw.shape[1], height=hw.shape[0])
        rc = self._lib.rs_depth_range_process(
            self._handle, left.ctypes.data, right.ctypes.data, hw.ctypes.data,
            out.ctypes.data, ctypes.byref(meta))
        if rc != 0:
            raise RuntimeError(last_error(self._lib, self._handle))
        return out
