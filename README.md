# close-range-depth-oss

CPU + CUDA reimplementation of RealSense `librealsense2-enhanced-depth`
(Improved Close Range Depth) for D400 cameras on **x86_64 + discrete CUDA**.

The vendor `.deb` (`librealsense2-enhanced-depth_1.0.45_arm64.deb`) is
Jetson/`aarch64`-only and closed-source. This repo clones its public API
(`Calibration`, `FrameMetadata`, `DepthRangeImprover`, the 4-symbol C ABI)
from the unpacked headers and reimplements the filter from scratch:
downscaled stereo over the near disparity band → median → merge with
hardware depth → hole-fill from neighboring close depths.

## Layout

* `rs_depth/` — `calibration.py`, `metadata.py`, `improver.py`
  (vendor-compatible `DepthRangeImprover(cal).process(irL, irR, depth_mm)`)
* `cuda/rs_depth_cuda.cu` — drop-in `librs_depth_range.so`
  (census-WTA stereo, same C ABI), `cuda/build.sh`
* `validate/jetson_capture.py` — Jetson/vendor-side capture (raw IR pair +
  HW depth + vendor output → `.npz` dataset)
* `validate/compare.py` — portable scoring (CSV metrics + HW|vendor|ours PNGs)
* `tests/` — CPU, CUDA-vs-CPU, and harness offline tests

## Result (D405, 200 close-up frames, band <67 mm)

* HW alone: ~10k band px/frame. Vendor: ~40k (+29k rescued, 42 ms on
  Jetson). Ours: +15–40k rescued depending on config, **0.2 ms/frame**
  @640x480 on RTX 5090 incl. transfers.
* Where both produce a depth: `agree10` 0.78–0.92 by config.

| config | recall of vendor set | agree10 |
|---|---|---|
| no fill | 0.12 | 0.92 |
| fill r15 | 0.32 | 0.77 |
| fill r60 | 0.58 | 0.63 |
| speckle-filter + r10 | 0.26 | 0.81 |

## Shortcomings (read before using)

1. **No ground-truth validation.** Everything is scored against the vendor
   output as reference, not measured geometry. "More close-range pixels" is
   proven; per-pixel correctness is bounded by vendor agreement (~0.8–0.9).
   A taped target at 50/100/200 mm is the real acceptance test (not done).
2. **Coverage/precision trade-off unresolved.** Telea-style fill recovers
   more pixels but smears (ours std ~17 vs vendor ~6); tight configs agree
   better but rescue less. A plane-guided fill would match the vendor's
   profile better — not implemented.
3. **Matcher gap.** CPU uses StereoSGBM, CUDA uses census-WTA with no
   left-right consistency check. Cross-checked on synthetic texture only.
4. **CUDA port lags the CPU design.** The `.so` has no fill stage yet and
   builds for `sm_120` only; the user-space toolkit path
   (`/tmp/opencode/cuda-root`) is hardcoded as default in `cuda/build.sh`.
5. **D405-tuned.** Floor/threshold defaults were validated on D405 captures
   (`T=67 mm`, floor 20 mm). D435-class geometry only has synthetic tests.
6. **Capture dataset not included** (`runs/` is git-ignored, 184 MB);
   reproduce with `validate/jetson_capture.py` on a Jetson + D400.

## Run

```bash
python3 tests/test_improver.py
LD_LIBRARY_PATH=<cuda>/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
  python3 tests/test_cuda.py
# Jetson: python3 validate/jetson_capture.py --out runs/x --frames 200
python3 validate/compare.py --data runs/x --out runs/cmp --zmin 0.02
```
