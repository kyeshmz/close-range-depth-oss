# close-range-depth-oss

Experimental CPU + CUDA close-range stereo for RealSense D400 cameras on
**x86_64 + discrete CUDA**. This is not a validated replacement for the
vendor's Improved Close Range Depth library.

The vendor `.deb` (`librealsense2-enhanced-depth_1.0.45_arm64.deb`) is
Jetson/`aarch64`-only and closed-source. This repo implements a subset of
its public Python API and retains its four C entry points and metadata
layout, with a separate implementation:
downscaled stereo over the near disparity band → median → merge with
hardware depth → hole-fill from neighboring close depths.

## Layout

* `rs_depth/` — `calibration.py`, `metadata.py`, `improver.py`
  (vendor-compatible `DepthRangeImprover(cal).process(irL, irR, depth_mm)`)
* `cuda/rs_depth_cuda.cu` — `librs_depth_range.so` (census-WTA stereo),
  `cuda/build.sh`. The project-specific `rs_depth_range_configure` extension
  sets minimum depth and median filtering without changing the original ABI.
* `rs_depth/_cuda.py` — shared ctypes bindings, used by the public API and
  comparison harness; metadata layout is checked against the compiled C header.
* `validate/jetson_capture.py` — Jetson/vendor-side capture (raw IR pair +
  HW depth + vendor output → `.npz` dataset)
* `validate/compare.py` — portable scoring (CSV metrics + HW|vendor|ours PNGs)
* `tests/` — CPU, CUDA-vs-CPU, and harness offline tests

## Correctness fixes and historical results

The earlier D405 recovery numbers came from the **CPU** implementation.
The separately reported 0.2 ms timing came from a **synthetic CUDA-only**
test without the Python fill stage. Combining these as one quality/speed
result was incorrect. Neither establishes current end-to-end performance.

The original CUDA floor was 120 mm, which prevented it from recovering
anything below the D405's 67 mm threshold. The comparison script also had
an incorrect metadata layout and an untyped handle destructor. Those paths
have been fixed; previous real-scene CUDA comparisons must not be treated
as validated results.

Both backends now:

* Preserve **every nonzero hardware depth**, including far measurements.
* Fill only zeros inside the selected ROI and depth band.
* Default to a 20 mm floor for baselines below 30 mm, otherwise 120 mm.
  This is an overridable preset, not camera identification or a physical
  minimum-range guarantee.
* Share optional CPU speckle cleanup and hole filling. These stages no
  longer erase hardware measurements or add depths outside the band.

Regression tests include a synthetic D405-like plane at approximately
50 mm, both backend paths, strided arrays, ROI bounds, and recovered-only
agreement metrics. They are not a substitute for measured-distance tests.

## Shortcomings (read before using)

1. **No ground-truth validation.** Everything is scored against the vendor
   output as reference, not measured geometry. More nonzero pixels and
   vendor agreement do not establish accuracy. Known-distance targets,
   occlusions, and motion still need testing after these fixes.
2. **Coverage/precision trade-off unresolved.** Telea-style fill can smear
   or invent depths. Low image texture does not prove that the vendor uses
   interpolation. Disable filling with `fill_holes=False` to inspect the
   stereo-only output.
3. **Matcher gap.** CPU uses StereoSGBM, CUDA uses census-WTA with no
   left-right consistency check. Cross-checked on synthetic texture only.
4. **Not an all-GPU pipeline.** The native `.so` has no hole-fill stage;
   the Python API runs optional filling on the CPU for either backend.
   The build targets `sm_120` only; the user-space toolkit path
   (`/tmp/opencode/cuda-root`) is hardcoded as default in `cuda/build.sh`.
5. **Limited camera coverage.** Real-scene exploration used one D405;
   the corrected CUDA path and D435-class geometry have synthetic tests only.
6. **Capture dataset not included** (`runs/` is git-ignored, 184 MB);
   reproduce with `validate/jetson_capture.py` on a Jetson + D400.

## Python API

```python
from rs_depth import Calibration, DepthRangeImprover

cal = Calibration.from_params(focal_length_px=395.16, baseline_m=0.01788)
with DepthRangeImprover(cal, backend="cuda", stereo_z_min_m=0.02,
                        fill_holes=False) as improver:
    improved_mm = improver.process(ir_left, ir_right, hardware_depth_mm)
```

Use calibration from your active camera profile, not these example values.
Inputs must be synchronized, rectified `uint8` IR images and matching
`uint16` depth in **millimetres** (not raw Z16 units unless the scale is 1 mm).
The default backend is `cpu`. Selecting `cuda` never silently falls back;
loading or processing errors raise `RuntimeError`. Set
`RS_DEPTH_CUDA_LIBRARY` or `cuda_library_path` for a nondefault library path.
Handles are reused and freed by `close()` or a context manager. Do not share
one improver across simultaneous calls.

## Build and test

Requires NumPy, OpenCV, and pytest. CUDA builds also require an NVIDIA GPU,
compatible driver, CUDA toolkit, and host compiler.

```bash
python3 -m pytest tests/test_improver.py tests/test_compare.py tests/test_bindings.py -q
CUDA_ROOT=/path/to/cuda bash cuda/build.sh
python3 -m pytest tests/test_cuda.py -q
# Jetson: python3 validate/jetson_capture.py --out runs/x --frames 200
python3 validate/compare.py --data runs/x --out runs/cmp --backend both --zmin 0.02 --no-fill
```

GPU tests are skipped if no built library exists. If your build links to a
shared CUDA runtime, add its library directory to `LD_LIBRARY_PATH`.
Comparison requires `--backend cuda` or `--backend both` to test CUDA;
requested backend errors stop the run instead of silently omitting scores.
