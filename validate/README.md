# Side-by-side validation (Jetson = reference rig)

## 1. On the Jetson (has camera + vendor `rs_depth`)

```bash
python3 validate/jetson_capture.py --out runs/d405_close --frames 200
```

Use a range appropriate for the camera (e.g. roughly 30–60 mm for testing
the D405's blind band, or 120–500 mm for longer-baseline D400 devices).
Include textured planes, edges, and actual targets. Saves one `.npz` per
frame: raw IR pair + HW depth + vendor output + calibration + timings.

## 2. Copy back and score (this box or anywhere)

```bash
scp -r jetson:runs/d405_close runs/
python3 validate/compare.py --data runs/d405_close --out runs/compare --backend both --zmin 0.02 --no-fill
```

The default backend is CPU only. `--backend cuda` requests CUDA only;
`--backend both` tests both and produces **HW | vendor | CPU | CUDA**
images. All columns share a color scale, with zero/invalid pixels black.
CUDA uses the same corrected bindings as the public API. A requested CUDA
failure is fatal, not silently skipped. `--no-fill` isolates stereo output;
otherwise both backends use the shared CPU fill stage.

`metrics.csv` includes:

* Band pixel counts and rescued counts (`HW == 0`, output nonzero and `< T`).
* `rescue_recall_vs_vendor` and `rescue_support_precision_vs_vendor`: overlap
  of the two rescue masks. These measure reference support, not physical truth.
* `recovered_median_abs_err_mm` and `recovered_agree10`: vendor agreement
  only on shared recovered pixels, excluding unchanged hardware measurements.
  They are NaN when no shared recovery exists.
* Legacy `median_rel_err` / `agree10`, including unchanged hardware pixels;
  these can look good even when the filter recovers nothing.
* `t_ours_ms`, including Python processing and any CPU fill. For `both`,
  CUDA fields have a `cuda_` prefix. First-frame timing includes startup.
  Vendor timings come from capture and are not a like-for-like benchmark
  across different machines or algorithms. The capture's `t_hw_ms` field
  times a memory copy, not camera/ASIC depth computation.

## What good looks like

Similar rescue counts alone do not mean the same pixels or correct depths.
Inspect rescue-mask overlap and recovered-only agreement, then validate
against known-distance targets. Vendor agreement is not ground truth.
