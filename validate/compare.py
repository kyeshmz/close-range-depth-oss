"""Score our reimplementation against Jetson captures (portable, our stack).

Reads `frame_*.npz` from `jetson_capture.py`, runs the requested backend(s),
and writes CSV metrics plus HW | vendor | ours PNGs. Requested CUDA failures
are fatal, not silently skipped. Vendor agreement is not ground truth.

Usage:
  python3 compare.py --data runs/jetson_close --out runs/compare
"""

import argparse
import os
import time

import numpy as np

OWN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(hw, vendor, ours, t):
    m = {}
    for name, d in (("hw", hw), ("vendor", vendor), ("ours", ours)):
        band = (d > 0) & (d < t)
        m[f"{name}_band_px"] = int(band.sum())
    m["rescued_vendor"] = int(((hw == 0) & (vendor > 0)
                               & (vendor < t)).sum())
    m["rescued_ours"] = int(((hw == 0) & (ours > 0) & (ours < t)).sum())
    both = (vendor > 0) & (ours > 0) & (vendor < t) & (ours < t)
    m["overlap_px"] = int(both.sum())
    if both.sum():
        rel = (np.abs(vendor[both].astype(float) - ours[both].astype(float))
               / np.maximum(vendor[both].astype(float), 1.0))
        m["median_rel_err"] = float(np.median(rel))
        m["agree10"] = float((rel < 0.10).mean())
    else:
        m["median_rel_err"] = float("nan")
        m["agree10"] = float("nan")
    rv = (hw == 0) & (vendor > 0) & (vendor < t)
    ro = (hw == 0) & (ours > 0) & (ours < t)
    recovered_overlap = rv & ro
    n = int(recovered_overlap.sum())
    m["recovered_overlap_px"] = n
    m["rescue_recall_vs_vendor"] = n / int(rv.sum()) if rv.any() else float("nan")
    m["rescue_support_precision_vs_vendor"] = (
        n / int(ro.sum()) if ro.any() else float("nan"))
    if n:
        error = np.abs(vendor[recovered_overlap].astype(float)
                       - ours[recovered_overlap].astype(float))
        m["recovered_median_abs_err_mm"] = float(np.median(error))
        m["recovered_agree10"] = float(
            (error / vendor[recovered_overlap] < 0.10).mean())
    else:
        m["recovered_median_abs_err_mm"] = float("nan")
        m["recovered_agree10"] = float("nan")
    return m


def side_by_side(hw, vendor, ours, t, path, backend="cpu", cuda=None):
    import cv2
    vis = []
    columns = [("HW", hw), ("vendor", vendor), (backend, ours)]
    if cuda is not None:
        columns.append(("cuda", cuda))
    for label, d in columns:
        clip = np.minimum(d.astype(np.float32), float(t))
        color = cv2.applyColorMap((clip / max(t, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        color[d == 0] = 0
        header = np.zeros((24, d.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        vis.append(np.vstack((header, color)))
    if not cv2.imwrite(path, np.hstack(vis)):
        raise OSError(f"Could not write {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--zmin", type=float, default=None,
                    help="override stereo floor in metres (default: baseline preset)")
    ap.add_argument("--backend", choices=("cpu", "cuda", "both"), default="cpu")
    ap.add_argument("--cuda-library", default=None)
    ap.add_argument("--no-fill", action="store_true",
                    help="disable the shared CPU hole-fill stage for both backends")
    args = ap.parse_args()
    if args.max_frames < 0:
        ap.error("--max-frames must be nonnegative")
    if not os.path.isdir(args.data):
        ap.error(f"Capture directory does not exist: {args.data}")
    os.makedirs(args.out, exist_ok=True)

    import sys
    sys.path.insert(0, OWN_ROOT)
    from rs_depth import Calibration, DepthRangeImprover

    files = sorted(f for f in os.listdir(args.data) if f.endswith(".npz"))
    if args.max_frames:
        files = files[:args.max_frames]

    if not files:
        ap.error("No capture frames found")
    backends = ("cpu", "cuda") if args.backend == "both" else (args.backend,)
    improvers = {}
    rows = []
    try:
        for fn in files:
            with np.load(os.path.join(args.data, fn), allow_pickle=False) as z:
                il, ir, hw = z["ir_left"], z["ir_right"], z["hw"]
                vendor = z["vendor"]
                t = int(z["threshold_mm"])
                cal = Calibration.from_params(float(z["fx"]), float(z["baseline_m"]))
                vendor_ms = float(z["t_vendor_ms"])
            if vendor.shape != hw.shape or vendor.dtype != np.uint16:
                raise ValueError(f"{fn}: vendor depth must match hardware shape and dtype")
            row = {"frame": fn, "backend": backends[0], "t_vendor_ms": vendor_ms}
            outputs = {}
            for backend in backends:
                key = (backend, cal.focal_length_px, cal.baseline_m, t)
                if key not in improvers:
                    improvers[key] = DepthRangeImprover(
                        cal, min_z_threshold_mm=t, stereo_z_min_m=args.zmin,
                        backend=backend, cuda_library_path=args.cuda_library,
                        fill_holes=not args.no_fill)
                start = time.perf_counter()
                outputs[backend] = improvers[key].process(il, ir, hw)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prefix = "cuda_" if args.backend == "both" and backend == "cuda" else ""
                row[prefix + "t_ours_ms"] = elapsed_ms
                row.update({prefix + k: v for k, v in
                            metrics(hw, vendor, outputs[backend], t).items()})
            rows.append(row)
            side_by_side(hw, vendor, outputs[backends[0]], t,
                         os.path.join(args.out, fn.replace(".npz", ".png")),
                         backend=backends[0],
                         cuda=outputs.get("cuda") if args.backend == "both" else None)
    finally:
        for improver in improvers.values():
            improver.close()

    import csv
    with open(os.path.join(args.out, "metrics.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for backend in backends:
        prefix = "cuda_" if args.backend == "both" and backend == "cuda" else ""
        agree = [r[prefix + "recovered_agree10"] for r in rows
                 if np.isfinite(r[prefix + "recovered_agree10"])]
        mean = float(np.mean(agree)) if agree else float("nan")
        print(f"backend={backend} frames={len(rows)} recovered_agree10={mean:.3f} "
              "(vendor agreement, not ground-truth accuracy)")


if __name__ == "__main__":
    main()
