"""Score our reimplementation against Jetson captures (portable, our stack).

Reads `frame_*.npz` from `jetson_capture.py`, runs our CPU (+ CUDA if a
compatible `librs_depth_range.so` loads), and writes per-frame metrics CSV
plus side-by-side PNGs (HW | vendor | ours) with identical color scaling.

Usage:
  python3 compare.py --data runs/jetson_close --out runs/compare
"""

import argparse
import ctypes
import os

import numpy as np

OWN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ours_cuda():
    so = os.path.join(OWN_ROOT, "cuda", "build", "librs_depth_range.so")
    if not os.path.exists(so):
        return None
    try:
        lib = ctypes.CDLL(so)

        class Meta(ctypes.Structure):
            _fields_ = [("pad", ctypes.c_byte * 252),
                        ("ir_format", ctypes.c_char * 8),
                        ("width", ctypes.c_int32),
                        ("height", ctypes.c_int32)]

        lib.rs_depth_range_create.restype = ctypes.c_void_p
        lib.rs_depth_range_process.restype = ctypes.c_int
        return lib, Meta
    except Exception as e:
        print("cuda backend unavailable:", e)
        return None


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
    return m


def side_by_side(hw, vendor, ours, t, path):
    import cv2
    vis = []
    for d in (hw, vendor, ours):
        clip = np.minimum(d.astype(np.float32), float(t))
        vis.append(cv2.applyColorMap((clip / max(t, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO))
    cv2.imwrite(path, np.hstack(vis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--zmin", type=float, default=0.12,
                    help="stereo floor in metres (D405 needs ~0.02)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import sys
    sys.path.insert(0, OWN_ROOT)
    from rs_depth import Calibration, DepthRangeImprover

    cuda = load_ours_cuda()
    files = sorted(f for f in os.listdir(args.data) if f.endswith(".npz"))
    if args.max_frames:
        files = files[:args.max_frames]

    rows = []
    for fn in files:
        z = np.load(os.path.join(args.data, fn))
        il, ir, hw = z["ir_left"], z["ir_right"], z["hw"]
        vendor = z["vendor"]
        t = int(z["threshold_mm"])
        cal = Calibration.from_params(float(z["fx"]), float(z["baseline_m"]))

        ours = DepthRangeImprover(cal,
                                    stereo_z_min_m=args.zmin).process(
            il, ir, hw)
        row = {"frame": fn, "t_vendor_ms": float(z["t_vendor_ms"]),
               **metrics(hw, vendor, ours, t)}
        if cuda is not None:
            lib, Meta = cuda
            meta = Meta(width=il.shape[1], height=il.shape[0])
            lib.rs_depth_range_create.argtypes = [ctypes.c_float,
                                                 ctypes.c_float, ctypes.c_int,
                                                 ctypes.c_float, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
            lib.rs_depth_range_process.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Meta)]
            h = lib.rs_depth_range_create(cal.focal_length_px,
                                          cal.baseline_m, 0, 0.35,
                                          -1, -1, -1, -1)
            oc = np.zeros_like(hw)
            rc = lib.rs_depth_range_process(
                h, il.ctypes.data, ir.ctypes.data, hw.ctypes.data,
                oc.ctypes.data, ctypes.byref(meta))
            lib.rs_depth_range_destroy(h)
            if rc == 0:
                cm = metrics(hw, vendor, oc, t)
                row.update({f"cuda_{k}": v for k, v in cm.items()
                            if k in ("median_rel_err", "agree10",
                                     "rescued_ours")})
        rows.append(row)
        side_by_side(hw, vendor, ours, t,
                     os.path.join(args.out, fn.replace(".npz", ".png")))

    import csv
    with open(os.path.join(args.out, "metrics.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agree = np.array([r["agree10"] for r in rows if r["agree10"] == r["agree10"]])
    print(f"frames={len(rows)} mean_agree10={agree.mean():.3f} "
          f"median_rel_err={np.median([r['median_rel_err'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
