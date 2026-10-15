"""Capture side-by-side validation data on the Jetson (vendor stack only).

Runs the D400, saves per-frame raw inputs + vendor improved output so the
portable `compare.py` can score our reimplementation against it offline.

Each frame saves `<out>/frame_XXXX.npz` with:
  ir_left, ir_right (H,W u8), hw (H,W u16 mm), vendor (H,W u16 mm),
  fx, baseline_m, depth_scale, threshold_mm, t_hw_ms, t_vendor_ms

Usage (on Jetson):
  python3 jetson_capture.py --out runs/d405_close --frames 200
"""

import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/jetson_close")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import pyrealsense2 as rs
    from rs_depth import (Calibration, DepthRangeImprover, FrameMetadata,
                          discover_backends)

    print("backends:", discover_backends())
    os.makedirs(args.out, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height,
                      rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.width, args.height,
                      rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height,
                      rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    try:
        ir1 = profile.get_stream(rs.stream.infrared, 1
                                 ).as_video_stream_profile()
        ir2 = profile.get_stream(rs.stream.infrared, 2)
        cal = Calibration.from_sdk(ir1.get_intrinsics(),
                                   ir1.get_extrinsics_to(ir2))
        try:
            scale = (profile.get_device().first_depth_sensor()
                     .get_depth_scale())
        except Exception:
            scale = 0.001
        improver = DepthRangeImprover(cal)
        t = improver.min_z_threshold_mm
        print(f"threshold {t} mm, depth_scale {scale}")

        # Warmup (AE/exposure settle)
        for _ in range(30):
            pipe.wait_for_frames()

        for i in range(args.frames):
            f = pipe.wait_for_frames()
            il = np.asanyarray(f.get_infrared_frame(1).get_data())
            ir = np.asanyarray(f.get_infrared_frame(2).get_data())
            hw = (np.asanyarray(f.get_depth_frame().get_data())
                  * scale * 1000.0).astype(np.uint16)
            try:
                meta = FrameMetadata.from_rs2_frameset(f)
            except Exception:
                meta = None

            t0 = time.perf_counter()
            il.copy()  # touch pages; HW depth is free (came from camera)
            t_hw = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            try:
                out = improver.process(il, ir, hw, metadata=meta)
            except TypeError:
                out = improver.process(il, ir, hw)
            t_vendor = (time.perf_counter() - t0) * 1000.0

            np.savez_compressed(
                os.path.join(args.out, f"frame_{i:04d}.npz"),
                ir_left=il, ir_right=ir, hw=hw, vendor=np.asanyarray(out),
                fx=np.float32(cal.focal_length_px),
                baseline_m=np.float32(cal.baseline_m),
                threshold_mm=np.int32(t),
                t_hw_ms=np.float32(t_hw), t_vendor_ms=np.float32(t_vendor))
            if i % 20 == 0:
                print(f"frame {i}/{args.frames}", flush=True)
    finally:
        pipe.stop()
    print("saved to", args.out)


if __name__ == "__main__":
    main()
