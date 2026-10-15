#!/bin/bash
# Build the CUDA close-range library (drop-in librs_depth_range.so).
# Uses the user-space toolkit at $CUDA_ROOT (default /tmp/opencode/cuda-root).
set -e
CUDA_ROOT="${CUDA_ROOT:-/tmp/opencode/cuda-root/usr/local/cuda-13.2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build"
"$CUDA_ROOT/bin/nvcc" -arch=sm_120 -O3 -shared -Xcompiler -fPIC \
  "$HERE/rs_depth_cuda.cu" -o "$HERE/build/librs_depth_range.so"
echo "built: $HERE/build/librs_depth_range.so"
