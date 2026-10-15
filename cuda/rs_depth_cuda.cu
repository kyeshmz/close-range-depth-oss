/* rs_depth_cuda.cu — CUDA close-range depth improver, vendor-ABI compatible.
 *
 * Pipeline mirrors rs_depth/improver.py (CPU): box-downscale the IR pair,
 * census-5x5 + winner-take-all stereo over the near disparity band,
 * disparity -> depth, 5x5 median, nearest-upscale merged with HW depth.
 *
 * Builds a drop-in `librs_depth_range.so` (same 4 C symbols, same structs).
 * No librealsense dependency. Requires CUDA 12.8+ for sm_120 (Blackwell).
 */

#include "rs_depth_abi.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#define CUDA_OK(call, h, msg) do {                                    \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
      snprintf((h)->err, sizeof((h)->err), "%s: %s", (msg),           \
               cudaGetErrorString(_e));                               \
      return -1;                                                      \
    }                                                                 \
  } while (0)

struct RsDepthRangeImpl {
  float focal_px = 388.0f;
  float baseline_m = 0.095f;
  int min_z_mm = 0;
  float scale = 0.35f;
  int crop[4] = {-1, -1, -1, -1};
  float zmin_m = 0.12f;
  int uniqueness = 10;  /* percent, same convention as OpenCV SGBM */

  int w = 0, h = 0, ws = 0, hs = 0;  /* cached geometry */
  uint8_t *d_il = nullptr, *d_ir = nullptr;    /* scaled IR */
  uint8_t *d_fil = nullptr, *d_fir = nullptr;  /* full-res staging */
  uint32_t *d_cenL = nullptr, *d_cenR = nullptr;
  float *d_z = nullptr, *d_zmed = nullptr;
  uint16_t *d_hw = nullptr, *d_out = nullptr;
  char err[256] = {0};
};

static int auto_threshold(float f, float b) {
  return (int)(f * b * 1000.0f / 105.0f);
}

static void free_bufs(RsDepthRangeImpl *h) {
  cudaFree(h->d_il); cudaFree(h->d_ir);
  cudaFree(h->d_fil); cudaFree(h->d_fir);
  cudaFree(h->d_cenL); cudaFree(h->d_cenR);
  cudaFree(h->d_z); cudaFree(h->d_zmed);
  cudaFree(h->d_hw); cudaFree(h->d_out);
  h->d_il = h->d_ir = h->d_fil = h->d_fir = nullptr;
  h->d_cenL = h->d_cenR = nullptr;
  h->d_z = h->d_zmed = nullptr;
  h->d_hw = h->d_out = nullptr;
  h->w = h->h = 0;
}

/* --- kernels --------------------------------------------------------- */

__global__ void k_downsample(const uint8_t *in, int W, int H,
                             uint8_t *out, int ws, int hs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ws || y >= hs) return;
  int x0 = x * W / ws, x1 = (x + 1) * W / ws;
  int y0 = y * H / hs, y1 = (y + 1) * H / hs;
  int sum = 0, n = 0;
  for (int yy = y0; yy < y1; ++yy)
    for (int xx = x0; xx < x1; ++xx) { sum += in[yy * W + xx]; ++n; }
  out[y * ws + x] = (uint8_t)(sum / (n ? n : 1));
}

__device__ inline uint8_t pxc(const uint8_t *img, int W, int H, int x, int y) {
  x = x < 0 ? 0 : (x >= W ? W - 1 : x);
  y = y < 0 ? 0 : (y >= H ? H - 1 : y);
  return img[y * W + x];
}

__global__ void k_census(const uint8_t *img, int W, int H, uint32_t *desc) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  uint8_t c = img[y * W + x];
  uint32_t d = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      if (!dx && !dy) continue;
      d = (d << 1) | (pxc(img, W, H, x + dx, y + dy) >= c ? 1u : 0u);
    }
  desc[y * W + x] = d;
}

__global__ void k_match(const uint32_t *L, const uint32_t *R, int ws, int hs,
                        int numDisp, int uniqueness,
                        float f_scaled, float baseline_m, float *z) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ws || y >= hs) return;
  float out = 0.0f;
  if (x >= numDisp) {
    uint32_t cl = L[y * ws + x];
    int best1 = 1 << 30, best2 = 1 << 30, bestd = 0;
    int lim = numDisp < x + 1 ? numDisp : x + 1;
    for (int d = 0; d < lim; ++d) {
      int c = __popc(cl ^ R[y * ws + x - d]);
      if (c < best1) { best2 = best1; best1 = c; bestd = d; }
      else if (c < best2) { best2 = c; }
    }
    if (bestd > 0 && (best2 - best1) * 100 > best1 * uniqueness)
      out = f_scaled * baseline_m * 1000.0f / (float)bestd;
  }
  z[y * ws + x] = out;
}

__global__ void k_median5(const float *in, int W, int H, float *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  float v[25];
  int n = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      int xx = x + dx, yy = y + dy;
      v[n++] = (xx < 0 || yy < 0 || xx >= W || yy >= H) ? 0.0f
                                                       : in[yy * W + xx];
    }
  for (int i = 1; i < 25; ++i) {  /* insertion sort, 25 elems */
    float t = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > t) { v[j + 1] = v[j]; --j; }
    v[j + 1] = t;
  }
  out[y * W + x] = v[12];
}

__global__ void k_merge(const float *z, int ws, int hs,
                        const uint16_t *hw, uint16_t *out, int W, int H,
                        int thresh_mm, int floor_mm,
                        int cx1, int cy1, int cx2, int cy2, int use_crop) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  uint16_t keep = hw[y * W + x];
  int in_roi = !use_crop || (x >= cx1 && x < cx2 && y >= cy1 && y < cy2);
  if (in_roi) {
    int sx = x * ws / W, sy = y * hs / H;
    float zc = z[sy * ws + sx];
    int zi = (int)(zc + 0.5f);
    int hwi = (int)keep;
    if (zc > 0.0f && zi >= floor_mm && zi < thresh_mm &&
        (hwi == 0 || hwi >= thresh_mm))
      keep = (uint16_t)zi;
  }
  out[y * W + x] = keep;
}

/* --- C API ------------------------------------------------------------ */

extern "C" {

RsDepthRange rs_depth_range_create(float focal_length_px, float baseline_m,
                                   int min_z_threshold, float scale_factor,
                                   int crop_x1, int crop_y1,
                                   int crop_x2, int crop_y2) {
  auto *h = new RsDepthRangeImpl();
  h->focal_px = focal_length_px;
  h->baseline_m = baseline_m;
  h->min_z_mm = min_z_threshold > 0 ? min_z_threshold
                                    : auto_threshold(focal_length_px, baseline_m);
  h->scale = (scale_factor > 0.0f && scale_factor <= 1.0f) ? scale_factor : 0.35f;
  h->crop[0] = crop_x1; h->crop[1] = crop_y1;
  h->crop[2] = crop_x2; h->crop[3] = crop_y2;
  return h;
}

void rs_depth_range_destroy(RsDepthRange h) {
  if (!h) return;
  free_bufs((RsDepthRangeImpl *)h);
  delete (RsDepthRangeImpl *)h;
}

const char *rs_depth_range_last_error(RsDepthRange h) {
  if (!h) return "invalid handle";
  return ((RsDepthRangeImpl *)h)->err;
}

int rs_depth_range_process(RsDepthRange h, const uint8_t *ir_left,
                           const uint8_t *ir_right, const uint16_t *depth_mm,
                           uint16_t *depth_out, const FrameMetadata *meta) {
  auto *H = (RsDepthRangeImpl *)h;
  if (!H || !ir_left || !ir_right || !depth_mm || !depth_out || !meta) {
    if (H) snprintf(H->err, sizeof(H->err), "null argument");
    return -1;
  }
  int W = meta->width, Ht = meta->height;
  if (W <= 0 || Ht <= 0) {
    snprintf(H->err, sizeof(H->err), "bad metadata dims %dx%d", W, Ht);
    return -1;
  }
  int ws = H->scale >= 1.0f ? W : (int)(W * H->scale + 0.5f);
  int hs = H->scale >= 1.0f ? Ht : (int)(Ht * H->scale + 0.5f);
  if (ws < 1) ws = 1;
  if (hs < 1) hs = 1;
  float f_scaled = H->focal_px * ((float)ws / (float)W);
  int numDisp = (int)ceil(f_scaled * H->baseline_m / H->zmin_m / 16.0f) * 16;
  if (numDisp < 16) numDisp = 16;
  if (numDisp >= ws) {
    snprintf(H->err, sizeof(H->err),
             "disparity search %d exceeds scaled width %d", numDisp, ws);
    return -1;
  }

  if (W != H->w || Ht != H->h || ws != H->ws || hs != H->hs) {
    free_bufs(H);
    size_t full = (size_t)W * Ht, small = (size_t)ws * hs;
    if (cudaMalloc(&H->d_fil, full) != cudaSuccess ||
        cudaMalloc(&H->d_fir, full) != cudaSuccess ||
        cudaMalloc(&H->d_il, small) != cudaSuccess ||
        cudaMalloc(&H->d_ir, small) != cudaSuccess ||
        cudaMalloc(&H->d_cenL, small * 4) != cudaSuccess ||
        cudaMalloc(&H->d_cenR, small * 4) != cudaSuccess ||
        cudaMalloc(&H->d_z, small * 4) != cudaSuccess ||
        cudaMalloc(&H->d_zmed, small * 4) != cudaSuccess ||
        cudaMalloc(&H->d_hw, full * 2) != cudaSuccess ||
        cudaMalloc(&H->d_out, full * 2) != cudaSuccess) {
      snprintf(H->err, sizeof(H->err), "device alloc failed");
      free_bufs(H);
      return -1;
    }
    H->w = W; H->h = Ht; H->ws = ws; H->hs = hs;
  }

  size_t full = (size_t)W * Ht;
  CUDA_OK(cudaMemcpy(H->d_fil, ir_left, full, cudaMemcpyHostToDevice), H,
          "upload L");
  CUDA_OK(cudaMemcpy(H->d_fir, ir_right, full, cudaMemcpyHostToDevice), H,
          "upload R");
  CUDA_OK(cudaMemcpy(H->d_hw, depth_mm, full * 2, cudaMemcpyHostToDevice), H,
          "upload depth");

  dim3 bs(16, 16);
  dim3 gs((ws + 15) / 16, (hs + 15) / 16);
  dim3 gf((W + 15) / 16, (Ht + 15) / 16);
  if (ws == W && hs == Ht) {
    CUDA_OK(cudaMemcpy(H->d_il, H->d_fil, full, cudaMemcpyDeviceToDevice), H,
            "copy");
    CUDA_OK(cudaMemcpy(H->d_ir, H->d_fir, full, cudaMemcpyDeviceToDevice), H,
            "copy");
  } else {
    k_downsample<<<gs, bs>>>(H->d_fil, W, Ht, H->d_il, ws, hs);
    k_downsample<<<gs, bs>>>(H->d_fir, W, Ht, H->d_ir, ws, hs);
  }
  k_census<<<gs, bs>>>(H->d_il, ws, hs, H->d_cenL);
  k_census<<<gs, bs>>>(H->d_ir, ws, hs, H->d_cenR);
  k_match<<<gs, bs>>>(H->d_cenL, H->d_cenR, ws, hs, numDisp, H->uniqueness,
                       f_scaled, H->baseline_m, H->d_z);
  k_median5<<<gs, bs>>>(H->d_z, ws, hs, H->d_zmed);
  int use_crop = (H->crop[0] >= 0 && H->crop[2] > H->crop[0] &&
                  H->crop[1] >= 0 && H->crop[3] > H->crop[1]);
  k_merge<<<gf, bs>>>(H->d_zmed, ws, hs, H->d_hw, H->d_out, W, Ht,
                       H->min_z_mm, (int)(H->zmin_m * 1000.0f),
                       H->crop[0], H->crop[1], H->crop[2], H->crop[3],
                       use_crop);
  CUDA_OK(cudaGetLastError(), H, "kernel launch");
  CUDA_OK(cudaDeviceSynchronize(), H, "sync");
  CUDA_OK(cudaMemcpy(depth_out, H->d_out, full * 2, cudaMemcpyDeviceToHost),
          H, "download");
  H->err[0] = '\0';
  return 0;
}

}  /* extern "C" */
