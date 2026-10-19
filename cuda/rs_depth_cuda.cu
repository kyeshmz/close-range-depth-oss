/* rs_depth_cuda.cu — CUDA close-range depth improver, vendor-ABI compatible.
 *
 * Box-downscale the IR pair,
 * census-5x5 + winner-take-all stereo over the near disparity band,
 * disparity -> depth, 5x5 median, nearest-upscale merged with HW depth.
 *
 * Retains the vendor's four C entry points and metadata layout, plus a
 * project-specific configure() extension. Only zero HW depths are filled.
 * No librealsense dependency. Requires CUDA 12.8+ for sm_120 (Blackwell).
 */

#include "rs_depth_abi.h"

#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdio>
#include <new>

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
  bool use_median = true;

  int w = 0, h = 0, ws = 0, hs = 0;  /* cached geometry */
  uint8_t *d_il = nullptr, *d_ir = nullptr;    /* scaled IR */
  uint8_t *d_fil = nullptr, *d_fir = nullptr;  /* full-res staging */
  uint32_t *d_cenL = nullptr, *d_cenR = nullptr;
  float *d_z = nullptr, *d_zmed = nullptr;
  uint16_t *d_hw = nullptr, *d_out = nullptr;
  char err[256] = {0};
};

static thread_local char create_error[256] = {0};

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
    if (zc > 0.0f && zi >= floor_mm && zi < thresh_mm &&
        keep == 0)
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
  create_error[0] = '\0';
  double threshold = min_z_threshold > 0 ? min_z_threshold
      : (double)focal_length_px * baseline_m * 1000.0 / 105.0;
  if (!std::isfinite(focal_length_px) || focal_length_px <= 0 ||
      !std::isfinite(baseline_m) || baseline_m <= 0 ||
      !std::isfinite(threshold) || threshold < 1 || threshold > 65535 ||
      !std::isfinite(scale_factor) || scale_factor <= 0 || scale_factor > 1) {
    snprintf(create_error, sizeof(create_error), "invalid calibration, threshold, or scale");
    return nullptr;
  }
  bool no_crop = crop_x1 == -1 && crop_y1 == -1 && crop_x2 == -1 && crop_y2 == -1;
  if (!no_crop && !(crop_x1 >= 0 && crop_y1 >= 0 &&
                    crop_x2 > crop_x1 && crop_y2 > crop_y1)) {
    snprintf(create_error, sizeof(create_error), "invalid crop region");
    return nullptr;
  }
  auto *h = new (std::nothrow) RsDepthRangeImpl();
  if (!h) {
    snprintf(create_error, sizeof(create_error), "handle allocation failed");
    return nullptr;
  }
  h->focal_px = focal_length_px;
  h->baseline_m = baseline_m;
  h->min_z_mm = (int)threshold;
  h->zmin_m = baseline_m < 0.03f ? 0.02f : 0.12f;
  h->scale = scale_factor;
  h->crop[0] = crop_x1; h->crop[1] = crop_y1;
  h->crop[2] = crop_x2; h->crop[3] = crop_y2;
  return h;
}

int rs_depth_range_configure(RsDepthRange h, float min_depth_m, int use_median) {
  auto *H = (RsDepthRangeImpl *)h;
  if (!H) return -1;
  if (!std::isfinite(min_depth_m) || min_depth_m <= 0 ||
      min_depth_m * 1000.0f >= H->min_z_mm ||
      (use_median != 0 && use_median != 1)) {
    snprintf(H->err, sizeof(H->err), "invalid minimum depth or median setting");
    return -1;
  }
  H->zmin_m = min_depth_m;
  H->use_median = use_median != 0;
  H->err[0] = '\0';
  return 0;
}

void rs_depth_range_destroy(RsDepthRange h) {
  if (!h) return;
  free_bufs((RsDepthRangeImpl *)h);
  delete (RsDepthRangeImpl *)h;
}

const char *rs_depth_range_last_error(RsDepthRange h) {
  if (!h) return create_error[0] ? create_error : "invalid handle";
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
  if (W <= 0 || Ht <= 0 || (size_t)W * Ht > INT_MAX) {
    snprintf(H->err, sizeof(H->err), "bad metadata dims %dx%d", W, Ht);
    return -1;
  }
  bool use_crop = H->crop[0] >= 0;
  if (use_crop && (H->crop[2] > W || H->crop[3] > Ht)) {
    snprintf(H->err, sizeof(H->err), "crop region exceeds image dimensions");
    return -1;
  }
  if (H->zmin_m * 1000.0f >= H->min_z_mm) {
    snprintf(H->err, sizeof(H->err), "minimum depth must be below threshold");
    return -1;
  }
  int ws = H->scale >= 1.0f ? W : (int)(W * H->scale + 0.5f);
  int hs = H->scale >= 1.0f ? Ht : (int)(Ht * H->scale + 0.5f);
  if (ws < 1) ws = 1;
  if (hs < 1) hs = 1;
  float f_scaled = H->focal_px * ((float)ws / (float)W);
  float search = ceilf(f_scaled * H->baseline_m / H->zmin_m / 16.0f) * 16;
  if (search < 16) search = 16;
  if (!std::isfinite(search) || search >= ws) {
    snprintf(H->err, sizeof(H->err),
             "disparity search %.0f exceeds scaled width %d", search, ws);
    return -1;
  }
  int numDisp = (int)search;

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
  if (H->use_median)
    k_median5<<<gs, bs>>>(H->d_z, ws, hs, H->d_zmed);
  k_merge<<<gf, bs>>>(H->use_median ? H->d_zmed : H->d_z,
                       ws, hs, H->d_hw, H->d_out, W, Ht,
                       H->min_z_mm, (int)ceilf(H->zmin_m * 1000.0f),
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
