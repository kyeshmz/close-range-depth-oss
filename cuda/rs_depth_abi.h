/* rs_depth_abi.h — C ABI for the close-range library.
 *
 * Field-for-field compatible with the vendor `rs_depth_range.h` /
 * `rs_depth_metadata.h` so binaries compiled against either header link and
 * run against either `librs_depth_range.so`. No librealsense dependency.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StreamMetadata {
    int64_t  frame_number;
    double   timestamp_us;
    double   arrival_time_us;
    double   sensor_timestamp_us;
    double   backend_timestamp_us;
    int32_t  exposure_us;
    int32_t  gain;
    int32_t  auto_exposure;
    int32_t  laser_power;
    int32_t  emitter_mode;
    int32_t  white_balance;
    float    temperature;
    float    actual_fps;
    int32_t  brightness;
    int32_t  contrast;
    int32_t  saturation;
} StreamMetadata;

typedef struct FrameMetadata {
    StreamMetadata ir;
    StreamMetadata depth;
    StreamMetadata color;
    char     ir_format[8];
    int32_t  width;
    int32_t  height;
} FrameMetadata;

typedef void* RsDepthRange;

RsDepthRange rs_depth_range_create(
    float focal_length_px,
    float baseline_m,
    int   min_z_threshold,
    float scale_factor,
    int   crop_x1,
    int   crop_y1,
    int   crop_x2,
    int   crop_y2);

void rs_depth_range_destroy(RsDepthRange h);

int rs_depth_range_process(
    RsDepthRange           h,
    const uint8_t*         ir_left,
    const uint8_t*         ir_right,
    const uint16_t*        depth_mm,
    uint16_t*              depth_out,
    const FrameMetadata* metadata);

const char* rs_depth_range_last_error(RsDepthRange h);

#ifdef __cplusplus
}  /* extern "C" */
#endif
