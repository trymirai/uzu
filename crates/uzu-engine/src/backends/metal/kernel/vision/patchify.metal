#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/vision.h"
#include "patch_grid.h"

using namespace uzu::vision;

// Convert RGB image into patch embedding input.
// A patch is [channel, frame, y, x], and every frame repeats the image.
//
// Example: image 4x4, 1 channel, patch 2x2:
//
// image (pixels)                  in-memory pixels (row by row):
//   a  b │ c  d                   a b c d e f g h i j k l m n o p
//   e  f │ g  h
//  ──────┼──────
//   i  j │ k  l
//   m  n │ o  p
//
// 1. image splits in 4 patches 2x2:
//  row 0 (left top):        a b e f
//  row 1 (right top):       c d g h
//  row 2 (left bottom):     i j m n
//  row 3 (right bottom):    k l o p
//
// 2. 3 channels:
//    in pixels:            aR aG aB bR bG bB …
//    in patch row 0:       aR bR eR fR │ aG bG eG fG │ aB bB eB fB
//
// 3. 2 frames:
//    row 0:  aR bR eR fR aR bR eR fR │ aG … aG … │ aB … aB …
//            └─ frame0 ─┘└─ frame1 ─┘
//
// @param pixels                — input image [height, width, CHANNELS]
// @param patches               — result patches [rows, CHANNELS * temporal_patch_size * patch_size * patch_size]
// @param image_mean            — mean for image normalization [CHANNELS], each value for each channel
// @param image_std             — std dev for image normalization [CHANNELS], each value for each channel
// @param width                 — image width
// @param patch_size            — patch side size
// @param merge_size            — block side size for merger (how many patches merge in 1 token)
// @param temporal_patch_size   — frames count per patch
// @param length                — number of elements in patches (rows * patch_dim), one thread per element
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(VisionPatchify)(
    const device uchar* pixels,
    device T* patches,
    const device float* image_mean,
    const device float* image_std,
    constant uint& width,
    constant uint& patch_size,
    constant uint& merge_size,
    constant uint& temporal_patch_size,
    constant uint& length,
    const uint output_index AXIS(length, 256)
) {
  const uint patch_area = patch_size * patch_size;
  const uint patch_dim = CHANNELS * temporal_patch_size * patch_area;
  const uint patch_row = output_index / patch_dim;
  const uint patch_col = output_index % patch_dim;

  const uint channel = patch_col / (temporal_patch_size * patch_area);
  const uint y_in_patch = (patch_col % patch_area) / patch_size;
  const uint x_in_patch = patch_col % patch_size;

  const uint grid_width = width / patch_size;
  const uint2 patch_position = vision_patch_coordinates(patch_row, grid_width, merge_size);
  const uint y = patch_position.y * patch_size + y_in_patch;
  const uint x = patch_position.x * patch_size + x_in_patch;

  const uint pixel_idx = (y * width + x) * CHANNELS + channel;
  const float pixel = static_cast<float>(pixels[pixel_idx]) / MAX_CHANNEL_VALUE;
  const float normalized_pixel = (pixel - image_mean[channel]) / image_std[channel];
  patches[output_index] = static_cast<T>(normalized_pixel);
}
