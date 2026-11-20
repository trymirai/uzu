use half::bf16;
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::vision::{CHANNELS, MAX_CHANNEL_VALUE},
    data_type::DataTypeRepr,
    engine::language_model::vision::patch_grid::vision_patch_coordinates,
};

/// Converts an RGB image into patch embedding input, see `metal/kernel/vision/patchify.metal`.
#[kernel(VisionPatchify)]
#[variants(T, bf16)]
pub fn vision_patchify<T: DataTypeRepr>(
    pixels: *const u8,
    patches: *mut T,
    image_mean: *const f32,
    image_std: *const f32,
    width: u32,
    patch_size: u32,
    merge_size: u32,
    temporal_patch_size: u32,
    length: u32,
) {
    let patch_area = patch_size * patch_size;
    let patch_dim = CHANNELS * temporal_patch_size * patch_area;
    let grid_width = width / patch_size;

    for output_index in 0..length {
        let patch_row = output_index / patch_dim;
        let patch_col = output_index % patch_dim;

        let channel = patch_col / (temporal_patch_size * patch_area);
        let y_in_patch = (patch_col % patch_area) / patch_size;
        let x_in_patch = patch_col % patch_size;

        let (patch_x, patch_y) = vision_patch_coordinates(patch_row, grid_width, merge_size);
        let y = patch_y * patch_size + y_in_patch;
        let x = patch_x * patch_size + x_in_patch;

        let pixel_idx = ((y * width + x) * CHANNELS + channel) as usize;
        let channel = channel as usize;
        unsafe {
            let pixel = f32::from(*pixels.add(pixel_idx)) / MAX_CHANNEL_VALUE;
            let normalized_pixel = (pixel - *image_mean.add(channel)) / *image_std.add(channel);
            *patches.add(output_index as usize) = T::from(normalized_pixel).unwrap();
        }
    }
}
