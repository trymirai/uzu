use shoji::types::session::vision::VisionImage;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisionImageError {
    #[error("Invalid vision patch size {patch_size} or spatial merge size {merge_size}")]
    InvalidBlockSize {
        patch_size: u32,
        merge_size: u32,
    },
    #[error("Image dimensions {width}x{height} must both be positive multiples of {block_size}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        block_size: u32,
    },
    #[error("RGB buffer contains {actual} bytes, expected {expected}")]
    PixelLength {
        actual: usize,
        expected: u32,
    },
    #[error("Image exceeds 32-bit vision kernel indexing limits")]
    ImageTooLarge,
}

/// Checks the RGB buffer and the two vision kernel dispatch sizes before returning the merged grid.
pub(super) fn merged_image_grid(
    image: &VisionImage,
    patch_size: u32,
    merge_size: u32,
    temporal_patch_size: u32,
    model_dim: u32,
) -> Result<(u32, u32), VisionImageError> {
    let block_size =
        patch_size.checked_mul(merge_size).filter(|&size| size != 0).ok_or(VisionImageError::InvalidBlockSize {
            patch_size,
            merge_size,
        })?;
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 || !width.is_multiple_of(block_size) || !height.is_multiple_of(block_size) {
        return Err(VisionImageError::InvalidDimensions {
            width,
            height,
            block_size,
        });
    }

    let expected =
        width.checked_mul(height).and_then(|pixels| pixels.checked_mul(3)).ok_or(VisionImageError::ImageTooLarge)?;
    if image.pixels.len() != expected as usize {
        return Err(VisionImageError::PixelLength {
            actual: image.pixels.len(),
            expected,
        });
    }

    expected.checked_mul(temporal_patch_size).ok_or(VisionImageError::ImageTooLarge)?;

    let patches = (width / patch_size) * (height / patch_size);
    patches.checked_mul(model_dim).ok_or(VisionImageError::ImageTooLarge)?;

    Ok((width / block_size, height / block_size))
}

/// Grid coordinates `(x, y)` of the `row`-th patch of the vision encoder input.
/// Same as `vision_patch_coordinates` in `metal/kernel/vision/patch_grid.h`, see the example there.
pub fn vision_patch_coordinates(
    row: u32,
    grid_width: u32,
    merge_size: u32,
) -> (u32, u32) {
    let block_size = merge_size * merge_size;
    let blocks_per_row = grid_width / merge_size;
    let block = row / block_size;
    let in_block = row % block_size;
    let y = (block / blocks_per_row) * merge_size + in_block / merge_size;
    let x = (block % blocks_per_row) * merge_size + in_block % merge_size;
    (x, y)
}

#[cfg(test)]
#[path = "../../../../unit/engine/language_model/vision/patch_grid_test.rs"]
mod tests;
