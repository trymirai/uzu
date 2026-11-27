use shoji::types::session::vision::VisionImage;
use thiserror::Error;

use crate::{backends::common::gpu_types::vision::CHANNELS, encodable_block::cell_grid::CellGrid};

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
    #[error("Image has {pixels} pixels, expected from {min_pixels} to {max_pixels}")]
    PixelCount {
        pixels: u32,
        min_pixels: u32,
        max_pixels: u32,
    },
    #[error("RGB buffer contains {actual} bytes, expected {expected}")]
    PixelLength {
        actual: usize,
        expected: u32,
    },
    #[error("Image exceeds 32-bit vision kernel indexing limits")]
    ImageTooLarge,
}

/// Checks the pixel count, the RGB buffer and the two vision kernel dispatch sizes before returning the patch grid.
pub(super) fn patch_grid(
    image: &VisionImage,
    patch_size: u32,
    merge_size: u32,
    temporal_patch_size: u32,
    model_dim: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<CellGrid, VisionImageError> {
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

    let pixels = width.checked_mul(height).ok_or(VisionImageError::ImageTooLarge)?;
    if !(min_pixels..=max_pixels).contains(&pixels) {
        return Err(VisionImageError::PixelCount {
            pixels,
            min_pixels,
            max_pixels,
        });
    }

    let expected = pixels.checked_mul(CHANNELS).ok_or(VisionImageError::ImageTooLarge)?;
    if image.pixels.len() != expected as usize {
        return Err(VisionImageError::PixelLength {
            actual: image.pixels.len(),
            expected,
        });
    }

    expected.checked_mul(temporal_patch_size).ok_or(VisionImageError::ImageTooLarge)?;

    let patches = (width / patch_size) * (height / patch_size);
    patches.checked_mul(model_dim).ok_or(VisionImageError::ImageTooLarge)?;

    Ok(CellGrid::new(width / patch_size, height / patch_size, merge_size))
}

#[cfg(test)]
#[path = "../../../../unit/engine/language_model/vision/patch_grid_test.rs"]
mod tests;
