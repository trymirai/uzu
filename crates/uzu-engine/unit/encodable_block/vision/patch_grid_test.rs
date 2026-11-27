use uzu_engine_macros::uzu_test;

use super::{VisionImage, VisionImageError, patch_grid};
use crate::encodable_block::cell_grid::CellGrid;

#[uzu_test]
fn vision_image_validation() {
    let grid_in_pixel_range = |width, height, bytes, patch_size, merge_size, min_pixels, max_pixels| {
        let image = VisionImage {
            width,
            height,
            pixels: vec![0; bytes].into_boxed_slice(),
        };
        patch_grid(&image, patch_size, merge_size, 2, 64, min_pixels, max_pixels)
    };
    let grid = |width, height, bytes, patch_size, merge_size| {
        grid_in_pixel_range(width, height, bytes, patch_size, merge_size, 0, u32::MAX)
    };

    assert_eq!(grid(8, 4, 96, 2, 2).unwrap(), CellGrid::new(4, 2, 2));
    for bytes in [95, 97] {
        assert!(matches!(grid(8, 4, bytes, 2, 2), Err(VisionImageError::PixelLength { .. })));
    }
    for (width, height) in [(0, 4), (4, 0), (6, 4), (4, 6), (3, 4)] {
        assert!(matches!(
            grid(width, height, (width * height * 3) as usize, 2, 2),
            Err(VisionImageError::InvalidDimensions { .. })
        ));
    }
    assert_eq!(grid_in_pixel_range(8, 4, 96, 2, 2, 32, 32).unwrap(), CellGrid::new(4, 2, 2));
    for (min_pixels, max_pixels) in [(33, u32::MAX), (0, 31)] {
        assert!(matches!(
            grid_in_pixel_range(8, 4, 96, 2, 2, min_pixels, max_pixels),
            Err(VisionImageError::PixelCount { .. })
        ));
    }
    for (patch, merge) in [(0, 2), (2, 0), (u32::MAX, 2)] {
        assert!(matches!(grid(4, 4, 48, patch, merge), Err(VisionImageError::InvalidBlockSize { .. })));
    }
    assert!(matches!(grid(u32::MAX, 1, 0, 1, 1), Err(VisionImageError::ImageTooLarge)));
}
