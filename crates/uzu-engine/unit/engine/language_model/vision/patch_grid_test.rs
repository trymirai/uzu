use uzu_engine_macros::uzu_test;

use super::{VisionImage, VisionImageError, merged_image_grid};

#[uzu_test]
fn vision_image_validation() {
    let grid = |width, height, bytes, patch_size, merge_size| {
        let image = VisionImage {
            width,
            height,
            pixels: vec![0; bytes].into_boxed_slice(),
        };
        merged_image_grid(&image, patch_size, merge_size, 2, 64)
    };

    assert_eq!(grid(8, 4, 96, 2, 2).unwrap(), (2, 1));
    for bytes in [95, 97] {
        assert!(matches!(grid(8, 4, bytes, 2, 2), Err(VisionImageError::PixelLength { .. })));
    }
    for (width, height) in [(0, 4), (4, 0), (6, 4), (4, 6), (3, 4)] {
        assert!(matches!(
            grid(width, height, (width * height * 3) as usize, 2, 2),
            Err(VisionImageError::InvalidDimensions { .. })
        ));
    }
    for (patch, merge) in [(0, 2), (2, 0), (u32::MAX, 2)] {
        assert!(matches!(grid(4, 4, 48, patch, merge), Err(VisionImageError::InvalidBlockSize { .. })));
    }
    assert!(matches!(grid(u32::MAX, 1, 0, 1, 1), Err(VisionImageError::ImageTooLarge)));
}
