mod layer;
mod model;
mod position_embedding;
mod prefill;
mod rope;

use itertools::iproduct;
pub use model::{VisionModel, VisionModelEncodeError, VisionModelLoadError};
pub use prefill::{VisionModelInputImage, VisionModelPrefillError, VisionModelPrefilledImages};

/// Row and column of each patch of a `height x width` grid, in HF's order: each `merge x merge` block of patches is contiguous.
/// The preprocessor emits patches in this order, so it's also the row order of the vision stream.
fn patch_coordinates(
    height: u32,
    width: u32,
    merge: u32,
) -> impl Iterator<Item = (u32, u32)> {
    assert!(
        height.is_multiple_of(merge) && width.is_multiple_of(merge),
        "{height}x{width} patches aren't divisible by the merge size {merge}"
    );
    iproduct!(0..height / merge, 0..width / merge, 0..merge, 0..merge)
        .map(move |(block_row, block_col, row, col)| (block_row * merge + row, block_col * merge + col))
}
