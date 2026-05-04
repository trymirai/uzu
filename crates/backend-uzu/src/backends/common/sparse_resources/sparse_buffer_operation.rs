use std::ops::Range;

use super::SparseResourceMappingMode;

/// A single sparse-buffer page-mapping operation. `range` is expressed in
/// **bytes**, page-aligned to the owning buffer's `MTLSparsePageSize`.
pub struct SparseBufferOperation {
    pub mode: SparseResourceMappingMode,
    pub range: Range<usize>,
}

impl SparseBufferOperation {
    pub fn new(
        mode: SparseResourceMappingMode,
        range: Range<usize>,
    ) -> Self {
        Self {
            mode,
            range,
        }
    }
}
