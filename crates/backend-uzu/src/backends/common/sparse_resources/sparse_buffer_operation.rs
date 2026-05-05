use std::ops::Range;

use super::SparseResourceMappingMode;

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
