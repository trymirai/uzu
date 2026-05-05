use metal::prelude::*;

use crate::backends::common::SparseBufferOperation;

impl From<&SparseBufferOperation> for MTL4UpdateSparseBufferMappingOperation {
    fn from(op: &SparseBufferOperation) -> Self {
        // `op.range` is a byte range; the heap offset matches the buffer offset.
        Self::new(op.mode.into(), op.range.clone(), op.range.start)
    }
}
