use rangemap::RangeMap;

use crate::backends::common::{SparseBufferOperation, SparseResourceMappingMode};

#[derive(Debug)]
pub(super) struct SparseBufferMappedPages {
    map: RangeMap<usize, ()>,
}

impl SparseBufferMappedPages {
    pub(super) fn new() -> Self {
        Self {
            map: RangeMap::new(),
        }
    }

    pub(super) fn execute(
        &mut self,
        operations: &[SparseBufferOperation],
    ) {
        operations.iter().for_each(|op| match op.mode {
            SparseResourceMappingMode::Map => self.map.insert(op.range.clone(), ()),
            SparseResourceMappingMode::Unmap => self.map.remove(op.range.clone()),
        });
    }

    pub(super) fn get_map(&self) -> &RangeMap<usize, ()> {
        &self.map
    }
}
