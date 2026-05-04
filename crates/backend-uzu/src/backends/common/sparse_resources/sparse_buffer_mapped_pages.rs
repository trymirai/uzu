use rangemap::RangeMap;

use super::{SparseBufferOperation, SparseResourceMappingMode};

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct SparseBufferMappedPages {
    map: RangeMap<usize, ()>,
}

#[allow(dead_code)]
impl SparseBufferMappedPages {
    pub(crate) fn new() -> Self {
        Self {
            map: RangeMap::new(),
        }
    }

    pub(crate) fn execute(
        &mut self,
        operations: &[SparseBufferOperation],
    ) {
        operations.iter().for_each(|op| match op.mode {
            SparseResourceMappingMode::Map => self.map.insert(op.range.clone(), ()),
            SparseResourceMappingMode::Unmap => self.map.remove(op.range.clone()),
        });
    }

    pub(crate) fn get_map(&self) -> &RangeMap<usize, ()> {
        &self.map
    }
}
