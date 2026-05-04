use std::ops::Range;

use bytesize::ByteSize;
use rangemap::RangeMap;

use super::{SparseBufferOperation, SparseResourceMappingMode};
use crate::backends::common::Backend;

pub trait SparseBuffer {
    type Backend: Backend<SparseBuffer = Self>;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );

    fn gpu_ptr(&self) -> usize;

    fn length(&self) -> ByteSize;

    fn get_mapped_pages(&self) -> &RangeMap<usize, ()>;

    fn get_page_size(&self) -> ByteSize;

    fn execute(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        operations: &[SparseBufferOperation],
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

#[allow(dead_code)]
pub trait SparseBufferExt: SparseBuffer {
    fn map_bytes(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        bytes_ranges: &[Range<usize>],
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let page_size_bytes = self.get_page_size().as_u64() as usize;
        let pages_ranges = bytes_ranges
            .iter()
            .map(|bytes_range| Range {
                start: bytes_range.start / page_size_bytes,
                end: bytes_range.end / page_size_bytes,
            })
            .collect::<Vec<Range<usize>>>();
        self.map_pages(context, &pages_ranges)
    }

    fn map_pages(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages_ranges: &[Range<usize>],
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let page_size_bytes = self.get_page_size().as_u64() as usize;
        let curr_mapped_pages = self.get_mapped_pages();

        let mut operations = Vec::<SparseBufferOperation>::new();
        for page_range in pages_ranges {
            let byte_range = (page_range.start * page_size_bytes)..(page_range.end * page_size_bytes);
            let gaps = curr_mapped_pages.gaps(&byte_range);
            for gap in gaps {
                operations.push(SparseBufferOperation::new(SparseResourceMappingMode::Map, gap));
            }
        }

        self.execute(context, &operations)
    }
}

#[derive(Debug)]
pub(crate) struct SparseBufferMappedPages {
    map: RangeMap<usize, ()>,
}

impl SparseBufferMappedPages {
    pub fn new() -> Self {
        Self {
            map: RangeMap::new(),
        }
    }

    pub fn execute(
        &mut self,
        operations: &[SparseBufferOperation],
    ) {
        operations.iter().for_each(|op| match op.mode {
            SparseResourceMappingMode::Map => self.map.insert(op.range.clone(), ()),
            SparseResourceMappingMode::Unmap => self.map.remove(op.range.clone()),
        });
    }

    pub fn get_map(&self) -> &RangeMap<usize, ()> {
        &self.map
    }
}
