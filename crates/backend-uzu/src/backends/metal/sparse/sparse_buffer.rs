use std::{fmt::Debug, ops::Range};

use bytesize::ByteSize;
use metal::prelude::*;
use rangemap::RangeMap;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer},
        metal::{Metal, error::MetalError, metal_extensions::SparsePageSizeExt},
    },
    prelude::MetalContext,
};

#[derive(Debug)]
pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mapped_pages: RangeMap<usize, ()>,
}

impl MetalSparseBuffer {
    pub(crate) fn new(
        context: &MetalContext,
        capacity: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        Ok(Self {
            buffer,
            mapped_pages: RangeMap::new(),
        })
    }
}

impl Buffer for MetalSparseBuffer {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn size(&self) -> ByteSize {
        self.buffer.size()
    }

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.buffer.set_label(label)
    }
}

impl SparseBuffer for MetalSparseBuffer {
    fn mapping(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let gaps: Vec<Range<usize>> = self.mapped_pages.gaps(pages).collect();
        for gap in gaps {
            context.sparse_heaps_mapper_mut().mapping(context, &self.buffer, &gap)?;
            self.mapped_pages.insert(gap, ());
        }
        Ok(())
    }

    fn unmapping(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let ranges: Vec<Range<usize>> =
            self.mapped_pages.overlapping(pages).map(|(r, _)| r.start.max(pages.start)..r.end.min(pages.end)).collect();
        for range in ranges {
            context.sparse_heaps_mapper_mut().unmapping(context, &self.buffer, &range)?;
            self.mapped_pages.remove(range);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/metal/sparse/sparse_buffer_test.rs"]
mod tests;
