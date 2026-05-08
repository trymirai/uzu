use std::{
    cmp::{max, min},
    fmt::Debug,
    ops::Range,
    rc::Weak,
};

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
    context: Weak<MetalContext>,
}

impl MetalSparseBuffer {
    pub(crate) fn new(
        context: Weak<MetalContext>,
        capacity: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let Some(ctx) = context.upgrade() else {
            return Err(MetalError::CannotCreateBuffer);
        };

        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = ctx.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        Ok(Self {
            buffer,
            mapped_pages: RangeMap::new(),
            context,
        })
    }
}

impl Drop for MetalSparseBuffer {
    fn drop(&mut self) {
        let Some(context) = self.context.upgrade() else {
            return;
        };

        let ranges: Vec<Range<usize>> = self.mapped_pages.iter().map(|(range, _)| range.clone()).collect();
        ranges.iter().for_each(|range| {
            let _ = context.sparse_heap_pool_mut().unmap(&context, &self.buffer, range);
        });
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
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let mut gaps_iter = self.mapped_pages.gaps(pages).collect::<Vec<_>>().into_iter();
        gaps_iter.try_for_each(|gap| {
            context.sparse_heap_pool_mut().map(context, &self.buffer, &gap)?;
            self.mapped_pages.insert(gap, ());
            Ok(())
        })
    }

    fn unmap(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        self.mapped_pages
            .overlapping(pages)
            .map(|(range, _)| max(range.start, pages.start)..min(range.end, pages.end))
            .collect::<Vec<_>>()
            .into_iter()
            .try_for_each(|range| {
                context.sparse_heap_pool_mut().unmap(context, &self.buffer, &range)?;
                self.mapped_pages.remove(range);
                Ok(())
            })
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/metal/sparse/sparse_buffer_test.rs"]
mod tests;
