use std::{
    cmp::{max, min},
    fmt::Debug,
    ops::Range,
};

use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions, MTLSparsePageSize};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer},
        metal::{Metal, error::MetalError, metal_extensions::SparsePageSizeExt},
    },
    prelude::MetalContext,
};

pub struct MetalSparseBuffer<'a> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mapped_pages: RangeSet<usize>,
    context: &'a MetalContext,
}

impl<'a> MetalSparseBuffer<'a> {
    pub(crate) fn new(
        context: &'a MetalContext,
        capacity: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let page_size_bytes = page_size.in_bytes();
        let aligned_capacity = capacity.next_multiple_of(page_size_bytes);

        let buffer = context
            .device
            .new_buffer_with_length_options_placement_sparse_page_size(
                aligned_capacity,
                MTLResourceOptions::STORAGE_MODE_PRIVATE,
                page_size,
            )
            .ok_or(MetalError::SparseBufferAlloc(aligned_capacity))?;

        Ok(Self {
            buffer,
            mapped_pages: RangeSet::new(),
            context,
        })
    }
}

impl Drop for MetalSparseBuffer<'_> {
    fn drop(&mut self) {
        for range in self.mapped_pages.iter() {
            self.context.sparse_heap_pool_mut().unmap(self.context, &self.buffer, &range).expect("Failed to unmap");
        }
    }
}

impl Debug for MetalSparseBuffer<'_> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("MetalSparseBuffer")
            .field("mapped_pages", &self.mapped_pages)
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl Buffer for MetalSparseBuffer<'_> {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn size(&self) -> usize {
        self.buffer.size()
    }

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.buffer.set_label(label)
    }
}

impl<'a> SparseBuffer for MetalSparseBuffer<'a> {
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        self.mapped_pages.gaps(pages).collect::<Vec<_>>().into_iter().try_for_each(|gap| {
            context.sparse_heap_pool_mut().map(context, &self.buffer, &gap)?;
            self.mapped_pages.insert(gap);
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
            .map(|range| max(range.start, pages.start)..min(range.end, pages.end))
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
