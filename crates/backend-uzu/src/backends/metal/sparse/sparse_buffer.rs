use std::{
    cmp::{max, min},
    fmt::Debug,
    ops::Range,
    rc::Rc,
};

use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions, MTLSparsePageSize};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer, SparseBufferExt},
        metal::{Metal, error::MetalError, metal_extensions::SparsePageSizeExt, sparse::MetalSparseMappingOpsBatch},
    },
    prelude::MetalContext,
};

pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mapped_pages: RangeSet<usize>,
    context: Rc<MetalContext>,
    page_size: MTLSparsePageSize,
}

impl MetalSparseBuffer {
    pub(crate) fn new(
        context: Rc<MetalContext>,
        capacity: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let aligned_capacity = capacity.next_multiple_of(page_size.in_bytes());
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
            page_size,
        })
    }

    pub(crate) fn mtl_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }
}

impl Debug for MetalSparseBuffer {
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

impl Drop for MetalSparseBuffer {
    fn drop(&mut self) {
        let context = self.context.clone();
        self.unmap(&context, &(0..self.total_pages())).expect("Failed to unmap");
    }
}

impl Buffer for MetalSparseBuffer {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn size(&self) -> usize {
        self.buffer.size()
    }
}

impl SparseBuffer for MetalSparseBuffer {
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        if pages.len() == 0 {
            return Ok(());
        }

        // prepare operations
        let mut all_batches: Vec<MetalSparseMappingOpsBatch> = Vec::new();
        let gaps = self.mapped_pages.gaps(pages).collect::<Vec<_>>();
        let pages_to_map = gaps.iter().map(|gap| gap.len()).sum();
        context.sparse_heap_pool_mut().ensure_enough_free_pages(context, pages_to_map)?;

        for gap in gaps.iter() {
            let mut pool = context.sparse_heap_pool_mut();
            let result = pool.create_map_operations(context, &self.buffer, gap);
            match result {
                Ok(batches) => {
                    pool.apply_map_operations(&batches);
                    all_batches.extend(batches);
                },
                Err(err) => return Err(err),
            };
        }

        // execute operations
        context.sparse_update_mappings(&all_batches);
        for gap in gaps {
            self.mapped_pages.insert(gap)
        }

        Ok(())
    }

    fn unmap(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        if pages.len() == 0 {
            return Ok(());
        }

        // prepare operations
        let mut all_batches: Vec<MetalSparseMappingOpsBatch> = Vec::new();
        let mapped_ranges = self
            .mapped_pages
            .overlapping(pages)
            .map(|range| max(range.start, pages.start)..min(range.end, pages.end))
            .collect::<Vec<_>>();
        for mapped_range in mapped_ranges.iter() {
            let batches = context.sparse_heap_pool().create_unmap_operations(&self.buffer, mapped_range);
            context.sparse_heap_pool_mut().apply_map_operations(&batches);
            all_batches.extend(batches);
        }

        // execute operations
        context.sparse_update_mappings(&all_batches);
        for mapped_range in mapped_ranges {
            self.mapped_pages.remove(mapped_range)
        }

        Ok(())
    }

    fn page_size_bytes(&self) -> usize {
        self.page_size.in_bytes()
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/metal/sparse/sparse_buffer_test.rs"]
mod tests;
