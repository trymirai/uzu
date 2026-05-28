use std::{
    cell::RefCell,
    cmp::{max, min},
    ops::Range,
    rc::Rc,
};

use metal::{MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLSparsePageSize, MTLSparseTextureMappingMode};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    backends::metal::{
        error::MetalError,
        metal_extensions::SparsePageSizeExt,
        sparse::{MetalSparseMappingOpsBatch, sparse_heap::MetalSparseHeap},
    },
    prelude::MetalContext,
};

pub struct MetalSparseHeapPool {
    heaps: Vec<Rc<RefCell<MetalSparseHeap>>>,
    heap_capacity: usize,
    page_size: MTLSparsePageSize,
}

impl MetalSparseHeapPool {
    pub fn new(
        page_size: MTLSparsePageSize,
        heap_capacity: usize,
    ) -> Self {
        Self {
            heaps: Vec::new(),
            page_size,
            heap_capacity,
        }
    }

    pub fn ensure_enough_free_pages(
        &mut self,
        context: &MetalContext,
        pages: usize,
    ) -> Result<(), MetalError> {
        let mut pages_to_alloc = pages;
        for heap in self.heaps.iter() {
            for free_pages_range in heap.borrow().free_pages().iter() {
                pages_to_alloc -= min(free_pages_range.len(), pages_to_alloc);
            }
        }
        if pages_to_alloc == 0 {
            return Ok(());
        }

        let new_heaps_count = (pages_to_alloc * self.page_size.in_bytes()).div_ceil(self.heap_capacity);
        for _ in 0..new_heaps_count {
            let heap = MetalSparseHeap::new(context, self.heap_capacity, self.page_size)?;
            context.update_peak_memory_usage();
            self.heaps.push(Rc::new(RefCell::new(heap)));
        }

        Ok(())
    }

    pub fn create_map_operations(
        &mut self,
        context: &MetalContext,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        buffer_pages: &Range<usize>,
    ) -> Result<Vec<MetalSparseMappingOpsBatch>, MetalError> {
        self.ensure_enough_free_pages(context, buffer_pages.len())?;

        let mut pages_to_map = buffer_pages.clone();
        let mut batches: Vec<MetalSparseMappingOpsBatch> = Vec::new();
        for heap in self.heaps.iter() {
            let mut heap_mtl_operations: Vec<MTL4UpdateSparseBufferMappingOperation> = Vec::new();
            for heap_free_pages_range in heap.borrow().free_pages().iter() {
                let map_pages_count = min(heap_free_pages_range.len(), pages_to_map.len());
                if map_pages_count == 0 {
                    break;
                }

                let mtl_operation = MTL4UpdateSparseBufferMappingOperation::new(
                    MTLSparseTextureMappingMode::Map,
                    pages_to_map.start..(pages_to_map.start + map_pages_count),
                    heap_free_pages_range.start,
                );
                heap_mtl_operations.push(mtl_operation);

                pages_to_map.start += map_pages_count;
            }

            if !heap_mtl_operations.is_empty() {
                let batch = MetalSparseMappingOpsBatch {
                    buffer: buffer.clone(),
                    heap: heap.clone(),
                    mtl_operations: heap_mtl_operations.into_boxed_slice(),
                };
                batches.push(batch);
            }
        }

        Ok(batches)
    }

    pub fn create_unmap_operations(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        buffer_pages: &Range<usize>,
    ) -> Vec<MetalSparseMappingOpsBatch> {
        let mut batches: Vec<MetalSparseMappingOpsBatch> = Vec::new();

        let buffer_address = buffer.gpu_address();
        for heap in self.heaps.iter() {
            let mut heap_mtl_operations: Vec<MTL4UpdateSparseBufferMappingOperation> = Vec::new();

            for (heap_range, mapping) in heap.borrow().mappings_for(buffer_address) {
                let mapped_buffer_pages = mapping.buffer_pages_for(&heap_range);
                let unmap_start_page = max(buffer_pages.start, mapped_buffer_pages.start);
                let unmap_end_page = min(buffer_pages.end, mapped_buffer_pages.end);
                if unmap_start_page < unmap_end_page {
                    let offset_within_range = unmap_start_page - mapped_buffer_pages.start;
                    let mtl_operation = MTL4UpdateSparseBufferMappingOperation::new(
                        MTLSparseTextureMappingMode::Unmap,
                        unmap_start_page..unmap_end_page,
                        heap_range.start + offset_within_range,
                    );
                    heap_mtl_operations.push(mtl_operation);
                };
            }

            if !heap_mtl_operations.is_empty() {
                let batch = MetalSparseMappingOpsBatch {
                    buffer: buffer.clone(),
                    heap: heap.clone(),
                    mtl_operations: heap_mtl_operations.into_boxed_slice(),
                };
                batches.push(batch);
            }
        }

        batches
    }

    pub fn apply_map_operations(
        &mut self,
        batches: &[MetalSparseMappingOpsBatch],
    ) {
        for batch in batches.iter() {
            for sparse_heap in self.heaps.iter() {
                if sparse_heap.borrow().heap() == batch.heap.borrow().heap() {
                    sparse_heap.borrow_mut().apply_mapping_operations(batch);
                }
            }
        }

        self.heaps.retain(|heap| !heap.borrow().is_empty());
    }

    #[cfg(test)]
    pub fn heap_capacity_bytes(&self) -> usize {
        self.heap_capacity
    }

    #[cfg(test)]
    pub fn heap_capacity_pages(&self) -> usize {
        self.heap_capacity / self.page_size.in_bytes()
    }

    #[cfg(test)]
    pub fn heaps_count(&self) -> usize {
        self.heaps.len()
    }

    pub fn page_size(&self) -> MTLSparsePageSize {
        self.page_size
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/metal/sparse/sparse_heap_pool_test.rs"]
mod tests;
