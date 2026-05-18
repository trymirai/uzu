use std::{
    cmp::{max, min},
    ops::Range,
};

use metal::{MTLBuffer, MTLSparsePageSize};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    backends::metal::{
        error::MetalError,
        metal_extensions::SparsePageSizeExt,
        sparse::{
            MetalSparseMappingOperations, sparse_heap::MetalSparseHeap, sparse_utils::MetalSparseHeapMappingParameters,
        },
    },
    prelude::MetalContext,
};

pub struct MetalSparseHeapPool {
    heaps: Vec<MetalSparseHeap>,
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

    pub fn ensure

    // pub fn map(
    //     &mut self,
    //     context: &MetalContext,
    //     buffer: &ProtocolObject<dyn MTLBuffer>,
    //     buffer_pages: &Range<usize>,
    // ) -> Result<Box<[MetalSparseMappingOperations]>, MetalError> {
    //     if buffer_pages.len() == 0 {
    //         return Ok(Box::new([]));
    //     }
    //
    //     let mut pages_to_map = buffer_pages.clone();
    //     let heap_capacity_pages = self.heap_capacity_pages();
    //
    //     // Try to find pages in existing heaps.
    //     // While `pages_to_map` is not empty in each heap find unmapped pages, collect them into `mappings` and map
    //     let mut existing_heaps_mappings: Vec<(usize, Box<[MetalSparseHeapMappingParameters]>)> = Vec::new();
    //     let mut sparse_mapping_operations: Vec<MetalSparseMappingOperations> = Vec::new();
    //     for (i, heap) in self.heaps.iter_mut().enumerate() {
    //         let mut mappings: Vec<MetalSparseHeapMappingParameters> = Vec::new();
    //
    //         for free_pages in heap.free_pages().iter() {
    //             let map_pages_count = min(free_pages.len(), pages_to_map.len());
    //             let mapping = MetalSparseHeapMappingParameters {
    //                 buffer_pages: pages_to_map.start..(pages_to_map.start + map_pages_count),
    //                 heap_page_offset: free_pages.start,
    //             };
    //             mappings.push(mapping);
    //
    //             pages_to_map.start += map_pages_count;
    //             if pages_to_map.len() == 0 {
    //                 break;
    //             }
    //         }
    //
    //         if let Some(operations) = heap.create_mapping_operations(buffer, &mappings, true) {
    //             sparse_mapping_operations.push(operations);
    //         }
    //         existing_heaps_mappings.push((i, mappings.into_boxed_slice()));
    //
    //         if pages_to_map.len() == 0 {
    //             break;
    //         }
    //     }
    //
    //     // If `pages_to_map` still not empty, it's necessary to allocate new heaps
    //     let new_heaps_required = pages_to_map.len().div_ceil(heap_capacity_pages);
    //     for _ in 0..new_heaps_required {
    //         let map_pages_count = min(heap_capacity_pages, pages_to_map.len());
    //         let buffer_pages = pages_to_map.start..(pages_to_map.start + map_pages_count);
    //         let op_params = MetalSparseHeapMappingParameters {
    //             buffer_pages,
    //             heap_page_offset: 0,
    //         };
    //
    //         match MetalSparseHeap::new(context, self.heap_capacity, self.page_size) {
    //             Ok(mut heap) => {
    //                 let operations = [op_params];
    //                 if let Some(mapping_operations) = heap.create_mapping_operations(buffer, &operations, true) {
    //                     sparse_mapping_operations.push(mapping_operations);
    //                 }
    //                 self.heaps.push(heap);
    //                 pages_to_map.start += map_pages_count;
    //                 existing_heaps_mappings.push((self.heaps.len() - 1, Box::new(operations)));
    //             },
    //             Err(err) => {
    //                 // it's necessary to unmap previously mapped pages in existing heaps
    //                 for (heap_pos, mappings) in existing_heaps_mappings {
    //                     let _ = self.heaps[heap_pos].create_mapping_operations(buffer, &mappings, false);
    //                 }
    //                 return Err(err);
    //             },
    //         };
    //     }
    //
    //     Ok(sparse_mapping_operations.into_boxed_slice())
    // }
    //
    // pub fn unmap(
    //     &mut self,
    //     _context: &MetalContext,
    //     buffer: &ProtocolObject<dyn MTLBuffer>,
    //     buffer_pages: &Range<usize>,
    // ) -> Result<Box<[MetalSparseMappingOperations]>, MetalError> {
    //     let buffer_address = buffer.gpu_address();
    //     let mut sparse_mapping_operations: Vec<MetalSparseMappingOperations> = Vec::new();
    //
    //     // Iterate over heaps, find mappings for `buffer_address`, unmap them
    //     self.heaps.iter_mut().for_each(|heap| {
    //         let unmappings: Vec<MetalSparseHeapMappingParameters> = heap
    //             .mappings_for(buffer_address)
    //             .filter_map(|(heap_range, mapping)| {
    //                 let mapped_buffer_pages = mapping.buffer_pages_for(&heap_range);
    //                 let unmap_start = max(buffer_pages.start, mapped_buffer_pages.start);
    //                 let unmap_end = min(buffer_pages.end, mapped_buffer_pages.end);
    //                 if unmap_start >= unmap_end {
    //                     return None;
    //                 };
    //
    //                 let offset_within_range = unmap_start - mapped_buffer_pages.start;
    //                 Some(MetalSparseHeapMappingParameters {
    //                     buffer_pages: unmap_start..unmap_end,
    //                     heap_page_offset: heap_range.start + offset_within_range,
    //                 })
    //             })
    //             .collect();
    //         if let Some(operations) = heap.create_mapping_operations(buffer, &unmappings, false) {
    //             sparse_mapping_operations.push(operations);
    //         }
    //     });
    //
    //     // Remove empty heaps
    //     self.heaps.retain(|heap| !heap.is_empty());
    //
    //     Ok(sparse_mapping_operations.into_boxed_slice())
    // }

    pub fn create_map_operations(
        &self,
        context: &MetalContext,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        buffer_pages: &Range<usize>,
    ) -> Box<[MetalSparseMappingOperations]> {
        let ops = Vec::new();
        ops.into_boxed_slice()
    }

    pub fn create_unmapping_operations(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        buffer_pages: &Range<usize>,
    ) -> Box<[MetalSparseMappingOperations]> {
        let ops = Vec::new();
        ops.into_boxed_slice()
    }

    #[cfg(test)]
    pub fn heap_capacity_bytes(&self) -> usize {
        self.heap_capacity
    }

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
