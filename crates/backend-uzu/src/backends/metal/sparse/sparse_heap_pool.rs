use std::{
    cmp::{max, min},
    ops::Range,
};

use metal::{MTL4CommandQueue, MTLBuffer, MTLDeviceExt, MTLSharedEvent, MTLSparsePageSize};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    backends::metal::{
        error::MetalError,
        metal_extensions::SparsePageSizeExt,
        sparse::{sparse_heap::MetalSparseHeap, sparse_utils::MetalSparseHeapMappingParameters},
    },
    prelude::MetalContext,
};

pub struct MetalSparseHeapPool {
    heaps: Vec<MetalSparseHeap>,
    heap_capacity: usize,
    page_size: MTLSparsePageSize,

    /// Signaled on `command_queue4` after every batch of map/unmap ops.
    /// Lets other queues (and the CPU) wait until the GPU has finished applying pending sparse mappings.
    mapping_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,

    /// Value last scheduled to be signaled on `mapping_event`.
    /// Bumped before each `signal_event_value` call so the latest value always represents
    /// "all map/unmap ops issued so far are complete on the GPU".
    pending_mapping_value: u64,
}

impl MetalSparseHeapPool {
    pub fn new(
        device: &ProtocolObject<dyn metal::MTLDevice>,
        page_size: MTLSparsePageSize,
        heap_capacity: usize,
    ) -> Result<Self, MetalError> {
        let mapping_event = device.new_shared_event().ok_or(MetalError::CannotCreateEvent)?;
        mapping_event.set_signaled_value(0);
        Ok(Self {
            heaps: Vec::new(),
            page_size,
            heap_capacity,
            mapping_event,
            pending_mapping_value: 0,
        })
    }

    pub fn map(
        &mut self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        buffer_pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        if buffer_pages.len() == 0 {
            return Ok(());
        }

        let mut pages_to_map = buffer_pages.clone();
        let heap_capacity_pages = self.heap_capacity_pages();

        // Try to find pages in existing heaps.
        // While `pages_to_map` is not empty in each heap find unmapped pages, collect them into `mappings` and map
        let mut existing_heaps_mappings: Vec<(usize, Box<[MetalSparseHeapMappingParameters]>)> = Vec::new();
        for (i, heap) in self.heaps.iter_mut().enumerate() {
            let mut mappings: Vec<MetalSparseHeapMappingParameters> = Vec::new();

            for free_pages in heap.free_pages().iter() {
                let map_pages_count = min(free_pages.len(), pages_to_map.len());
                let mapping = MetalSparseHeapMappingParameters {
                    buffer_pages: pages_to_map.start..(pages_to_map.start + map_pages_count),
                    heap_page_offset: free_pages.start,
                };
                mappings.push(mapping);

                pages_to_map.start += map_pages_count;
                if pages_to_map.len() == 0 {
                    break;
                }
            }

            heap.execute(buffer, &context.command_queue4, &mappings, true);
            existing_heaps_mappings.push((i, mappings.into_boxed_slice()));

            if pages_to_map.len() == 0 {
                break;
            }
        }

        // If `pages_to_map` still not empty, it's necessary to allocate new heaps
        let new_heaps_required = pages_to_map.len().div_ceil(heap_capacity_pages);
        for _ in 0..new_heaps_required {
            let map_pages_count = min(heap_capacity_pages, pages_to_map.len());
            let buffer_pages = pages_to_map.start..(pages_to_map.start + map_pages_count);
            let op_params = MetalSparseHeapMappingParameters {
                buffer_pages,
                heap_page_offset: 0,
            };

            match MetalSparseHeap::new(context, self.heap_capacity, self.page_size) {
                Ok(mut heap) => {
                    let operations = [op_params];
                    heap.execute(buffer, &context.command_queue4, &operations, true);
                    self.heaps.push(heap);
                    pages_to_map.start += map_pages_count;
                    existing_heaps_mappings.push((self.heaps.len() - 1, Box::new(operations)));
                },
                Err(err) => {
                    // it's necessary to unmap previously mapped pages in existing heaps
                    for (heap_pos, mappings) in existing_heaps_mappings {
                        self.heaps[heap_pos].execute(buffer, &context.command_queue4, &mappings, false);
                    }
                    self.signal_pending(&context.command_queue4);
                    return Err(err);
                },
            };
        }

        self.signal_pending(&context.command_queue4);
        Ok(())
    }

    pub fn unmap(
        &mut self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        buffer_pages: &Range<usize>,
    ) -> Result<(), MetalError> {
        let buffer_address = buffer.gpu_address();

        // Iterate over heaps, find mappings for `buffer_address`, unmap them
        self.heaps.iter_mut().for_each(|heap| {
            let unmappings: Vec<MetalSparseHeapMappingParameters> = heap
                .mappings_for(buffer_address)
                .filter_map(|(heap_range, mapping)| {
                    let mapped_buffer_pages = mapping.buffer_pages_for(&heap_range);
                    let unmap_start = max(buffer_pages.start, mapped_buffer_pages.start);
                    let unmap_end = min(buffer_pages.end, mapped_buffer_pages.end);
                    if unmap_start >= unmap_end {
                        return None;
                    };

                    let offset_within_range = unmap_start - mapped_buffer_pages.start;
                    Some(MetalSparseHeapMappingParameters {
                        buffer_pages: unmap_start..unmap_end,
                        heap_page_offset: heap_range.start + offset_within_range,
                    })
                })
                .collect();
            heap.execute(buffer, &context.command_queue4, &unmappings, false);
        });

        // Remove empty heaps
        self.heaps.retain(|heap| !heap.is_empty());

        self.signal_pending(&context.command_queue4);
        Ok(())
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

    pub fn wait_pending(&self) -> Result<(), MetalError> {
        if self.pending_mapping_value == 0 {
            return Ok(());
        }
        if self.mapping_event.wait_until_signaled_value_timeout_ms(self.pending_mapping_value, 1000) {
            return Ok(());
        }
        Err(MetalError::CommandBufferExecutionFailed("Timed out waiting for sparse buffer mapping updates".to_string()))
    }

    fn signal_pending(
        &mut self,
        cmd_queue: &ProtocolObject<dyn MTL4CommandQueue>,
    ) {
        self.pending_mapping_value += 1;
        cmd_queue.signal_event_value(ProtocolObject::from_ref(&*self.mapping_event), self.pending_mapping_value);
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/metal/sparse/sparse_heap_pool_test.rs"]
mod tests;
