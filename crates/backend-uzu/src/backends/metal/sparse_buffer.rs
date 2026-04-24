use std::{
    ptr::NonNull,
    sync::atomic::{AtomicU64, Ordering},
};

use metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use objc2_foundation::NSRange;

use crate::{
    backends::{
        common::{Backend, SparseBuffer},
        metal::{Metal, error::MetalError},
    },
    prelude::MetalContext,
};

#[derive(Debug)]
pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,

    sync_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    sync_counter: AtomicU64,

    page_size: MTLSparsePageSize,
    capacity: usize,
    length: usize,
}

impl MetalSparseBuffer {
    pub fn new(
        context: &MetalContext,
        capacity: usize,
    ) -> Result<MetalSparseBuffer, MetalError> {
        let page_size = MTLSparsePageSize::KB16;
        let page_size_bytes = get_page_size_bytes(page_size);
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        let heap_desc = MTLHeapDescriptor::new();
        heap_desc.set_type(MTLHeapType::Placement);
        heap_desc.set_storage_mode(MTLStorageMode::Private);
        heap_desc.set_size(aligned_capacity);
        heap_desc.set_max_compatible_placement_sparse_page_size(page_size);
        let Some(heap) = context.device.new_heap_with_descriptor(&heap_desc) else {
            return Err(MetalError::SparseHeapAlloc(aligned_capacity, page_size_bytes));
        };

        Ok(Self {
            buffer,
            heap,
            queue: context.command_queue4.clone(),
            sync_event: context.device.new_shared_event().unwrap(),
            sync_counter: AtomicU64::new(0),
            page_size,
            capacity: aligned_capacity,
            length: 0,
        })
    }

    fn execute(
        &mut self,
        operations: &[MTL4UpdateSparseBufferMappingOperation],
    ) {
        let value = self.sync_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::new(operations.as_ptr() as *mut _).unwrap(),
            operations.len(),
        );
        self.queue.signal_event_value(self.sync_event.as_ref(), value);
        self.sync_event.wait_until_signaled_value_timeout_ms(value, u64::MAX);
    }
}

impl SparseBuffer for MetalSparseBuffer {
    type Backend = Metal;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer {
        &mut self.buffer
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn extend(
        &mut self,
        add_length: usize,
    ) {
        let page_size_bytes = get_page_size_bytes(self.page_size);
        let mapped_pages = self.length.div_ceil(page_size_bytes);
        self.length += add_length;

        let new_mapped_pages = self.length.div_ceil(page_size_bytes);
        let new_pages_count = new_mapped_pages - mapped_pages;
        if new_pages_count == 0 {
            return;
        }

        let operation = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Map,
            buffer_range: NSRange::new(mapped_pages, new_pages_count),
            heap_offset: mapped_pages,
        };
        self.execute(&[operation]);
    }

    fn length(&self) -> usize {
        self.length
    }
}

fn get_page_size_bytes(size: MTLSparsePageSize) -> usize {
    match size {
        MTLSparsePageSize::KB16 => 16 * 1024,
        MTLSparsePageSize::KB64 => 64 * 1024,
        MTLSparsePageSize::KB256 => 256 * 1024,
    }
}
