use std::ptr::NonNull;

use metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
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

    capacity: usize,
    length: usize,
}

impl MetalSparseBuffer {
    pub fn new(
        context: &MetalContext,
        capacity: usize,
    ) -> Result<MetalSparseBuffer, MetalError> {
        let page_size = MTLSparsePageSize::KB16;
        let page_size_bytes = Self::get_page_size_bytes(page_size);
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        let head_desc = MTLHeapDescriptor::new();
        head_desc.set_type(MTLHeapType::Placement);
        head_desc.set_storage_mode(MTLStorageMode::Shared);
        head_desc.set_size(aligned_capacity);
        head_desc.set_sparse_page_size(page_size);
        let Some(heap) = context.device.new_heap_with_descriptor(&head_desc) else {
            return Err(MetalError::SparseHeapAlloc(aligned_capacity, page_size_bytes));
        };

        Ok(Self {
            buffer,
            heap,
            queue: context.command_queue4.clone(),
            capacity: aligned_capacity,
            length: 0,
        })
    }

    fn map(
        &mut self,
        offset: usize,
        length: usize,
    ) {
        let op = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Map,
            buffer_range: NSRange::new(offset, length),
            heap_offset: self.length,
        };
        self.length += length;
        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::from_ref(&op),
            1,
        )
    }

    #[allow(dead_code)]
    fn unmap(
        &self,
        offset: usize,
        length: usize,
    ) {
        let op = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Unmap,
            buffer_range: NSRange::new(offset, length),
            heap_offset: 0,
        };
        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::from_ref(&op),
            1,
        )
    }

    fn get_page_size_bytes(size: MTLSparsePageSize) -> usize {
        match size {
            MTLSparsePageSize::KB16 => 16 * 1024,
            MTLSparsePageSize::KB64 => 64 * 1024,
            MTLSparsePageSize::KB256 => 256 * 1024,
        }
    }
}

impl SparseBuffer for MetalSparseBuffer {
    type Backend = Metal;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn extend(
        &mut self,
        add_length: usize,
    ) {
        self.map(self.length, add_length);
    }

    fn length(&self) -> usize {
        self.length
    }
}
