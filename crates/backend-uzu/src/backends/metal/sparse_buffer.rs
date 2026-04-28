use std::{fmt::Debug, ops::Range, os::raw::c_void, ptr::NonNull};

use backend_uzu::backends::common::Buffer;
use metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;

use crate::{
    backends::{
        common::SparseBuffer,
        metal::{Metal, error::MetalError},
    },
    prelude::MetalContext,
};

#[derive(Debug)]
struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,

    page_size: MTLSparsePageSize,
    capacity: usize,
    length: usize,
}

impl MetalSparseBuffer {
    pub fn new(
        context: &MetalContext,
        capacity: usize,
    ) -> Result<MetalSparseBuffer, MetalError> {
        let Some(queue) = context.command_queue4.clone() else {
            return Err(MetalError::SparseRequireMtl4Queue);
        };

        let page_size = MTLSparsePageSize::KB256;
        let page_size_bytes = Self::get_page_size_bytes(page_size);
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
            queue: queue.clone(),
            page_size,
            capacity: aligned_capacity,
            length: 0,
        })
    }

    fn map(
        &self,
        pages: Range<usize>,
    ) {
        let operation = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Map,
            buffer_range: NSRange::new(pages.start, pages.len()),
            heap_offset: pages.start,
        };
        let operations = &[operation];
        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::new(operations.as_ptr() as *mut _).unwrap(),
            operations.len(),
        );
    }

    fn get_page_size_bytes(size: MTLSparsePageSize) -> usize {
        match size {
            MTLSparsePageSize::KB16 => 16 * 1024,
            MTLSparsePageSize::KB64 => 64 * 1024,
            MTLSparsePageSize::KB256 => 256 * 1024,
        }
    }
}

impl Buffer for MetalSparseBuffer {
    type Backend = Metal;

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.buffer.set_label(label);
    }

    // Returns 0 because of MTLResourceOptions::STORAGE_MODE_PRIVATE
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.buffer.cpu_ptr()
    }

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn length(&self) -> usize {
        self.buffer.length()
    }
}

impl SparseBuffer for MetalSparseBuffer {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn extend(
        &mut self,
        add_length: usize,
    ) {
        let page_size_bytes = Self::get_page_size_bytes(self.page_size);
        let aligned_capacity = self.capacity.div_ceil(page_size_bytes) * page_size_bytes;
        assert!(self.length + add_length <= aligned_capacity, "SparseBuffer capacity overflow");

        let mapped_pages = self.length.div_ceil(page_size_bytes);
        let new_length = self.length + add_length;
        let new_mapped_pages = new_length.div_ceil(page_size_bytes);
        let new_pages_count = new_mapped_pages - mapped_pages;

        self.length = new_length;
        if new_pages_count == 0 {
            return;
        }

        self.map(mapped_pages..mapped_pages + new_pages_count)
    }
}
