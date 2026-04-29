use std::ptr::NonNull;

use backend_uzu::{
    backends::{
        common::{Backend, SparsePagesOperation},
        metal::Metal,
    },
    prelude::MetalContext,
};
use metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use objc2_foundation::NSRange;

use crate::backends::{common::SparsePages, metal::error::MetalError};

#[derive(Debug)]
pub struct MetalSparsePages {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,

    page_size: usize,
    total_pages: usize,
}

impl MetalSparsePages {
    pub fn new(
        context: &MetalContext,
        page_size: MTLSparsePageSize,
        total_pages: usize,
    ) -> Result<MetalSparsePages, MetalError> {
        let Some(queue) = context.command_queue4.clone() else {
            return Err(MetalError::SparseRequireMtl4Queue);
        };

        let page_size_bytes = get_page_size_bytes(page_size);
        let buffer_length = page_size_bytes * total_pages;
        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            buffer_length,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(buffer_length));
        };

        let heap_desc = MTLHeapDescriptor::new();
        heap_desc.set_type(MTLHeapType::Placement);
        heap_desc.set_storage_mode(MTLStorageMode::Private);
        heap_desc.set_size(buffer_length);
        heap_desc.set_max_compatible_placement_sparse_page_size(page_size);
        let Some(heap) = context.device.new_heap_with_descriptor(&heap_desc) else {
            return Err(MetalError::SparseHeapAlloc(buffer_length, page_size_bytes));
        };

        Ok(Self {
            buffer,
            heap,
            queue: queue.clone(),
            page_size: page_size_bytes,
            total_pages,
        })
    }

    fn execute(
        &mut self,
        operations: &[MTL4UpdateSparseBufferMappingOperation],
    ) {
        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::new(operations.as_ptr() as *mut _).unwrap(),
            operations.len(),
        );
    }
}

impl SparsePages for MetalSparsePages {
    type Backend = Metal;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer {
        &mut self.buffer
    }

    fn execute(
        &mut self,
        operations: &[SparsePagesOperation],
    ) {
        let mtl4_operations = operations
            .iter()
            .map(|op| {
                let mode = if op.map {
                    MTLSparseTextureMappingMode::Map
                } else {
                    MTLSparseTextureMappingMode::Unmap
                };
                MTL4UpdateSparseBufferMappingOperation {
                    mode,
                    buffer_range: NSRange::new(op.pages.start, op.pages.len()),
                    heap_offset: op.pages.start,
                }
            })
            .collect::<Vec<_>>();
        self.execute(&mtl4_operations);
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn total_pages(&self) -> usize {
        self.total_pages
    }
}

pub fn get_page_size_bytes(size: MTLSparsePageSize) -> usize {
    match size {
        MTLSparsePageSize::KB16 => 16 * 1024,
        MTLSparsePageSize::KB64 => 64 * 1024,
        MTLSparsePageSize::KB256 => 256 * 1024,
    }
}
