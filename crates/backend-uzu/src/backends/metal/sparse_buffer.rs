use std::{fmt::Debug, ptr::NonNull};

use metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use objc2_foundation::NSRange;
use rangemap::RangeMap;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer, SparseBufferMappedPages, SparseBufferOperation},
        metal::{Metal, error::MetalError},
    },
    prelude::MetalContext,
};

#[derive(Debug)]
pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    mapped_pages: SparseBufferMappedPages,
    page_size: MTLSparsePageSize,
}

impl MetalSparseBuffer {
    pub fn new(
        context: &MetalContext,
        capacity: usize,
    ) -> Result<Self, MetalError> {
        let page_size = MTLSparsePageSize::KB256;
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
            mapped_pages: SparseBufferMappedPages::new(),
            page_size,
        })
    }

    fn execute(
        &mut self,
        context: &MetalContext,
        operations: &[MTL4UpdateSparseBufferMappingOperation],
    ) -> Result<(), MetalError> {
        context.command_queue4.update_buffer_mappings_heap_operations_count(
            &self.buffer,
            Some(&self.heap),
            NonNull::new(operations.as_ptr() as *mut _).unwrap(),
            operations.len(),
        );

        Ok(())
    }
}

impl SparseBuffer for MetalSparseBuffer {
    type Backend = Metal;

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.buffer.set_label(label)
    }

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn length(&self) -> usize {
        self.buffer.length()
    }

    fn get_mapped_pages(&self) -> &RangeMap<usize, ()> {
        &self.mapped_pages.get_map()
    }

    fn get_page_size(&self) -> usize {
        get_page_size_bytes(self.page_size)
    }

    fn execute(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        operations: &[SparseBufferOperation],
    ) -> Result<(), MetalError> {
        let mtl_operations = operations
            .iter()
            .map(|op| {
                let mode = if op.map {
                    MTLSparseTextureMappingMode::Map
                } else {
                    MTLSparseTextureMappingMode::Unmap
                };
                MTL4UpdateSparseBufferMappingOperation {
                    mode,
                    buffer_range: NSRange::new(op.range.start, op.range.len()),
                    heap_offset: op.range.start,
                }
            })
            .collect::<Vec<MTL4UpdateSparseBufferMappingOperation>>();

        self.execute(context, &mtl_operations)?;
        self.mapped_pages.execute(operations);

        Ok(())
    }
}

fn get_page_size_bytes(size: MTLSparsePageSize) -> usize {
    match size {
        MTLSparsePageSize::KB16 => 16 * 1024,
        MTLSparsePageSize::KB64 => 64 * 1024,
        MTLSparsePageSize::KB256 => 256 * 1024,
    }
}
