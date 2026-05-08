use std::ops::Range;

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use rangemap::RangeMap;

use crate::{
    backends::metal::{
        error::MetalError,
        metal_extensions::SparsePageSizeExt,
        sparse::sparse_utils::{MetalSparseHeapBufferMapping, MetalSparseHeapMappingParameters},
    },
    prelude::MetalContext,
};

pub(super) struct MetalSparseHeap {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    mapped_pages: RangeMap<usize, MetalSparseHeapBufferMapping>,
}

impl MetalSparseHeap {
    pub fn new(
        context: &MetalContext,
        capacity_bytes: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity_bytes.div_ceil(page_size_bytes) * page_size_bytes;

        let heap_desc = MTLHeapDescriptor::new();
        // Sparse buffers must be backed by a Placement heap with a sparse page size set;
        // `MTLHeapType::Sparse` is for sparse textures and trips a runtime assertion when
        // passed to updateBufferMappings.
        heap_desc.set_type(MTLHeapType::Placement);
        heap_desc.set_storage_mode(MTLStorageMode::Private);
        heap_desc.set_size(aligned_capacity);
        heap_desc.set_max_compatible_placement_sparse_page_size(page_size);
        let heap = context
            .device
            .new_heap_with_descriptor(&heap_desc)
            .ok_or_else(|| MetalError::SparseHeapAlloc(aligned_capacity, page_size.byte_size()))?;
        Ok(Self {
            heap,
            mapped_pages: RangeMap::new(),
        })
    }

    pub fn mapped_pages(&self) -> &RangeMap<usize, MetalSparseHeapBufferMapping> {
        &self.mapped_pages
    }

    pub fn execute(
        &mut self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        cmd_queue: &ProtocolObject<dyn MTL4CommandQueue>,
        operations: &[MetalSparseHeapMappingParameters],
        map: bool,
    ) {
        let mtl_operations: Vec<MTL4UpdateSparseBufferMappingOperation> = operations
            .iter()
            .map(|op| {
                let mode = if map {
                    MTLSparseTextureMappingMode::Map
                } else {
                    MTLSparseTextureMappingMode::Unmap
                };
                MTL4UpdateSparseBufferMappingOperation::new(mode, op.buffer_pages.clone(), op.heap_page_offset)
            })
            .collect();

        cmd_queue.update_buffer_mappings(buffer, Some(&self.heap), &mtl_operations);

        mtl_operations.iter().for_each(|mtl_op| {
            let heap_range = Range {
                start: mtl_op.heap_offset,
                end: mtl_op.heap_offset + mtl_op.buffer_range().len(),
            };
            if map {
                let buffer_mapping = MetalSparseHeapBufferMapping::new(buffer.gpu_address(), mtl_op.buffer_range());
                self.mapped_pages.insert(heap_range, buffer_mapping);
            } else {
                self.mapped_pages.remove(heap_range);
            }
        });
    }
}
