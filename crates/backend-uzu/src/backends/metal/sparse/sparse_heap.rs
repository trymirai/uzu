use std::ops::Range;

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use rangemap::RangeMap;

use crate::{
    backends::metal::{error::MetalError, metal_extensions::SparsePageSizeExt},
    prelude::MetalContext,
};

pub(super) struct MetalSparseHeapOperationParameters {
    pub(super) buffer_pages: Range<usize>,
    pub(super) heap_page_offset: usize,
}

pub(super) struct MetalSparseHeap {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    /// map of heap page ranges to buffer gpu address (u64) and buffer pages (Range<usize>)
    mapped_pages: RangeMap<usize, (u64, Range<usize>)>,
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

    pub fn mapped_pages(&self) -> &RangeMap<usize, (u64, Range<usize>)> {
        &self.mapped_pages
    }

    pub fn execute(
        &mut self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        cmd_queue: &ProtocolObject<dyn MTL4CommandQueue>,
        ops_parameters: &[MetalSparseHeapOperationParameters],
        mapping: bool,
    ) {
        if ops_parameters.is_empty() {
            return;
        }

        let operations = ops_parameters
            .iter()
            .map(|op| {
                let mode = if mapping {
                    MTLSparseTextureMappingMode::Map
                } else {
                    MTLSparseTextureMappingMode::Unmap
                };
                let buffer_range = Range {
                    start: op.buffer_pages.start,
                    end: op.buffer_pages.end,
                };
                MTL4UpdateSparseBufferMappingOperation::new(mode, buffer_range, op.heap_page_offset)
            })
            .collect::<Vec<_>>();

        cmd_queue.update_buffer_mappings(buffer, Some(&self.heap), &operations);

        for op in &operations {
            let heap_range = Range {
                start: op.heap_offset,
                end: op.heap_offset + op.buffer_range().len(),
            };
            if mapping {
                self.mapped_pages.insert(heap_range, (buffer.gpu_address(), op.buffer_range()));
            } else {
                self.mapped_pages.remove(heap_range);
            }
        }
    }
}
