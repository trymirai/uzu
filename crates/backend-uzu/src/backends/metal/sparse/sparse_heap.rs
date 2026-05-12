use std::{collections::HashMap, ops::Range};

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::{RangeMap, RangeSet};

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
    // Mappings are tracked per buffer (keyed by gpu_address) so multiple
    // buffers can alias overlapping heap page ranges without losing track of
    // either side.
    buffer_mappings: HashMap<u64, RangeMap<usize, MetalSparseHeapBufferMapping>>,
    free_pages: RangeSet<usize>,
}

impl MetalSparseHeap {
    pub fn new(
        context: &MetalContext,
        capacity_bytes: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let aligned_capacity = capacity_bytes.next_multiple_of(page_size.in_bytes());

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
            .ok_or(MetalError::SparseHeapAlloc(aligned_capacity, page_size.in_bytes()))?;

        let mut free_pages = RangeSet::new();
        free_pages.insert(0..(aligned_capacity / page_size.in_bytes()));
        Ok(Self {
            heap,
            buffer_mappings: HashMap::new(),
            free_pages,
        })
    }

    pub fn free_pages(&self) -> &RangeSet<usize> {
        &self.free_pages
    }

    /// Iterates `(heap_range, mapping)` pairs that belong to a single buffer.
    pub fn mappings_for(
        &self,
        buffer_address: u64,
    ) -> impl Iterator<Item = (Range<usize>, &MetalSparseHeapBufferMapping)> + '_ {
        self.buffer_mappings
            .get(&buffer_address)
            .into_iter()
            .flat_map(|m| m.iter().map(|(r, mapping)| (r.clone(), mapping)))
    }

    pub fn is_empty(&self) -> bool {
        self.buffer_mappings.is_empty()
    }

    /// Command queue calls doesn't have any synchronization.
    /// It is the responsibility of the caller.
    pub fn execute(
        &mut self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        cmd_queue: &ProtocolObject<dyn MTL4CommandQueue>,
        operations: &[MetalSparseHeapMappingParameters],
        map: bool,
    ) {
        if operations.is_empty() {
            return;
        }

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

        let buffer_address = buffer.gpu_address();
        let entry = self.buffer_mappings.entry(buffer_address).or_default();
        mtl_operations.iter().for_each(|mtl_op| {
            let heap_range = mtl_op.heap_offset..(mtl_op.heap_offset + mtl_op.buffer_range().len());
            if map {
                let buffer_range = mtl_op.buffer_range();
                let buffer_mapping = MetalSparseHeapBufferMapping::new(mtl_op.heap_offset, buffer_range.start);
                entry.insert(heap_range.clone(), buffer_mapping);
                self.free_pages.remove(heap_range);
            } else {
                entry.remove(heap_range.clone());
                self.free_pages.insert(heap_range);
            }
        });

        if entry.is_empty() {
            self.buffer_mappings.remove(&buffer_address);
        }
    }
}
