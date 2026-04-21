use std::{cell::RefCell, ops::Deref, ptr::NonNull, rc::Rc};

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

const PAGE_SIZE: MTLSparsePageSize = MTLSparsePageSize::KB16;

#[derive(Debug)]
pub struct MetalSparseBuffer {
    buffer: Rc<RefCell<Retained<ProtocolObject<dyn MTLBuffer>>>>,
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
        let page_size_bytes = Self::get_page_size_bytes(PAGE_SIZE);
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            PAGE_SIZE,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        let head_desc = MTLHeapDescriptor::new();
        head_desc.set_type(MTLHeapType::Placement);
        head_desc.set_storage_mode(MTLStorageMode::Shared);
        head_desc.set_size(aligned_capacity);
        head_desc.set_sparse_page_size(PAGE_SIZE);
        head_desc.set_max_compatible_placement_sparse_page_size(PAGE_SIZE);
        let Some(heap) = context.device.new_heap_with_descriptor(&head_desc) else {
            return Err(MetalError::SparseHeapAlloc(aligned_capacity, page_size_bytes));
        };

        Ok(Self {
            buffer: Rc::new(RefCell::new(buffer)),
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
        let tile_size = Self::get_page_size_bytes(PAGE_SIZE);
        let tiles_count = length.div_ceil(tile_size);
        let tiles_offset = offset.div_ceil(tile_size);
        let tiles_heap_offset = self.length / tile_size;

        let op = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Map,
            buffer_range: NSRange::new(tiles_offset, tiles_count),
            heap_offset: tiles_heap_offset,
        };
        self.length += tiles_count * tile_size;

        self.queue.update_buffer_mappings_heap_operations_count(
            &self.buffer.borrow().deref(),
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
            &self.buffer.borrow().deref(),
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

    fn buffer(&self) -> Rc<RefCell<<Self::Backend as Backend>::Buffer>> {
        self.buffer.clone()
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
