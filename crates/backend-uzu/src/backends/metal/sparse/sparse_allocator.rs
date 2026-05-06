use std::ops::Range;

use bytesize::ByteSize;
use metal::{
    MTL4CommandQueueExt, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLDeviceExt, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLSparsePageSize, MTLStorageMode,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeMap;

use crate::{
    backends::metal::{error::MetalError, metal_extensions::SparsePageSizeExt},
    prelude::MetalContext,
};

pub struct MetalSparseHeap {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    mapped_pages: RangeMap<usize, ()>,
}

pub struct MetalSparseHeapsHolder {
    heaps: Vec<MetalSparseHeap>,
    page_size: MTLSparsePageSize,
}

impl MetalSparseHeapsHolder {
    pub fn new() -> Self {
        let page_size = MTLSparsePageSize::KB256;
        Self {
            heaps: vec![],
            page_size,
        }
    }

    pub fn map(
        &self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        pages: Range<usize>,
    ) -> Result<(), MetalError> {
        Ok(())
    }

    pub fn unmap(
        &self,
        context: &MetalContext,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        pages: Range<usize>,
    ) -> Result<(), MetalError> {
        Ok(())
    }

    pub fn page_size(&self) -> MTLSparsePageSize {
        self.page_size
    }

    fn create_heap(
        context: &MetalContext,
        capacity_bytes: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Retained<ProtocolObject<dyn MTLHeap>>, MetalError> {
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
        context
            .device
            .new_heap_with_descriptor(&heap_desc)
            .ok_or_else(|| MetalError::SparseHeapAlloc(aligned_capacity, page_size.byte_size()))
    }
}
