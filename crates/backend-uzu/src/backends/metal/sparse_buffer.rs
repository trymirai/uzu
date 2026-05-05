use std::fmt::Debug;

use bytesize::ByteSize;
use metal::prelude::*;
use rangemap::RangeMap;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer, SparseBufferMappedPages, SparseBufferOperation},
        metal::{Metal, error::MetalError, metal_extensions::SparsePageSizeExt},
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
        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        let heap_desc = MTLHeapDescriptor::new();
        // Sparse buffers must be backed by a Placement heap with a sparse page size set;
        // `MTLHeapType::Sparse` is for sparse textures and trips a runtime assertion when
        // passed to updateBufferMappings.
        heap_desc.set_type(MTLHeapType::Placement);
        heap_desc.set_storage_mode(MTLStorageMode::Private);
        heap_desc.set_size(aligned_capacity);
        heap_desc.set_max_compatible_placement_sparse_page_size(page_size);
        let Some(heap) = context.device.new_heap_with_descriptor(&heap_desc) else {
            return Err(MetalError::SparseHeapAlloc(aligned_capacity, page_size.byte_size()));
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
        context.command_queue4.update_buffer_mappings(&self.buffer, Some(&self.heap), operations);

        Ok(())
    }
}

impl Buffer for MetalSparseBuffer {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn size(&self) -> ByteSize {
        self.buffer.size()
    }

    fn set_label(
        &mut self,
        label: Option<&str>,
    ) {
        self.buffer.set_label(label)
    }
}

impl SparseBuffer for MetalSparseBuffer {
    fn get_mapped_pages(&self) -> &RangeMap<usize, ()> {
        &self.mapped_pages.get_map()
    }

    fn get_page_size(&self) -> ByteSize {
        self.page_size.byte_size()
    }

    fn execute(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        operations: &[SparseBufferOperation],
    ) -> Result<(), MetalError> {
        let mtl_operations = operations.iter().map(MTL4UpdateSparseBufferMappingOperation::from).collect::<Vec<_>>();

        self.execute(context, &mtl_operations)?;
        self.mapped_pages.execute(operations);

        Ok(())
    }
}
