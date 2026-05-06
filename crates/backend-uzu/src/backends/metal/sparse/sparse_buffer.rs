use std::{fmt::Debug, ops::Range};

use bytesize::ByteSize;
use metal::prelude::*;

use crate::{
    backends::{
        common::{Backend, Buffer, SparseBuffer},
        metal::{Metal, error::MetalError, metal_extensions::SparsePageSizeExt},
    },
    prelude::MetalContext,
};

#[derive(Debug)]
pub struct MetalSparseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalSparseBuffer {
    pub fn new(
        context: &MetalContext,
        capacity: usize,
        page_size: MTLSparsePageSize,
    ) -> Result<Self, MetalError> {
        let page_size_bytes = page_size.byte_size().as_u64() as usize;
        let aligned_capacity = capacity.div_ceil(page_size_bytes) * page_size_bytes;

        let Some(buffer) = context.device.new_buffer_with_length_options_placement_sparse_page_size(
            aligned_capacity as usize,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
            page_size,
        ) else {
            return Err(MetalError::SparseBufferAlloc(aligned_capacity));
        };

        Ok(Self {
            buffer,
        })
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
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        context.sparse_heaps().map(context, &self.buffer, pages)
    }

    fn unmap(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        context.sparse_heaps().unmap(context, &self.buffer, pages)
    }
}
