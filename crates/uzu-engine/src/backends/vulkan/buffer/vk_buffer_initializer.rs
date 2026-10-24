use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, CommandBufferEncoding, Context, GlobalBufferInitializer},
    vulkan::{Error, VkBuffer, VkBufferMemory, VkContext, Vulkan},
};

/// A staging buffer the host fills, which ending initialization copies into its device-local buffer.
pub struct VkBufferInitializer {
    staging: Arc<VkBuffer>,
    buffer: Arc<VkBuffer>,
}

impl VkBufferInitializer {
    pub fn new(
        context: &Arc<VkContext>,
        size: u64,
    ) -> Result<Self, Error> {
        Ok(Self {
            staging: Arc::new(VkBuffer::new(context.clone(), size, VkBufferMemory::Staging)?),
            buffer: Arc::new(VkBuffer::new(context.clone(), size, VkBufferMemory::Device)?),
        })
    }
}

impl Buffer for VkBufferInitializer {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        Buffer::size(&self.buffer)
    }
}

impl BufferCpuAccessible for VkBufferInitializer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.staging.cpu_ptr()
    }
}

impl GlobalBufferInitializer for VkBufferInitializer {
    /// Submits the copy without waiting: later command buffers on the queue see it, and the completion thread frees
    /// the staging buffer once it is done.
    fn end_initialization(self) -> Result<Arc<VkBuffer>, Error> {
        let mut encoding = self.buffer.context().create_command_buffer(None, None)?;
        encoding.encode_copy(&self.staging, 0..VkBuffer::size(&self.staging), &self.buffer, 0)?;
        drop(encoding.end_encoding().try_submit()?);
        Ok(self.buffer)
    }
}
