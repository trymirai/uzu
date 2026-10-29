use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{
    common::{Buffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu},
    vulkan::{Error, VkBuffer, VkBufferMemory, VkContext, Vulkan},
};

/// A staging buffer the host fills, which `handoff` copies into its device-local buffer.
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

impl CpuBuffer for VkBufferInitializer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.staging.cpu_ptr()
    }
}

impl VkBufferInitializer {
    fn submit_copy(&self) -> Result<(), Error> {
        let mut encoding = self.buffer.context().create_command_buffer(None, None)?;
        encoding.encode_copy(&self.staging, 0..VkBuffer::size(&self.staging), &self.buffer, 0)?;
        drop(encoding.end_encoding().try_submit()?);
        Ok(())
    }
}

impl GlobalBufferCpuGpu for VkBufferInitializer {
    /// Submits the copy without waiting: later command buffers on the queue see it, and the completion thread frees
    /// the staging buffer once it is done. A failed submission fails every later command buffer of the context.
    fn handoff(self) -> Arc<VkBuffer> {
        if let Err(error) = self.submit_copy() {
            self.buffer.context().fail_uploads(error);
        }
        self.buffer
    }
}
