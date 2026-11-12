use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{
    common::{Buffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu},
    vulkan::{Error, VkBuffer, VkBufferMemory, VkContext, Vulkan},
};

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

    fn submit_copy(&mut self) -> Result<(), Error> {
        let mut encoding = self.buffer.context().create_command_buffer(None, None)?;
        encoding.encode_copy(&self.staging, &mut self.buffer);
        drop(encoding.end_encoding().try_submit()?);
        Ok(())
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

impl GlobalBufferCpuGpu for VkBufferInitializer {
    fn handoff(mut self) -> Arc<VkBuffer> {
        if let Err(error) = self.submit_copy() {
            self.buffer.context().fail(error);
        }
        self.buffer
    }
}
