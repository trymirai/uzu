use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{
    common::{Buffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu},
    vulkan::{Error, VkBuffer, VkBufferMemory, VkContext, Vulkan},
};

pub struct VkBufferInitializer {
    host: Arc<VkBuffer>,
    device: Option<Arc<VkBuffer>>,
}

impl VkBufferInitializer {
    pub fn new(
        context: Arc<VkContext>,
        size: usize,
    ) -> Result<Self, Error> {
        let device = (!context.physical_device().unified_memory)
            .then(|| VkBuffer::new(context.clone(), size, VkBufferMemory::Device))
            .transpose()?;
        Ok(Self {
            host: VkBuffer::new(context, size, VkBufferMemory::Staging)?,
            device,
        })
    }
}

impl Buffer for VkBufferInitializer {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        Buffer::size(&self.host)
    }
}

impl CpuBuffer for VkBufferInitializer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.host.cpu_ptr()
    }
}

impl GlobalBufferCpuGpu for VkBufferInitializer {
    fn handoff(self) -> Arc<VkBuffer> {
        let Some(mut device) = self.device else {
            return self.host;
        };
        let context = self.host.context();
        let submitted = context.create_command_buffer(None, None).and_then(|mut encoding| {
            encoding.encode_copy(&self.host, &mut device);
            encoding.end_encoding().try_submit()
        });
        if let Err(error) = submitted {
            context.fail(error);
        }
        device
    }
}
