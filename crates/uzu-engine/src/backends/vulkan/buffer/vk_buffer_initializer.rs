use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{
    common::{Buffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu},
    vulkan::{Error, VkBuffer, VkBufferMemory, VkContext, VkDeviceCapabilities, Vulkan},
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
        let device = (!context.physical_device().capabilities.contains(VkDeviceCapabilities::UNIFIED_MEMORY))
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
    fn handoff(self) -> Result<Arc<VkBuffer>, Error> {
        let Some(mut device) = self.device else {
            return Ok(self.host);
        };
        let mut encoding = self.host.context().create_command_buffer(None, None)?;
        encoding.encode_copy(&self.host, &mut device);
        encoding.end_encoding().try_submit()?;
        Ok(device)
    }
}
