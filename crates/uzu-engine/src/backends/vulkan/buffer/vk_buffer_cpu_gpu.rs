use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use ash::vk;

use crate::backends::{
    common::{Buffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu, GpuBuffer},
    vulkan::{Error, VkBuffer, VkContext, Vulkan},
};

pub struct VkBufferCpuGpu {
    context: Arc<VkContext>,
    buffer: Arc<VkBuffer>,
    cpu_ptr: NonNull<c_void>,
}

// The pointer is a persistent mapping of `buffer`, which lives as long as this struct.
unsafe impl Send for VkBufferCpuGpu {}
unsafe impl Sync for VkBufferCpuGpu {}

impl VkBufferCpuGpu {
    // Written by the CPU, read by the GPU.
    pub fn staging(
        context: Arc<VkContext>,
        size: usize,
    ) -> Result<Self, Error> {
        Self::new(context, size, vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE)
    }

    // Written by the GPU, read by the CPU.
    pub fn readback(
        context: Arc<VkContext>,
        size: usize,
    ) -> Result<Self, Error> {
        Self::new(context, size, vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM)
    }

    fn new(
        context: Arc<VkContext>,
        size: usize,
        host_access: vk_mem::AllocationCreateFlags,
    ) -> Result<Self, Error> {
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::MAPPED | host_access,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let (buffer, mapped) = VkBuffer::allocate(&context, size, &allocation_info)?;
        Ok(Self {
            context,
            buffer,
            cpu_ptr: NonNull::new(mapped).expect("VMA maps host-visible MAPPED allocations"),
        })
    }

    pub fn buffer(&self) -> &Arc<VkBuffer> {
        &self.buffer
    }
}

impl Buffer for VkBufferCpuGpu {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        self.buffer.size()
    }
}

impl GpuBuffer for VkBufferCpuGpu {}

impl CpuBuffer for VkBufferCpuGpu {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.cpu_ptr
    }
}

impl GlobalBufferCpuGpu for VkBufferCpuGpu {
    fn handoff(self) -> Result<Arc<VkBuffer>, Error> {
        if self.buffer.device_local() {
            return Ok(self.buffer);
        }
        let mut device = VkBuffer::new(&self.context, self.size())?;
        let mut encoding = self.context.create_command_buffer(None, None)?;
        encoding.encode_copy(&self, &mut device);
        encoding.end_encoding().try_submit()?;
        Ok(device)
    }
}
