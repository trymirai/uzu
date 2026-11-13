use std::{convert::Infallible, ops::Range, os::raw::c_void, ptr::NonNull, sync::Arc};

use ash::vk;

use crate::{
    backends::{
        common::{
            Buffer, BufferRef, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu, GlobalGpuBuffer,
            GpuBuffer, SparseBuffer,
            allocator::{bump::BumpAllocation, pool::PoolAllocation},
        },
        vulkan::{Error, VkContext, VkDeviceCapabilities, Vulkan},
    },
    utils::downcast::downcast_ref,
};

pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    size: vk::DeviceSize,
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl VkBuffer {
    pub fn new(
        context: Arc<VkContext>,
        size: usize,
    ) -> Result<Arc<Self>, Error> {
        Ok(Self::allocate(context, size, VkBufferMemory::Device)?.0)
    }

    fn allocate(
        context: Arc<VkContext>,
        size: usize,
        memory: VkBufferMemory,
    ) -> Result<(Arc<Self>, *mut c_void), Error> {
        let size = size as u64;
        let (buffer, allocation, info) = context.allocate_buffer(size, memory)?;
        let device_address = unsafe {
            context.device().get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let buffer = Arc::new(Self {
            context,
            allocation,
            buffer,
            device_address,
            size,
        });
        Ok((buffer, info.mapped_data))
    }

    pub fn context(&self) -> &Arc<VkContext> {
        &self.context
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn resolve(
        buffer: &(impl Buffer<Backend = Vulkan> + ?Sized),
        range: std::range::Range<usize>,
    ) -> (&Arc<VkBuffer>, Range<u64>) {
        let (buffer, offset) = if let Some(buffer) = downcast_ref::<Arc<VkBuffer>>(buffer) {
            (buffer, 0)
        } else if let Some(buffer) = downcast_ref::<VkBufferCpuGpu>(buffer) {
            (&buffer.buffer, 0)
        } else if let Some(allocation) = downcast_ref::<BumpAllocation<VkBufferCpuGpu>>(buffer) {
            (&allocation.page().buffer, allocation.range().start)
        } else if let Some(allocation) = downcast_ref::<PoolAllocation<Arc<VkBuffer>>>(buffer) {
            (allocation.page(), allocation.range().start)
        } else {
            unreachable!("unsupported Vulkan buffer type")
        };
        (buffer, (offset + range.start) as u64..(offset + range.end) as u64)
    }

    pub fn resolve_ref<'a>(buffer: impl BufferRef<Backend = Vulkan> + 'a) -> (&'a Arc<Self>, Range<u64>) {
        let (buffer, range) = buffer.parts();
        Self::resolve(buffer, range)
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe { self.context.free_buffer(self.buffer, &mut self.allocation) };
    }
}

impl Buffer for Arc<VkBuffer> {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        VkBuffer::size(self) as usize
    }
}

impl GpuBuffer for Arc<VkBuffer> {}

impl GlobalGpuBuffer for Arc<VkBuffer> {}

pub struct VkBufferCpuGpu {
    buffer: Arc<VkBuffer>,
    cpu_ptr: NonNull<c_void>,
}

unsafe impl Send for VkBufferCpuGpu {}
unsafe impl Sync for VkBufferCpuGpu {}

impl VkBufferCpuGpu {
    pub fn new(
        context: Arc<VkContext>,
        size: usize,
        memory: VkBufferMemory,
    ) -> Result<Self, Error> {
        let (buffer, mapped) = VkBuffer::allocate(context, size, memory)?;
        Ok(Self {
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
        Buffer::size(&self.buffer)
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
        let context = self.buffer.context();
        if context.physical_device().capabilities.contains(VkDeviceCapabilities::UNIFIED_MEMORY) {
            return Ok(self.buffer);
        }
        let mut device = VkBuffer::new(context.clone(), Buffer::size(&self))?;
        let mut encoding = context.create_command_buffer(None, None)?;
        encoding.encode_copy(&self, &mut device);
        encoding.end_encoding().try_submit()?;
        Ok(device)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkBufferMemory {
    Device,
    Staging,
    Readback,
}

impl VkBufferMemory {
    pub fn allocation_create_info(
        self,
        discrete: bool,
    ) -> vk_mem::AllocationCreateInfo {
        let mapped = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let write = vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
        let (usage, flags, required_flags) = match self {
            Self::Device if discrete => (
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::WITHIN_BUDGET,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ),
            Self::Device => {
                (vk_mem::MemoryUsage::AutoPreferDevice, vk_mem::AllocationCreateFlags::empty(), Default::default())
            },
            Self::Staging => (vk_mem::MemoryUsage::AutoPreferHost, write, mapped),
            Self::Readback => (
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                mapped,
            ),
        };
        vk_mem::AllocationCreateInfo {
            usage,
            flags,
            required_flags,
            ..Default::default()
        }
    }
}

pub struct VkSparseBuffer {
    never: Infallible,
}

impl Buffer for VkSparseBuffer {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        match self.never {}
    }
}

impl GpuBuffer for VkSparseBuffer {}

impl SparseBuffer for VkSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), Error> {
        match self.never {}
    }
}
