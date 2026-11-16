use std::{ops::Range, os::raw::c_void, sync::Arc};

use ash::vk;
use vk_mem::Alloc;

use crate::{
    backends::{
        common::{
            Buffer, BufferRef, GlobalGpuBuffer, GpuBuffer,
            allocator::{bump::BumpAllocation, pool::PoolAllocation},
        },
        vulkan::{Error, VkBufferCpuGpu, VkBufferError, VkContext, VkDevice, Vulkan},
    },
    utils::downcast::downcast_ref,
};

const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

pub struct VkBuffer {
    device: Arc<VkDevice>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    size: usize,
}

// The VMA allocation is an opaque handle that VMA synchronizes internally.
unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl VkBuffer {
    pub fn new(
        context: &VkContext,
        size: usize,
    ) -> Result<Arc<Self>, Error> {
        let allocation_info = if context.device().physical_device.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            // Fail instead of spilling into system memory, which a discrete GPU reads over PCIe.
            vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::WITHIN_BUDGET,
                required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            }
        } else {
            vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            }
        };
        Ok(Self::allocate(context, size, &allocation_info)?.0)
    }

    pub fn allocate(
        context: &VkContext,
        size: usize,
        allocation_info: &vk_mem::AllocationCreateInfo,
    ) -> Result<(Arc<Self>, *mut c_void), Error> {
        if size == 0 {
            return Err(VkBufferError::Empty.into());
        }
        let device = context.device();
        let buffer_info = vk::BufferCreateInfo::default().size(size as u64).usage(BUFFER_USAGE);
        let create = || unsafe { device.memory_allocator.create_buffer(&buffer_info, allocation_info) };
        let (buffer, allocation) = match create() {
            // Work that is still executing may hold the memory this allocation needs.
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                context.wait_for_submitted()?;
                create()
            },
            created => created,
        }
        .map_err(VkBufferError::Allocation)?;
        let mapped = device.memory_allocator.get_allocation_info(&allocation).mapped_data;
        let device_address =
            unsafe { device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };
        let buffer = Arc::new(Self {
            device: device.clone(),
            allocation,
            buffer,
            device_address,
            size,
        });
        Ok((buffer, mapped))
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    // Whether the GPU reads this buffer from its own memory.
    pub fn device_local(&self) -> bool {
        let memory_type = self.device.memory_allocator.get_allocation_info(&self.allocation).memory_type;
        let memory = unsafe { self.device.memory_allocator.get_memory_properties() };
        memory.memory_types[memory_type as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    pub fn resolve(
        buffer: &(impl Buffer<Backend = Vulkan> + ?Sized),
        range: std::range::Range<usize>,
    ) -> (&Arc<VkBuffer>, Range<u64>) {
        let (buffer, offset) = if let Some(buffer) = downcast_ref::<Arc<VkBuffer>>(buffer) {
            (buffer, 0)
        } else if let Some(buffer) = downcast_ref::<VkBufferCpuGpu>(buffer) {
            (buffer.buffer(), 0)
        } else if let Some(allocation) = downcast_ref::<BumpAllocation<VkBufferCpuGpu>>(buffer) {
            (allocation.page().buffer(), allocation.range().start)
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
        unsafe { self.device.memory_allocator.destroy_buffer(self.buffer, &mut self.allocation) };
    }
}

impl Buffer for Arc<VkBuffer> {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        self.size
    }
}

impl GpuBuffer for Arc<VkBuffer> {}

impl GlobalGpuBuffer for Arc<VkBuffer> {}
