use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use ash::vk;
use vk_mem::Alloc;

use super::{Error, VkBufferError, VkContext};

/// Storage buffer that is permanently mapped into host-visible, host-coherent memory.
pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    cpu_ptr: NonNull<c_void>,
    size: vk::DeviceSize,
}

// The mapping is owned by this buffer and stays valid until drop; host access goes through `&`/`&mut`.
unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl VkBuffer {
    pub fn new(
        context: Arc<VkContext>,
        size: vk::DeviceSize,
    ) -> Result<Self, Error> {
        if size == 0 {
            return Err(Error::EmptyBuffer);
        }
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            flags: vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let allocator = context.memory_allocator();
        let (buffer, mut allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &allocation_info) }.map_err(VkBufferError::Allocation)?;
        let Some(cpu_ptr) = NonNull::new(allocator.get_allocation_info(&allocation).mapped_data) else {
            unsafe { allocator.destroy_buffer(buffer, &mut allocation) };
            return Err(VkBufferError::MemoryMap(vk::Result::ERROR_MEMORY_MAP_FAILED).into());
        };
        let device_address = unsafe {
            context.device().get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Self {
            context,
            allocation,
            buffer,
            device_address,
            cpu_ptr,
            size,
        })
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

    /// Host-coherent mapping of the whole buffer. Dereferencing it must not overlap GPU work that
    /// accesses the same bytes: read or write only before submission or after completion.
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        self.cpu_ptr
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn fill(
        &mut self,
        data: &[u8],
    ) -> Result<(), VkBufferError> {
        if data.len() as u64 > self.size {
            return Err(VkBufferError::SizeOutOfBounds {
                requested: data.len(),
                size: self.size,
            });
        }
        unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr.as_ptr().cast::<u8>(), data.len()) }.copy_from_slice(data);
        Ok(())
    }

    /// # Safety
    /// No GPU work or host writer may access the buffer during the copy: every command buffer that
    /// writes it must have completed, and no other host thread may write through `cpu_ptr`.
    pub unsafe fn get_bytes(&self) -> Box<[u8]> {
        Box::from(unsafe { std::slice::from_raw_parts(self.cpu_ptr.as_ptr().cast::<u8>(), self.size as usize) })
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.memory_allocator().destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
