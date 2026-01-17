use std::sync::Arc;
use ash::vk;
use bytemuck::AnyBitPattern;
use crate::backends::vulkan::context::VkContext;

pub struct VkBuffer {
    device: Arc<ash::Device>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}

impl VkBuffer {
    pub fn ssbo(
        context: &VkContext,
        size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(
            context,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE |
                vk::MemoryPropertyFlags::HOST_CACHED |
                vk::MemoryPropertyFlags::HOST_COHERENT |
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    pub fn new(
        context: &VkContext,
        size: usize,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = {
            let info = vk::BufferCreateInfo::default()
                .size(size as vk::DeviceSize)
                .usage(usage);
            unsafe { context.device().create_buffer(&info, None)? }
        };

        let memory_requirements = unsafe {
            context.device().get_buffer_memory_requirements(buffer)
        };
        let memory_type_index = match context.physical_device().get_memory_type(
            memory_requirements.memory_type_bits, memory_flags
        ) {
            Some(index) => index,
            None => return Err(Box::new(VkBufferError::MemoryTypeIndexNotFound))
        };
        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe {
            context.device().allocate_memory(&memory_allocate_info, None)
        }?;
        unsafe {
            context.device().bind_buffer_memory(buffer, memory, 0)
        }?;

        Ok(Self {
            device: context.device(),
            buffer,
            memory,
            size
        })
    }

    pub fn fill(&mut self, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.fill_with_offset(data, 0u64)
    }

    pub fn fill_with_offset(&mut self, data: &[u8], offset: u64) -> Result<(), Box<dyn std::error::Error>> {
        let len = data.len() as u64;
        unsafe {
            let memory_ptr = self.device.map_memory(self.memory, offset, len, vk::MemoryMapFlags::empty())?;
            let slice = std::slice::from_raw_parts_mut(memory_ptr as *mut u8, len as usize);
            slice.copy_from_slice(data);
            self.device.unmap_memory(self.memory);
        };
        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl VkBuffer {
    pub fn get_write_memory_barrier(&self) -> vk::BufferMemoryBarrier<'_> {
        self.get_memory_barrier(vk::AccessFlags::HOST_WRITE, vk::AccessFlags::SHADER_READ)
    }

    pub fn get_read_memory_barrier(&self) -> vk::BufferMemoryBarrier<'_> {
        self.get_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::HOST_READ)
    }

    fn get_memory_barrier(
        &self,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags
    ) -> vk::BufferMemoryBarrier<'_> {
        vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.buffer)
            .offset(0)
            .size(self.size as vk::DeviceSize)
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VkBufferError {
    #[error("Memory type index not found")]
    MemoryTypeIndexNotFound
}

pub fn bytes_to_slice<T>(bytes: &[u8]) -> &[T]
where T: AnyBitPattern
{
    unsafe {
        let slice = core::slice::from_raw_parts(bytes.as_ptr(), bytes.len());
        bytemuck::cast_slice(slice)
    }
}

pub fn slice_to_bytes<T>(slice: &[T]) -> &[u8] {
    let len = slice.len() * size_of::<T>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
}