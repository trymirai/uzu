use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::Arc;
use ash::vk;
use bytemuck::AnyBitPattern;
use vk_mem::Alloc;
use crate::backends::vulkan::context::VkContext;

pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
}

impl VkBuffer {
    pub fn new_with_info(
        context: Arc<VkContext>,
        info: &VkBufferCreateInfo
    ) -> Result<Self, VkBufferError> {
        Self::new(context, &info.allocation_info, &info.buffer_info)
    }

    pub fn new(
        context: Arc<VkContext>,
        allocation_info: &vk_mem::AllocationCreateInfo,
        buffer_info: &vk::BufferCreateInfo,
    ) -> Result<Self, VkBufferError> {
        let allocator = context.memory_allocator();
        let (buffer, allocation) = match unsafe {
            allocator.create_buffer(&buffer_info, &allocation_info)
        } {
            Ok((buffer, allocation)) => (buffer, allocation),
            Err(err) => return Err(VkBufferError::Allocation(err))
        };

        Ok(Self {
            context,
            allocation,
            buffer,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn map_action_unmap<F>(&self, mut action: F) -> Result<(), VkBufferError>
        where F: FnMut(*mut u8) -> ()
    {
        let mut alloc_cell = RefCell::new(self.allocation);
        let ptr = match unsafe {
            self.context.memory_allocator().map_memory(alloc_cell.get_mut())
        } {
            Ok(ptr) => Ok(ptr),
            Err(err) => Err(VkBufferError::MemoryMap(err))
        };

        action(ptr?);

        unsafe {
            self.context.memory_allocator().unmap_memory(alloc_cell.get_mut())
        }
        Ok(())
    }

    pub fn fill(&mut self, data: &[u8]) -> Result<(), VkBufferError> {
        self.map_action_unmap(|ptr| {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
            slice.copy_from_slice(data);
        })?;
        Ok(())
    }

    pub fn get_bytes(&self) -> Result<&[u8], VkBufferError> {
        let mut slice: &[u8] = &[];
        self.map_action_unmap(|ptr| {
            slice = unsafe { std::slice::from_raw_parts(ptr, self.size() as usize) };
        })?;
        Ok(slice)
    }

    pub fn get_bytes_mut(&self) -> Result<&mut [u8], VkBufferError> {
        let mut slice: &mut [u8] = &mut [];
        self.map_action_unmap(|ptr| {
            slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.size() as usize) };
        })?;
        Ok(slice)
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.context.memory_allocator().get_allocation_info(&self.allocation).size
    }

    pub fn get_memory_barrier(
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
            .size(vk::WHOLE_SIZE)
    }

    pub fn get_memory_barrier2(
        &self,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) -> vk::BufferMemoryBarrier2<'_> {
        vk::BufferMemoryBarrier2::default()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)
    }

    pub fn set_name(&mut self, name: &str) {
        unsafe {
            self.context.memory_allocator().set_allocation_user_data(&mut self.allocation, name.as_ptr() as *mut c_void)
        }
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.memory_allocator().destroy_buffer(self.buffer, &mut self.allocation)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VkBufferError {
    #[error("Buffer allocation error: {0:?}")]
    Allocation(vk::Result),

    #[error("Can not map memory: {0:?}")]
    MemoryMap(vk::Result),

    #[error("Memory type index not found")]
    MemoryTypeIndexNotFound
}

pub struct VkBufferCreateInfo<'a> {
    pub allocation_info: vk_mem::AllocationCreateInfo,
    pub buffer_info: vk::BufferCreateInfo<'a>,
}

impl <'a> VkBufferCreateInfo<'a> {
    pub fn new(
        size: vk::DeviceSize,
        host_readable: bool,
        host_writeable: bool,
    ) -> Self {
        let mut allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let mut buffer_info = vk::BufferCreateInfo::default()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(size);

        if host_writeable {
            allocation_info.flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            buffer_info.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if host_readable {
            allocation_info.flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            buffer_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        Self { allocation_info, buffer_info }
    }
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