use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::Arc,
};

use ash::vk;
use vk_mem::Alloc;

use super::{VkBufferCreateInfo, VkBufferError, VkContext};

pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    size: vk::DeviceSize,
}

impl VkBuffer {
    pub fn new_with_info(
        context: Arc<VkContext>,
        info: &VkBufferCreateInfo<'_>,
    ) -> Result<Self, VkBufferError> {
        Self::new(context, &info.allocation_info, &info.buffer_info)
    }

    pub fn new(
        context: Arc<VkContext>,
        allocation_info: &vk_mem::AllocationCreateInfo,
        buffer_info: &vk::BufferCreateInfo<'_>,
    ) -> Result<Self, VkBufferError> {
        let (buffer, allocation) = unsafe { context.memory_allocator().create_buffer(buffer_info, allocation_info) }
            .map_err(VkBufferError::Allocation)?;
        Ok(Self {
            context,
            allocation,
            buffer,
            size: buffer_info.size,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn map_action_unmap<T>(
        &mut self,
        action: impl FnOnce(&mut [u8]) -> T,
    ) -> Result<T, VkBufferError> {
        let allocator = self.context.memory_allocator();
        let ptr = unsafe { allocator.map_memory(&mut self.allocation) }.map_err(VkBufferError::MemoryMap)?;
        if let Err(error) = allocator.invalidate_allocation(&self.allocation, 0, self.size) {
            unsafe {
                allocator.unmap_memory(&mut self.allocation);
            }
            return Err(VkBufferError::CacheInvalidate(error));
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            action(unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) })
        }));
        let flushed = allocator.flush_allocation(&self.allocation, 0, self.size);
        unsafe {
            allocator.unmap_memory(&mut self.allocation);
        }
        let result = match result {
            Ok(value) => value,
            Err(panic) => resume_unwind(panic),
        };
        flushed.map_err(VkBufferError::CacheFlush)?;
        Ok(result)
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
        self.map_action_unmap(|bytes| bytes[..data.len()].copy_from_slice(data))
    }

    pub fn get_bytes(&mut self) -> Result<Box<[u8]>, VkBufferError> {
        self.map_action_unmap(|bytes| Box::from(&*bytes))
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn get_memory_barrier(
        &self,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
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
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.memory_allocator().destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
