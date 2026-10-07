use ash::vk;

pub struct VkBufferCreateInfo<'a> {
    pub allocation_info: vk_mem::AllocationCreateInfo,
    pub buffer_info: vk::BufferCreateInfo<'a>,
}
impl<'a> VkBufferCreateInfo<'a> {
    pub fn new(
        size: vk::DeviceSize,
        host_readable: bool,
        host_writeable: bool,
    ) -> Self {
        let mut allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let mut buffer_info = vk::BufferCreateInfo::default().usage(vk::BufferUsageFlags::STORAGE_BUFFER).size(size);

        if host_writeable {
            allocation_info.flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            buffer_info.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if host_readable {
            allocation_info.flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            buffer_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        Self {
            allocation_info,
            buffer_info,
        }
    }
}
