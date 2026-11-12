use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkBufferMemory {
    Device,
    Constants,
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
            Self::Constants => (vk_mem::MemoryUsage::AutoPreferDevice, write, mapped),
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
