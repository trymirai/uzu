use ash::vk;

/// Where a buffer's memory lives and how the host accesses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkBufferMemory {
    /// Accessed only by the GPU: global and scratch buffers.
    Device,
    /// Written by the host for one command buffer and read by the GPU.
    Constants,
    /// Written once by the host and copied into a device buffer.
    Staging,
    /// Written by the GPU and read by the host.
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
            // On a discrete GPU, exhausting video memory fails rather than spilling to system memory.
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
