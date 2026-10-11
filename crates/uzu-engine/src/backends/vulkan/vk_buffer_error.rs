use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkBufferError {
    #[error("Buffer allocation error: {0:?}")]
    Allocation(#[source] vk::Result),

    #[error("Can not map memory: {0:?}")]
    MemoryMap(#[source] vk::Result),

    #[error("Cannot copy {requested} bytes into a buffer of {size} bytes")]
    SizeOutOfBounds {
        requested: usize,
        size: vk::DeviceSize,
    },
}
