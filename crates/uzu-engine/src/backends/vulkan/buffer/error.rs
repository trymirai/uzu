use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkBufferError {
    #[error("cannot allocate a buffer: {0}")]
    Allocation(#[source] vk::Result),
    #[error("buffers must not be empty")]
    Empty,
}
