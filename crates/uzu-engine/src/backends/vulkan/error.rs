#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Vulkan(#[from] ash::vk::Result),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    EntryPoint(#[from] std::ffi::NulError),
    #[error(transparent)]
    Context(#[from] super::VkContextError),
    #[error(transparent)]
    Buffer(#[from] super::VkBufferError),
    #[error("timestamp query pool capacity exceeded")]
    TimestampCapacity,
    #[error("timestamp query period is out of range")]
    TimestampRange,
    #[error("the selected queue does not support timestamps")]
    TimestampUnsupported,
}
