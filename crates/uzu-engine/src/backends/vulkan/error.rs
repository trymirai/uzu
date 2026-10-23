#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not supported")]
    NotSupported,
    #[error(transparent)]
    Vulkan(#[from] ash::vk::Result),
    #[error(transparent)]
    Context(#[from] super::VkContextError),
    #[error(transparent)]
    Buffer(#[from] super::VkBufferError),
    #[error(transparent)]
    Pipeline(#[from] super::VkPipelineError),
    #[error(transparent)]
    CommandBuffer(#[from] super::VkCommandBufferError),
    #[error(transparent)]
    Kernel(#[from] super::VkKernelError),
    #[error("Vulkan validation failed: {0}")]
    Validation(String),
}
