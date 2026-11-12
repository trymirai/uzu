#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not supported")]
    NotSupported,
    #[error(transparent)]
    Context(#[from] crate::backends::vulkan::VkContextError),
    #[error(transparent)]
    Buffer(#[from] crate::backends::vulkan::VkBufferError),
    #[error(transparent)]
    Pipeline(#[from] crate::backends::vulkan::VkPipelineError),
    #[error(transparent)]
    CommandBuffer(#[from] crate::backends::vulkan::VkCommandBufferError),
    #[error(transparent)]
    Kernel(#[from] crate::backends::vulkan::VkKernelError),
    #[error("Vulkan validation failed: {0}")]
    Validation(String),
}
