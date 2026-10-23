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
    #[error("kernel {kernel} has no variant for {data_types:?}")]
    KernelVariant {
        kernel: &'static str,
        data_types: Box<[crate::data_type::DataType]>,
    },
    #[error("kernel {kernel} requires {condition}")]
    KernelPrecondition {
        kernel: &'static str,
        condition: &'static str,
    },
    #[error("work group {size:?} exceeds the device limit {limit:?} or {invocations} invocations")]
    WorkGroupSize {
        size: [u32; 3],
        limit: [u32; 3],
        invocations: u32,
    },
    #[error("Vulkan validation failed: {0}")]
    Validation(String),
}
