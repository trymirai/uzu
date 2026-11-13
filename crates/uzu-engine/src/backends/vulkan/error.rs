use std::sync::Arc;

use ash::vk;

use crate::data_type::DataType;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not supported")]
    NotSupported,
    #[error(transparent)]
    Context(#[from] VkContextError),
    #[error(transparent)]
    Buffer(#[from] VkBufferError),
    #[error(transparent)]
    Pipeline(#[from] VkPipelineError),
    #[error(transparent)]
    CommandBuffer(#[from] VkCommandBufferError),
    #[error(transparent)]
    Kernel(#[from] VkKernelError),
    #[error("Vulkan validation failed: {0}")]
    Validation(String),
}

#[derive(Debug, thiserror::Error)]
pub enum VkBufferError {
    #[error("cannot allocate a buffer: {0}")]
    Allocation(#[source] vk::Result),
    #[error("buffers must not be empty")]
    Empty,
}

#[derive(Debug, thiserror::Error)]
pub enum VkCommandBufferError {
    #[error("the completion thread stopped before the command buffer completed")]
    Wait,
    #[error("buffer range {start}..{end} is reversed or exceeds a buffer of {size} bytes")]
    BufferRange {
        start: u64,
        end: u64,
        size: u64,
    },
    #[error("cannot dispatch {0}: {1}")]
    Dispatch(&'static str, #[source] Box<VkCommandBufferError>),
    #[error("dispatch of {groups:?} groups exceeds the device limit {limit:?}")]
    DispatchGroups {
        groups: [u32; 3],
        limit: [u32; 3],
    },
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, thiserror::Error)]
pub enum VkContextError {
    #[error("cannot spawn the completion thread: {0}")]
    CompletionThreadSpawn(#[source] std::io::Error),

    #[error("cannot create the Vulkan device: {0}")]
    DeviceCreate(#[source] vk::Result),

    #[error("cannot load the Vulkan library: {0}")]
    EntryLoad(#[source] ash::LoadingError),

    #[error("cannot create the Vulkan instance: {0}")]
    InstanceCreate(#[source] vk::Result),

    #[error("cannot create the memory allocator: {0}")]
    MemoryAllocatorCreate(#[source] vk::Result),

    #[error("cannot create the pipeline cache: {0}")]
    PipelineCacheCreate(#[source] vk::Result),

    #[error("cannot read the pipeline cache {0}: {1}")]
    PipelineCacheRead(std::path::PathBuf, #[source] std::io::Error),

    #[error("cannot enumerate Vulkan devices: {0}")]
    DeviceEnumeration(#[source] vk::Result),

    #[error("cannot query device capabilities: {0}")]
    CapabilityQuery(#[source] vk::Result),

    #[error("cannot enumerate Vulkan layers or instance extensions: {0}")]
    InstanceEnumeration(#[source] vk::Result),

    #[error("cannot create the debug messenger: {0}")]
    DebugMessengerCreate(#[source] vk::Result),

    #[error("no Vulkan 1.4 device has the features the backend needs")]
    NoSuitableDevice,

    #[error("the Vulkan validation layer is not installed")]
    ValidationNotSupported,

    #[error("GPU work that nobody awaited failed, so buffers may hold wrong data: {0}")]
    Failed(#[source] Arc<Error>),
}

#[derive(Debug, thiserror::Error)]
pub enum VkPipelineError {
    #[error("shader is not valid SPIR-V: {0}")]
    Spirv(#[source] std::io::Error),
    #[error("compute pipelines cannot require subgroups of {0} invocations on this device")]
    SubgroupSize(u32),
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, thiserror::Error)]
pub enum VkKernelError {
    #[error("kernel {kernel} has no variant for {data_types:?}")]
    KernelVariant {
        kernel: &'static str,
        data_types: Box<[DataType]>,
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
}
