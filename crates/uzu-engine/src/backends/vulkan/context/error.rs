use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::Error;

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
