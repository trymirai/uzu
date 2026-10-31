use ash::vk;

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

    #[error("cannot read the pipeline cache: {0}")]
    PipelineCacheRead(#[source] std::io::Error),

    #[error(transparent)]
    Device(#[from] crate::backends::vulkan::VkDeviceError),

    #[error("cannot enumerate Vulkan devices: {0}")]
    DeviceEnumeration(#[source] vk::Result),

    #[error("cannot enumerate Vulkan layers or instance extensions: {0}")]
    InstanceEnumeration(#[source] vk::Result),

    #[error("cannot create the debug messenger: {0}")]
    DebugMessengerCreate(#[source] vk::Result),

    #[error("no Vulkan 1.4 device has the features the backend needs")]
    NoSuitableDevice,

    #[error("the Vulkan validation layer is not installed")]
    ValidationNotSupported,

    #[error("a buffer upload failed, so buffers may hold no data: {0}")]
    UploadFailed(String),
}
