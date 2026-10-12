use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkContextError {
    #[error("Command pool creation error: {0}")]
    CommandPoolCreate(#[source] vk::Result),

    #[error("Vulkan device creation error: {0}")]
    DeviceCreateError(#[source] vk::Result),

    #[error("Vulkan entry loading error: {0}")]
    EntryLoadingError(#[source] ash::LoadingError),

    #[error("Vulkan instance creation error: {0}")]
    InstanceCreate(#[source] vk::Result),

    #[error("Memory allocator creation error: {0}")]
    MemoryAllocatorCreate(#[source] vk::Result),

    #[error("Vulkan physical devices not found: {0}")]
    PhysicalDevicesNotFound(#[source] vk::Result),

    #[error("Vulkan capability query failed: {0}")]
    CapabilityQuery(#[from] vk::Result),

    #[error("Vulkan physical devices queue not found")]
    PhysicalDeviceQueueNotFound,

    #[error("Vulkan suitable physical devices not found")]
    PhysicalDeviceSuitableNotFound,

    #[error("Validation layer is not supported")]
    ValidationNotSupported,
}
