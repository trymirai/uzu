use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkDeviceError {
    #[error("cannot query device capabilities: {0}")]
    CapabilityQuery(#[source] vk::Result),
}
