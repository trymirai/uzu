use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Vulkan backend is not available")]
    Unavailable,
}
