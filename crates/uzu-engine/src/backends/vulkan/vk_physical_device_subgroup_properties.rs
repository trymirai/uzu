use ash::vk;

pub struct VkPhysicalDeviceSubgroupProperties {
    pub size: u32,
    /// A supported size requested by every compute pipeline when the native size differs from 32.
    pub required_size: Option<u32>,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub supported_stages: vk::ShaderStageFlags,
}
