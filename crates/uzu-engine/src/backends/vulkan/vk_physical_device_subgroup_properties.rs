use ash::vk;

#[derive(Default)]
pub struct VkPhysicalDeviceSubgroupProperties {
    pub size: u32,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub supported_stages: vk::ShaderStageFlags,
}
