use std::ops::RangeInclusive;

use ash::vk;

pub struct VkPhysicalDeviceSubgroupProperties {
    pub size: u32,
    /// The most subgroups of a workgroup whose pipeline requires a subgroup size.
    pub max_compute_workgroup_subgroups: u32,
    pub supported_operations: vk::SubgroupFeatureFlags,
    /// Subgroup sizes a compute pipeline may require, in full subgroups.
    pub requirable_sizes: Option<RangeInclusive<u32>>,
}

impl VkPhysicalDeviceSubgroupProperties {
    pub fn requirable(
        &self,
        size: u32,
    ) -> bool {
        self.requirable_sizes.as_ref().is_some_and(|sizes| sizes.contains(&size))
    }
}
