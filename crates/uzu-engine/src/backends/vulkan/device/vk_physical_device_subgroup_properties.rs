use std::ops::RangeInclusive;

pub struct VkPhysicalDeviceSubgroupProperties {
    pub size: u32,
    pub max_compute_workgroup_subgroups: u32,
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
