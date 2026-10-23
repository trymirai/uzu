use ash::vk;

use crate::backends::vulkan::{VkDeviceCapabilities, VkPhysicalDevice, VkPhysicalDeviceSubgroupProperties};

/// The queried facts an identity-free selector reads: raw capabilities, resource limits, subgroup facts, the optional
/// unit count and the single-buffer and single-allocation bounds. Vendor, driver and device names never enter it.
#[derive(Clone, Copy)]
pub struct VkDeviceFacts<'a> {
    pub capabilities: VkDeviceCapabilities,
    pub limits: &'a vk::PhysicalDeviceLimits,
    pub subgroups: &'a VkPhysicalDeviceSubgroupProperties,
    /// See [`VkPhysicalDevice::compute_units`]; not a measure of compute capacity.
    pub compute_units: Option<u32>,
    pub max_buffer_size: u64,
    pub max_memory_allocation_size: u64,
}

impl<'a> VkDeviceFacts<'a> {
    pub fn new(device: &'a VkPhysicalDevice) -> Self {
        Self {
            capabilities: device.capabilities,
            limits: &device.properties.limits,
            subgroups: &device.subgroup_properties,
            compute_units: device.compute_units,
            max_buffer_size: device.max_buffer_size,
            max_memory_allocation_size: device.max_memory_allocation_size,
        }
    }

    /// The largest buffer one allocation can back.
    pub fn max_buffer_bytes(&self) -> u64 {
        self.max_buffer_size.min(self.max_memory_allocation_size)
    }
}
