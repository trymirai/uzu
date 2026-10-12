use std::ffi::CStr;

use ash::vk;

use super::{VkContextError, VkPhysicalDeviceFeatures, VkPhysicalDeviceSubgroupProperties};

pub struct VkPhysicalDevice {
    pub device: vk::PhysicalDevice,
    pub supported_extensions: Vec<String>,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub features: VkPhysicalDeviceFeatures,
    pub subgroup_properties: VkPhysicalDeviceSubgroupProperties,
}
impl VkPhysicalDevice {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, VkContextError> {
        // extensions
        let mut extensions = Vec::new();
        let ext_prop_vec = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
        for ext_prop in &ext_prop_vec {
            let cow_ext_name = unsafe { CStr::from_ptr(ext_prop.extension_name.as_ptr()) }.to_string_lossy();
            extensions.push(cow_ext_name.to_string());
        }

        // properties
        let mut device_subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut device_subgroup_properties);
        let (properties, subgroup_properties) = {
            unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) }
            (
                properties2.properties,
                VkPhysicalDeviceSubgroupProperties {
                    size: device_subgroup_properties.subgroup_size,
                    supported_operations: device_subgroup_properties.supported_operations,
                    supported_stages: device_subgroup_properties.supported_stages,
                },
            )
        };

        // memory properties
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // features
        let mut vk11features = vk::PhysicalDeviceVulkan11Features::default();
        let mut vk12features = vk::PhysicalDeviceVulkan12Features::default();
        let mut vk13features = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vk11features)
            .push_next(&mut vk12features)
            .push_next(&mut vk13features);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) }
        let features = VkPhysicalDeviceFeatures {
            shader_int16: features2.features.shader_int16 == 1,
            storage_buffer16_bit_access: vk11features.storage_buffer16_bit_access == 1,
            storage_push_constant16: vk11features.storage_push_constant16 == 1,
            shader_float16: vk12features.shader_float16 == 1,
            shader_subgroup_extended_types: vk12features.shader_subgroup_extended_types == 1,
            buffer_device_address: vk12features.buffer_device_address == 1,
            host_query_reset: vk12features.host_query_reset == 1,
            maintenance4: vk13features.maintenance4 == 1,
            synchronization2: vk13features.synchronization2 == 1,
        };

        Ok(Self {
            device: physical_device,
            supported_extensions: extensions,
            properties,
            features,
            subgroup_properties,
            memory_properties,
        })
    }
}
