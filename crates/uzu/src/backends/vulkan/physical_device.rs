use ash::vk;
use std::ffi::CStr;

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
    ) -> Self {
        // extensions
        let mut extensions = Vec::new();
        if let Ok(ext_prop_vec) = unsafe { instance.enumerate_device_extension_properties(physical_device) } {
            for ext_prop in &ext_prop_vec {
                let cow_ext_name = unsafe { CStr::from_ptr(ext_prop.extension_name.as_ptr()) }.to_string_lossy();
                extensions.push(cow_ext_name.to_string());
            }
        }

        // properties
        let mut device_subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut vk11_properties = vk::PhysicalDeviceVulkan11Properties::default();
        let mut vk12_properties = vk::PhysicalDeviceVulkan12Properties::default();
        let mut vk13_properties = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut device_subgroup_properties)
            .push_next(&mut vk11_properties)
            .push_next(&mut vk12_properties)
            .push_next(&mut vk13_properties);
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
        };

        Self {
            device: physical_device,
            supported_extensions: extensions,
            properties,
            features,
            subgroup_properties,
            memory_properties,
        }
    }

    pub fn get_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_properties.memory_types[i as usize].property_flags.contains(properties)
            {
                return Some(i as u32);
            }
        }
        None
    }
}

pub struct VkPhysicalDeviceSubgroupProperties {
    pub size: u32,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub supported_stages: vk::ShaderStageFlags,
}

impl Default for VkPhysicalDeviceSubgroupProperties {
    fn default() -> Self {
        Self {
            size: 0,
            supported_operations: Default::default(),
            supported_stages: Default::default(),
        }
    }
}

/// Here is only features that required by app
pub struct VkPhysicalDeviceFeatures {
    // Version 1.0
    pub shader_int16: bool,

    // Version 1.1
    pub storage_buffer16_bit_access: bool,
    pub storage_push_constant16: bool,

    // Version 1.2
    pub shader_float16: bool,
    pub shader_subgroup_extended_types: bool,
}

impl VkPhysicalDeviceFeatures {
    pub fn contains(
        &self,
        other: &Self,
    ) -> bool {
        (self.shader_int16 || !other.shader_int16)
            && (self.storage_buffer16_bit_access || !other.storage_buffer16_bit_access)
            && (self.storage_push_constant16 || !other.storage_push_constant16)
            && (self.shader_float16 || !other.shader_float16)
            && (self.shader_subgroup_extended_types || !other.shader_subgroup_extended_types)
    }
}

impl Default for VkPhysicalDeviceFeatures {
    fn default() -> Self {
        Self {
            shader_int16: true,
            storage_buffer16_bit_access: true,
            storage_push_constant16: true,
            shader_float16: true,
            shader_subgroup_extended_types: true,
        }
    }
}
