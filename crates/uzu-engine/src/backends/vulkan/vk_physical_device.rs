use std::ffi::CStr;

use ash::{ext, khr, vk};

use super::{
    VkContextError, VkCooperativeMatrix2Features, VkDeviceCapabilities, VkPhysicalDeviceFeatures,
    VkPhysicalDeviceSubgroupProperties, VkShaderBfloat16Features,
};
use crate::data_type::DataType;

pub struct VkPhysicalDevice {
    pub device: vk::PhysicalDevice,
    pub supported_extensions: Vec<String>,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub features: VkPhysicalDeviceFeatures,
    pub subgroup_properties: VkPhysicalDeviceSubgroupProperties,
    pub capabilities: VkDeviceCapabilities,
}
impl VkPhysicalDevice {
    pub fn float_tensor_cores(
        &self,
        data_type: DataType,
    ) -> bool {
        match data_type {
            DataType::BF16 => self.capabilities.contains(VkDeviceCapabilities::BF16_TENSOR_CORES),
            DataType::F16 => self.capabilities.contains(VkDeviceCapabilities::FP16_TENSOR_CORES),
            _ => false,
        }
    }

    pub fn new(
        entry: &ash::Entry,
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
        let has_extension = |name: &CStr| extensions.iter().any(|extension| extension.as_bytes() == name.to_bytes());

        // properties
        let mut device_subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut float_controls = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
        let mut vk13properties = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vk13properties)
            .push_next(&mut device_subgroup_properties)
            .push_next(&mut float_controls)
            .push_next(&mut driver_properties);
        let (properties, mut subgroup_properties) = {
            unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) }
            (
                properties2.properties,
                VkPhysicalDeviceSubgroupProperties {
                    size: device_subgroup_properties.subgroup_size,
                    required_size: None,
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
        let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut bfloat16_features = VkShaderBfloat16Features::default();
        let mut matrix2_features = VkCooperativeMatrix2Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vk11features)
            .push_next(&mut vk12features)
            .push_next(&mut vk13features);
        let matrix_extension = has_extension(khr::cooperative_matrix::NAME);
        if matrix_extension {
            features2 = features2.push_next(&mut matrix_features);
        }
        let matrix2_extension = has_extension(VkCooperativeMatrix2Features::EXTENSION_NAME);
        if matrix2_extension {
            features2 = features2.push_next(&mut matrix2_features);
        }
        let bfloat16_extension = has_extension(VkShaderBfloat16Features::EXTENSION_NAME);
        if bfloat16_extension {
            features2 = features2.push_next(&mut bfloat16_features);
        }
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) }
        let core_features = features2.features;
        if subgroup_properties.size != 32
            && vk13features.subgroup_size_control == vk::TRUE
            && vk13properties.min_subgroup_size <= 32
            && vk13properties.max_subgroup_size >= 32
            && vk13properties.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
        {
            subgroup_properties.required_size = Some(32);
        }
        let full_subgroups = vk13features.compute_full_subgroups == vk::TRUE;
        let simdgroups_32 = subgroup_properties.size == 32 || subgroup_properties.required_size == Some(32);
        // Subgroup-scope cooperative matrices on subgroups of exactly 32 invocations, the kernels' simdgroups.
        let matrices = matrix_extension && matrix_features.cooperative_matrix == vk::TRUE && simdgroups_32;
        let matrix_instance = khr::cooperative_matrix::Instance::new(entry, instance);
        let shapes = match matrices {
            true => unsafe { matrix_instance.get_physical_device_cooperative_matrix_properties(physical_device) }?,
            false => Vec::new(),
        };
        let supports = |depth: u32, input: vk::ComponentTypeKHR, accumulator: vk::ComponentTypeKHR| {
            shapes.iter().any(|matrix| {
                (matrix.m_size, matrix.n_size, matrix.k_size) == (16, 16, depth)
                    && (matrix.a_type, matrix.b_type) == (input, input)
                    && (matrix.c_type, matrix.result_type) == (accumulator, accumulator)
                    && matrix.scope == vk::ScopeKHR::SUBGROUP
                    && matrix.saturating_accumulation == vk::FALSE
            })
        };
        let int8_k32 = supports(32, vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32);
        let int8_k16 = !int8_k32 && supports(16, vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32);
        let unsigned_dot_product = vk13properties.integer_dot_product4x8_bit_packed_unsigned_accelerated == vk::TRUE
            && vk13properties.integer_dot_product4x8_bit_packed_signed_accelerated != vk::TRUE;
        let integer_dot_product = vk13features.shader_integer_dot_product == vk::TRUE && simdgroups_32;
        let capabilities = [
            (
                VkDeviceCapabilities::FLOAT_CONTROLS,
                [
                    float_controls.shader_rounding_mode_rte_float16,
                    float_controls.shader_rounding_mode_rte_float32,
                    float_controls.shader_signed_zero_inf_nan_preserve_float16,
                    float_controls.shader_signed_zero_inf_nan_preserve_float32,
                ]
                .iter()
                .all(|&supported| supported == vk::TRUE),
            ),
            (VkDeviceCapabilities::FULL_SUBGROUPS, full_subgroups),
            (VkDeviceCapabilities::MEMORY_BUDGET, has_extension(ext::memory_budget::NAME)),
            (
                VkDeviceCapabilities::BF16_TENSOR_CORES,
                bfloat16_extension
                    && bfloat16_features.shader_bfloat16_type == vk::TRUE
                    && bfloat16_features.shader_bfloat16_cooperative_matrix == vk::TRUE
                    && supports(16, vk::ComponentTypeKHR::from_raw(1_000_141_000), vk::ComponentTypeKHR::FLOAT32),
            ),
            // PowerVR rounds each FP16 product to FP16 before FP32 accumulation, outside this kernel's error bounds.
            (
                VkDeviceCapabilities::FP16_TENSOR_CORES,
                driver_properties.driver_id != vk::DriverId::IMAGINATION_PROPRIETARY
                    && supports(16, vk::ComponentTypeKHR::FLOAT16, vk::ComponentTypeKHR::FLOAT32),
            ),
            (VkDeviceCapabilities::INT8_TENSOR_CORES_K32, int8_k32),
            (VkDeviceCapabilities::INT8_TENSOR_CORES_K16, int8_k16),
            (
                VkDeviceCapabilities::INT8_ELEMENT_FOLD,
                (int8_k32 || int8_k16)
                    && matrix2_extension
                    && matrix2_features.cooperative_matrix_per_element_operations == vk::TRUE,
            ),
            (VkDeviceCapabilities::INTEGER_DOT_PRODUCT, integer_dot_product),
            (VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT, unsigned_dot_product),
            (
                VkDeviceCapabilities::GEMV_INT8_NATIVE_128,
                integer_dot_product
                    && unsigned_dot_product
                    && subgroup_properties.size == 128
                    && full_subgroups
                    && vk13features.subgroup_size_control == vk::TRUE
                    && (vk13properties.min_subgroup_size..=vk13properties.max_subgroup_size).contains(&128)
                    && vk13properties.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
                    && vk13properties.max_compute_workgroup_subgroups >= 2
                    && subgroup_properties.supported_operations.contains(vk::SubgroupFeatureFlags::CLUSTERED),
            ),
        ]
        .into_iter()
        .filter_map(|(capability, supported)| supported.then_some(capability))
        .collect();
        let features = [
            (VkPhysicalDeviceFeatures::SHADER_INT16, core_features.shader_int16),
            (VkPhysicalDeviceFeatures::STORAGE_BUFFER_16BIT_ACCESS, vk11features.storage_buffer16_bit_access),
            (VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT16, vk11features.storage_push_constant16),
            (VkPhysicalDeviceFeatures::SHADER_INT8, vk12features.shader_int8),
            (VkPhysicalDeviceFeatures::STORAGE_BUFFER_8BIT_ACCESS, vk12features.storage_buffer8_bit_access),
            (VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT8, vk12features.storage_push_constant8),
            (VkPhysicalDeviceFeatures::SHADER_FLOAT16, vk12features.shader_float16),
            (VkPhysicalDeviceFeatures::SHADER_SUBGROUP_EXTENDED_TYPES, vk12features.shader_subgroup_extended_types),
            (VkPhysicalDeviceFeatures::BUFFER_DEVICE_ADDRESS, vk12features.buffer_device_address),
            (VkPhysicalDeviceFeatures::HOST_QUERY_RESET, vk12features.host_query_reset),
            (VkPhysicalDeviceFeatures::VULKAN_MEMORY_MODEL, vk12features.vulkan_memory_model),
            (VkPhysicalDeviceFeatures::VULKAN_MEMORY_MODEL_DEVICE_SCOPE, vk12features.vulkan_memory_model_device_scope),
            (VkPhysicalDeviceFeatures::MAINTENANCE4, vk13features.maintenance4),
            (VkPhysicalDeviceFeatures::SYNCHRONIZATION2, vk13features.synchronization2),
        ]
        .into_iter()
        .filter_map(|(feature, supported)| (supported == vk::TRUE).then_some(feature))
        .collect();

        Ok(Self {
            device: physical_device,
            supported_extensions: extensions,
            properties,
            features,
            subgroup_properties,
            memory_properties,
            capabilities,
        })
    }
}
