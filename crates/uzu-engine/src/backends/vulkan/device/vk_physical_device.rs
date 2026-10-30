use std::ffi::CStr;

use ash::{amd, arm, ext, khr, nv, vk};

use crate::backends::vulkan::{
    VkCooperativeMatrix2Features, VkDeviceCapabilities, VkDeviceError, VkPhysicalDeviceFeatures,
    VkPhysicalDeviceSubgroupProperties, VkShaderBfloat16Features,
};

pub struct VkPhysicalDevice {
    pub device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub features: VkPhysicalDeviceFeatures,
    pub subgroup_properties: VkPhysicalDeviceSubgroupProperties,
    pub capabilities: VkDeviceCapabilities,
    /// Streaming multiprocessors or compute units, where the device reports them.
    pub compute_units: Option<u32>,
}
impl VkPhysicalDevice {
    /// The resource limits, which the identity-free matmul selectors read without naming `properties`.
    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.properties.limits
    }

    /// The first compute queue family; Vulkan 1.4's timestampComputeAndGraphics gives each one timestamps.
    pub fn compute_queue_family(&self) -> Option<u32> {
        self.queue_families
            .iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|index| index as u32)
    }

    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, VkDeviceError> {
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(VkDeviceError::CapabilityQuery)?;
        let has_extension = |name: &CStr| {
            extensions
                .iter()
                .any(|extension| extension.extension_name_as_c_str().is_ok_and(|extension| extension == name))
        };

        let mut device_subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut float_controls = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut vk13properties = vk::PhysicalDeviceVulkan13Properties::default();
        let mut sm_builtins = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
        let mut core_properties = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
        let mut core_builtins = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vk13properties)
            .push_next(&mut device_subgroup_properties)
            .push_next(&mut float_controls);
        let unit_extensions = [
            has_extension(nv::shader_sm_builtins::NAME),
            has_extension(amd::shader_core_properties2::NAME),
            has_extension(arm::shader_core_builtins::NAME),
        ];
        if unit_extensions[0] {
            properties2 = properties2.push_next(&mut sm_builtins);
        }
        if unit_extensions[1] {
            properties2 = properties2.push_next(&mut core_properties);
        }
        if unit_extensions[2] {
            properties2 = properties2.push_next(&mut core_builtins);
        }
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) }
        let properties = properties2.properties;
        let compute_units = unit_extensions
            .into_iter()
            .zip([
                sm_builtins.shader_sm_count,
                core_properties.active_compute_unit_count,
                core_builtins.shader_core_count,
            ])
            .find_map(|(present, units)| (present && units > 0).then_some(units));

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut vk11features = vk::PhysicalDeviceVulkan11Features::default();
        let mut vk12features = vk::PhysicalDeviceVulkan12Features::default();
        let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut bfloat16_features = VkShaderBfloat16Features::default();
        let mut matrix2_features = VkCooperativeMatrix2Features::default();
        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut vk11features).push_next(&mut vk12features);
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
        let subgroup_properties = VkPhysicalDeviceSubgroupProperties {
            size: device_subgroup_properties.subgroup_size,
            max_compute_workgroup_subgroups: vk13properties.max_compute_workgroup_subgroups,
            supported_operations: device_subgroup_properties.supported_operations,
            // Vulkan 1.3 makes subgroup size control and full subgroups mandatory, but not for compute pipelines.
            requirable_sizes: vk13properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                .then_some(vk13properties.min_subgroup_size..=vk13properties.max_subgroup_size),
        };
        let matrices = matrix_extension && matrix_features.cooperative_matrix == vk::TRUE;
        let matrix_instance = khr::cooperative_matrix::Instance::new(entry, instance);
        let shapes = match matrices {
            true => unsafe { matrix_instance.get_physical_device_cooperative_matrix_properties(physical_device) }
                .map_err(VkDeviceError::CapabilityQuery)?,
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
        let int8_k16 = supports(16, vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32);
        let calibrated_timestamps = has_extension(khr::calibrated_timestamps::NAME)
            && unsafe {
                khr::calibrated_timestamps::Instance::new(entry, instance)
                    .get_physical_device_calibrateable_time_domains(physical_device)
            }
            .map_err(VkDeviceError::CapabilityQuery)?
            .contains(&vk::TimeDomainKHR::DEVICE);
        let bf16_type = bfloat16_extension && bfloat16_features.shader_bfloat16_type == vk::TRUE;
        let unsigned_dot_product = vk13properties.integer_dot_product4x8_bit_packed_unsigned_accelerated == vk::TRUE
            && vk13properties.integer_dot_product4x8_bit_packed_signed_accelerated != vk::TRUE;
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
            (VkDeviceCapabilities::SUBGROUP_SIZE_32, subgroup_properties.requirable(32)),
            (VkDeviceCapabilities::MEMORY_BUDGET, has_extension(ext::memory_budget::NAME)),
            (VkDeviceCapabilities::BF16_TYPE, bf16_type),
            (VkDeviceCapabilities::CALIBRATED_TIMESTAMPS, calibrated_timestamps),
            (
                VkDeviceCapabilities::BF16_TENSOR_CORES,
                bf16_type
                    && bfloat16_features.shader_bfloat16_cooperative_matrix == vk::TRUE
                    && supports(16, VkShaderBfloat16Features::COMPONENT_TYPE, vk::ComponentTypeKHR::FLOAT32),
            ),
            (VkDeviceCapabilities::INT8_TENSOR_CORES_K32, int8_k32),
            (VkDeviceCapabilities::INT8_TENSOR_CORES_K16, int8_k16),
            (
                VkDeviceCapabilities::INT8_ELEMENT_FOLD,
                (int8_k32 || int8_k16)
                    && matrix2_extension
                    && matrix2_features.cooperative_matrix_per_element_operations == vk::TRUE,
            ),
            (VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT, unsigned_dot_product),
        ]
        .into_iter()
        .filter_map(|(capability, supported)| supported.then_some(capability))
        .collect();
        let features = [
            (VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT8, vk12features.storage_push_constant8),
            (VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT16, vk11features.storage_push_constant16),
            (VkPhysicalDeviceFeatures::SHADER_FLOAT16, vk12features.shader_float16),
        ]
        .into_iter()
        .filter_map(|(feature, supported)| (supported == vk::TRUE).then_some(feature))
        .collect();

        Ok(Self {
            device: physical_device,
            properties,
            features,
            subgroup_properties,
            queue_families,
            capabilities,
            compute_units,
        })
    }
}
