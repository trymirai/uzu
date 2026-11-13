use std::{ffi::CStr, ops::RangeInclusive};

use ash::{amd, arm, khr, nv, vk};
use bitflags::bitflags;

use crate::{
    backends::vulkan::{VkContextError, VkCooperativeMatrix2Features, VkShaderBfloat16Features},
    data_type::DataType,
};

pub const API_VERSION_1_4: u32 = vk::make_api_version(0, 1, 4, 0);

pub struct VkPhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub name: Box<str>,
    pub limits: VkDeviceLimits,
    pub capabilities: VkDeviceCapabilities,
    pub gpu_core_count: Option<u32>,
}

impl VkPhysicalDevice {
    pub fn select(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<(Self, u32), VkContextError> {
        let handles = unsafe { instance.enumerate_physical_devices() }.map_err(VkContextError::DeviceEnumeration)?;
        let mut candidates = Vec::new();
        for handle in handles {
            if let Some(candidate) = Self::probe(entry, instance, handle)? {
                candidates.push(candidate);
            }
        }
        candidates
            .into_iter()
            .min_by_key(|(device_type, ..)| match *device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                vk::PhysicalDeviceType::OTHER => 4,
                _ => 5,
            })
            .map(|(_, queue_family_index, physical_device)| (physical_device, queue_family_index))
            .ok_or(VkContextError::NoSuitableDevice)
    }

    fn probe(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Option<(vk::PhysicalDeviceType, u32, Self)>, VkContextError> {
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(VkContextError::CapabilityQuery)?;
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
        if properties.api_version < API_VERSION_1_4 {
            return Ok(None);
        }
        let gpu_core_count = unit_extensions
            .into_iter()
            .zip([
                sm_builtins.shader_sm_count,
                core_properties.active_compute_unit_count,
                core_builtins.shader_core_count,
            ])
            .find_map(|(present, units)| (present && units > 0).then_some(units));

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let unified_memory = memory.memory_heap_count == 1
            && memory.memory_types_as_slice().iter().any(|memory_type| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE)
            });

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
        let limits = VkDeviceLimits {
            max_work_group_count: properties.limits.max_compute_work_group_count,
            max_work_group_size: properties.limits.max_compute_work_group_size,
            max_work_group_invocations: properties.limits.max_compute_work_group_invocations,
            max_work_group_subgroups: vk13properties.max_compute_workgroup_subgroups,
            max_shared_memory_size: properties.limits.max_compute_shared_memory_size,
            subgroup_size: device_subgroup_properties.subgroup_size,
            requirable_subgroup_sizes: vk13properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                .then_some(vk13properties.min_subgroup_size..=vk13properties.max_subgroup_size),
        };
        let matrices = matrix_extension && matrix_features.cooperative_matrix == vk::TRUE;
        let matrix_instance = khr::cooperative_matrix::Instance::new(entry, instance);
        let shapes = if matrices {
            unsafe { matrix_instance.get_physical_device_cooperative_matrix_properties(physical_device) }
                .map_err(VkContextError::CapabilityQuery)?
        } else {
            Vec::new()
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
        let bf16_type = bfloat16_extension && bfloat16_features.shader_bfloat16_type == vk::TRUE;
        let unsigned_dot_product = vk13properties.integer_dot_product4x8_bit_packed_unsigned_accelerated == vk::TRUE
            && vk13properties.integer_dot_product4x8_bit_packed_signed_accelerated != vk::TRUE;
        let capabilities = [
            (VkDeviceCapabilities::DISCRETE, properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU),
            (VkDeviceCapabilities::UNIFIED_MEMORY, unified_memory),
            (VkDeviceCapabilities::SUBGROUP_SIZE_32, limits.subgroup_size_requirable(32)),
            (VkDeviceCapabilities::BF16_TYPE, bf16_type),
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
                int8_k32 && matrix2_extension && matrix2_features.cooperative_matrix_per_element_operations == vk::TRUE,
            ),
            (VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT, unsigned_dot_product),
        ]
        .into_iter()
        .filter_map(|(capability, supported)| supported.then_some(capability))
        .collect();
        let meets_requirements = [
            vk12features.storage_push_constant8,
            vk11features.storage_push_constant16,
            vk12features.shader_float16,
            float_controls.shader_rounding_mode_rte_float16,
            float_controls.shader_rounding_mode_rte_float32,
            float_controls.shader_signed_zero_inf_nan_preserve_float16,
            float_controls.shader_signed_zero_inf_nan_preserve_float32,
        ]
        .iter()
        .all(|&supported| supported == vk::TRUE)
            && device_subgroup_properties.supported_operations.contains(
                vk::SubgroupFeatureFlags::ARITHMETIC
                    | vk::SubgroupFeatureFlags::BALLOT
                    | vk::SubgroupFeatureFlags::SHUFFLE,
            );

        let Some(queue_family_index) = queue_families
            .iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .filter(|_| meets_requirements)
        else {
            return Ok(None);
        };
        let device = Self {
            handle: physical_device,
            name: properties.device_name_as_c_str().unwrap_or_default().to_string_lossy().into(),
            limits,
            capabilities,
            gpu_core_count,
        };
        Ok(Some((properties.device_type, queue_family_index as u32, device)))
    }
}

pub struct VkDeviceLimits {
    pub max_work_group_count: [u32; 3],
    pub max_work_group_size: [u32; 3],
    pub max_work_group_invocations: u32,
    pub max_work_group_subgroups: u32,
    pub max_shared_memory_size: u32,
    pub subgroup_size: u32,
    pub requirable_subgroup_sizes: Option<RangeInclusive<u32>>,
}

impl VkDeviceLimits {
    pub fn subgroup_size_requirable(
        &self,
        size: u32,
    ) -> bool {
        self.requirable_subgroup_sizes.as_ref().is_some_and(|sizes| sizes.contains(&size))
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VkDeviceCapabilities: u32 {
        const SUBGROUP_SIZE_32 = 1 << 0;
        const BF16_TENSOR_CORES = 1 << 1;
        const INT8_TENSOR_CORES_K32 = 1 << 2;
        const INT8_TENSOR_CORES_K16 = 1 << 3;
        const INT8_TENSOR_CORES = Self::INT8_TENSOR_CORES_K32.bits() | Self::INT8_TENSOR_CORES_K16.bits();
        const INT8_ELEMENT_FOLD = 1 << 4;
        const UNSIGNED_DOT_PRODUCT = 1 << 5;
        const BF16_TYPE = 1 << 6;
        const DISCRETE = 1 << 7;
        const UNIFIED_MEMORY = 1 << 8;
    }
}

impl VkDeviceCapabilities {
    pub fn float_tensor_cores(
        self,
        data_type: DataType,
    ) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32) && data_type == DataType::BF16 && self.contains(Self::BF16_TENSOR_CORES)
    }

    pub fn prefers_int8_gemm(self) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32)
            && self.intersects(Self::INT8_TENSOR_CORES)
            && !self.contains(Self::UNSIGNED_DOT_PRODUCT)
    }
}
