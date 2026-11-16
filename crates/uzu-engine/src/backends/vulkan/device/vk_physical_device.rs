use std::{ffi::CStr, ops::RangeInclusive};

use ash::{amd, arm, khr, nv, vk};
use bitflags::bitflags;

use crate::backends::vulkan::{VkContextError, VkCooperativeMatrix2Features, VkKernelError, VkShaderBfloat16Features};

pub const API_VERSION_1_4: u32 = vk::make_api_version(0, 1, 4, 0);

pub struct VkPhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub name: Box<str>,
    pub device_type: vk::PhysicalDeviceType,
    pub queue_family_index: u32,
    pub limits: VkDeviceLimits,
    pub capabilities: VkDeviceCapabilities,
    pub gpu_core_count: Option<u32>,
}

impl VkPhysicalDevice {
    pub fn select(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Self, VkContextError> {
        let handles = unsafe { instance.enumerate_physical_devices() }.map_err(VkContextError::DeviceEnumeration)?;
        let mut candidates = Vec::new();
        for handle in handles {
            if let Some(candidate) = Self::new(entry, instance, handle)? {
                candidates.push(candidate);
            }
        }
        candidates
            .into_iter()
            .min_by_key(|candidate| match candidate.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                vk::PhysicalDeviceType::OTHER => 4,
                _ => 5,
            })
            .ok_or(VkContextError::NoSuitableDevice)
    }

    // Returns None for a device the backend cannot run on.
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Option<Self>, VkContextError> {
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
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vk13properties)
            .push_next(&mut device_subgroup_properties)
            .push_next(&mut float_controls);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) }
        let properties = properties2.properties;
        if properties.api_version < API_VERSION_1_4 {
            return Ok(None);
        }
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let Some(queue_family_index) =
            queue_families.iter().position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            return Ok(None);
        };

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
        // Vulkan 1.4 mandates every other feature that `create_device` enables.
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
        if !meets_requirements {
            return Ok(None);
        }

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
        Ok(Some(Self {
            handle: physical_device,
            name: properties.device_name_as_c_str().unwrap_or_default().to_string_lossy().into(),
            device_type: properties.device_type,
            queue_family_index: queue_family_index as u32,
            limits,
            capabilities,
            gpu_core_count: gpu_core_count(instance, physical_device, has_extension),
        }))
    }

    pub fn create_device(
        &self,
        instance: &ash::Instance,
    ) -> Result<ash::Device, VkContextError> {
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.queue_family_index)
            .queue_priorities(&queue_priorities);

        let bf16_type = self.capabilities.contains(VkDeviceCapabilities::BF16_TYPE);
        let matrices = self
            .capabilities
            .intersects(VkDeviceCapabilities::BF16_TENSOR_CORES | VkDeviceCapabilities::INT8_TENSOR_CORES);
        let element_fold = self.capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD);
        let mut device_extensions = Vec::new();
        if matrices {
            device_extensions.push(khr::cooperative_matrix::NAME.as_ptr());
        }
        if bf16_type {
            device_extensions.push(VkShaderBfloat16Features::EXTENSION_NAME.as_ptr());
        }
        if element_fold {
            device_extensions.push(VkCooperativeMatrix2Features::EXTENSION_NAME.as_ptr());
        }

        let vk10_features = vk::PhysicalDeviceFeatures::default().shader_int16(true);
        let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .storage_push_constant16(true);
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(true)
            .storage_buffer8_bit_access(true)
            .storage_push_constant8(true)
            .shader_float16(true)
            .shader_subgroup_extended_types(true)
            .buffer_device_address(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .maintenance4(true)
            .synchronization2(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .shader_integer_dot_product(true);

        let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut bfloat16_features = VkShaderBfloat16Features::default();
        bfloat16_features.shader_bfloat16_type = vk::TRUE;
        bfloat16_features.shader_bfloat16_cooperative_matrix =
            self.capabilities.contains(VkDeviceCapabilities::BF16_TENSOR_CORES).into();
        let mut matrix2_features = VkCooperativeMatrix2Features::default();
        matrix2_features.cooperative_matrix_per_element_operations = vk::TRUE;
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions)
            .enabled_features(&vk10_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);
        if matrices {
            device_create_info = device_create_info.push_next(&mut matrix_features);
        }
        if bf16_type {
            device_create_info = device_create_info.push_next(&mut bfloat16_features);
        }
        if element_fold {
            device_create_info = device_create_info.push_next(&mut matrix2_features);
        }
        unsafe { instance.create_device(self.handle, &device_create_info, None) }.map_err(VkContextError::DeviceCreate)
    }
}

// Streaming multiprocessors, compute units or shader cores, where the vendor reports them.
fn gpu_core_count(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    has_extension: impl Fn(&CStr) -> bool,
) -> Option<u32> {
    let mut nvidia = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
    let mut amd = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
    let mut arm = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if has_extension(nv::shader_sm_builtins::NAME) {
        properties = properties.push_next(&mut nvidia);
    }
    if has_extension(amd::shader_core_properties2::NAME) {
        properties = properties.push_next(&mut amd);
    }
    if has_extension(arm::shader_core_builtins::NAME) {
        properties = properties.push_next(&mut arm);
    }
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) }
    // A struct that was not queried keeps its zero default.
    [nvidia.shader_sm_count, amd.active_compute_unit_count, arm.shader_core_count].into_iter().find(|&count| count > 0)
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

    pub fn check_work_group(
        &self,
        kernel: &'static str,
        size: [u32; 3],
        subgroup_size: Option<u32>,
    ) -> Result<(), VkKernelError> {
        assert!(!size.contains(&0), "{kernel}: work group {size:?} has a zero dimension");
        let invocations = size.iter().product::<u32>();
        if size.iter().zip(self.max_work_group_size).any(|(&size, limit)| size > limit)
            || invocations > self.max_work_group_invocations
        {
            return Err(VkKernelError::WorkGroupSize {
                kernel,
                size,
                limit: self.max_work_group_size,
                invocations: self.max_work_group_invocations,
            });
        }
        if let Some(subgroup_size) = subgroup_size
            && (!size[0].is_multiple_of(subgroup_size)
                || invocations.div_ceil(subgroup_size) > self.max_work_group_subgroups)
        {
            return Err(VkKernelError::KernelPrecondition {
                kernel,
                condition: "a workgroup of whole subgroups of its required size, within maxComputeWorkgroupSubgroups",
            });
        }
        Ok(())
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
    }
}
