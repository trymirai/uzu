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
    /// Asahi's active GPU cores and Apple family, used by the copied Metal tile policy.
    pub asahi_topology: Option<(u32, u32)>,
    /// Whether shaders may declare round-to-nearest-even for 16- and 32-bit float results, as every generated kernel
    /// does.
    pub shader_rounding_mode_rte_float16: bool,
    pub shader_rounding_mode_rte_float32: bool,
    /// Whether shaders may preserve 16- and 32-bit signed zeros, infinities and NaNs, as every generated kernel does.
    pub shader_signed_zero_inf_nan_preserve_float16: bool,
    pub shader_signed_zero_inf_nan_preserve_float32: bool,
    /// Which widths may set rounding modes independently. The backend conservatively requires `ALL`, so declaring the
    /// 16- and 32-bit modes is valid whatever 64-bit uses.
    pub rounding_mode_independence: vk::ShaderFloatControlsIndependence,
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
        let mut float_controls_properties = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
        let mut drm_properties = vk::PhysicalDeviceDrmPropertiesEXT::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut device_subgroup_properties)
            .push_next(&mut float_controls_properties)
            .push_next(&mut driver_properties);
        if extensions.iter().any(|extension| extension == "VK_EXT_physical_device_drm") {
            properties2 = properties2.push_next(&mut drm_properties);
        }
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
        let asahi_topology = if unsafe { CStr::from_ptr(driver_properties.driver_name.as_ptr()) } == c"Honeykrisp" {
            asahi_topology(&drm_properties)?
        } else {
            None
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
            shader_int8: vk12features.shader_int8 == 1,
            storage_buffer8_bit_access: vk12features.storage_buffer8_bit_access == 1,
            storage_push_constant8: vk12features.storage_push_constant8 == 1,
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
            asahi_topology,
            memory_properties,
            shader_rounding_mode_rte_float16: float_controls_properties.shader_rounding_mode_rte_float16 == 1,
            shader_rounding_mode_rte_float32: float_controls_properties.shader_rounding_mode_rte_float32 == 1,
            shader_signed_zero_inf_nan_preserve_float16: float_controls_properties
                .shader_signed_zero_inf_nan_preserve_float16
                == 1,
            shader_signed_zero_inf_nan_preserve_float32: float_controls_properties
                .shader_signed_zero_inf_nan_preserve_float32
                == 1,
            rounding_mode_independence: float_controls_properties.rounding_mode_independence,
        })
    }
}

#[cfg(target_os = "linux")]
fn asahi_topology(drm: &vk::PhysicalDeviceDrmPropertiesEXT<'_>) -> Result<Option<(u32, u32)>, VkContextError> {
    use std::{fs::File, io, os::fd::AsRawFd};

    unsafe extern "C" {
        fn ioctl(
            fd: std::ffi::c_int,
            request: std::ffi::c_ulong,
            ...
        ) -> std::ffi::c_int;
    }
    if drm.has_render == 0 {
        return Err(io::Error::other("Asahi device has no DRM render node").into());
    }
    let file = File::open(format!("/dev/char/{}:{}", drm.render_major, drm.render_minor))?;
    // drm_asahi_params_global's stable prefix: 40 header bytes followed by 64 u64 core masks.
    // u64 storage keeps both the prefix and masks aligned; GET_PARAMS accepts a userspace prefix.
    let mut params = [0u64; 69];
    // drm_asahi_get_params: (u32 group=0, u32 pad=0, u64 pointer, u64 size).
    let mut request = [0u64, params.as_mut_ptr() as u64, size_of_val(&params) as u64];
    // SAFETY: request is the 24-byte GET_PARAMS ABI, pointing to writable, aligned storage of its stated size.
    if unsafe { ioctl(file.as_raw_fd(), 0x4018_6440, request.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let header: &[u32] = bytemuck::cast_slice(&params[..5]);
    let clusters = header[7] as usize;
    if request[2] < size_of_val(&params) as u64 || clusters > 64 {
        return Err(io::Error::other("Asahi returned an incomplete GPU topology").into());
    }
    let cores = params[5..5 + clusters].iter().map(|mask| mask.count_ones()).sum();
    let family = match header[2] {
        13 => 7,
        14 => 8,
        15 => 9,
        _ => return Ok(None),
    };
    Ok(Some((cores, family)))
}

#[cfg(not(target_os = "linux"))]
fn asahi_topology(_drm: &vk::PhysicalDeviceDrmPropertiesEXT<'_>) -> Result<Option<(u32, u32)>, VkContextError> {
    Ok(None)
}
