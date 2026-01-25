use std::os::raw::{c_char, c_void};
use std::sync::Arc;
use ash::{khr, vk};
use crate::backends::vulkan::ffi;
use crate::backends::vulkan::logger::{VkLogger, VkPrintlnLogger};
use crate::backends::vulkan::physical_device::{VkPhysicalDevice, VkPhysicalDeviceFeatures};

const VK_LAYER_KHRONOS_VALIDATION: &str = "VK_LAYER_KHRONOS_validation";

/// https://docs.vulkan.org/refpages/latest/refpages/index.html
pub struct VkContext {
    physical_device: VkPhysicalDevice,
    device: Arc<ash::Device>,
    memory_allocator: Option<vk_mem::Allocator>,
    queue: vk::Queue,
    queue_family_index: u32
}

impl VkContext {
    pub fn new(create_info: VkContextCreateInfo) -> Result<Self, VkContextError> {
        let entry = get_entry()?;
        let instance = create_instance(&entry, create_info.with_validation, create_info.logger)?;
        let physical_device = get_physical_device(&instance, &create_info.required_extensions, &create_info.required_features)?;
        let (device, queue_family_index) = get_logical_device(&instance, &physical_device, &create_info.required_extensions, &create_info.required_features)?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let memory_allocator = create_memory_allocator(&instance, &device, physical_device.device)?;

        Ok(Self {
            physical_device,
            device: Arc::new(device),
            memory_allocator: Some(memory_allocator),
            queue,
            queue_family_index
        })
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.clone()
    }

    pub fn memory_allocator(&self) -> &vk_mem::Allocator {
        &self.memory_allocator.as_ref().unwrap()
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn physical_device(&self) -> &VkPhysicalDevice {
        &self.physical_device
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.memory_allocator = None;
            self.device.destroy_device(None);
        }
    }
}

pub struct VkContextCreateInfo {
    pub api_version: u32,
    pub with_validation: bool,
    pub logger: Box<dyn VkLogger>,
    pub required_extensions: Vec<&'static str>,
    pub required_features: VkPhysicalDeviceFeatures,
}

impl Default for VkContextCreateInfo {
    fn default() -> Self {
        Self {
            api_version: vk::API_VERSION_1_2,
            with_validation: true,
            logger: Box::new(VkPrintlnLogger::new()),
            required_extensions: vec![
                khr::shader_float_controls::NAME.to_str().unwrap(),
                khr::shader_float16_int8::NAME.to_str().unwrap(),
                khr::shader_subgroup_extended_types::NAME.to_str().unwrap(),
            ],
            required_features: VkPhysicalDeviceFeatures {
                shader_int16: true,
                shader_float16: true,
                shader_subgroup_extended_types: true,
                storage_buffer16_bit_access: true,
                storage_push_constant16: true,
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VkContextError {
    #[error("Vulkan device creation error: {0}")]
    DeviceCreateError(vk::Result),

    #[error("Vulkan entry loading error: {0}")]
    EntryLoadingError(ash::LoadingError),

    #[error("Vulkan instance creation error: {0}")]
    InstanceCreate(vk::Result),

    #[error("Memory allocator creation error: {0}")]
    MemoryAllocatorCreate(vk::Result),

    #[error("Vulkan physical devices not found: {0}")]
    PhysicalDevicesNotFound(vk::Result),

    #[error("Vulkan physical devices queue not found")]
    PhysicalDeviceQueueNotFound,

    #[error("Vulkan suitable physical devices not found")]
    PhysicalDeviceSuitableNotFound,

    #[error("Validation layer is not supported")]
    ValidationNotSupported
}

fn get_entry() -> Result<ash::Entry, VkContextError> {
    #[cfg(any(target_os = "macos"))]
    // default loader tries to load lib from /usr/lib/, but on macOS this folder is protected by SIP
    let entry_result = unsafe { ash::Entry::load_from("/usr/local/lib/libvulkan.dylib") };

    #[cfg(not(any(target_os = "macos")))]
    let entry_result = unsafe { ash::Entry::load() };

    match entry_result {
        Ok(entry) => Ok(entry),
        Err(err) => Err(VkContextError::EntryLoadingError(err))
    }
}

fn create_instance(
    entry: &ash::Entry,
    with_validation: bool,
    logger: Box<dyn VkLogger>
) -> Result<ash::Instance, VkContextError> {
    let mut instance_extensions: Vec<*const c_char> = Vec::new();
    instance_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
    instance_extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());

    let mut instance_layers: Vec<*const c_char> = Vec::new();
    let mut instance_nexts = Vec::new();

    if with_validation {
        if !is_layer_supported(&entry, VK_LAYER_KHRONOS_VALIDATION) {
            return Err(VkContextError::ValidationNotSupported)
        }
        instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        instance_layers.push(ffi::str_to_ptr_const_char(VK_LAYER_KHRONOS_VALIDATION));

        let msg_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(crate::backends::vulkan::logger::debug_message_callback),
            p_user_data: &logger as *const _ as *mut c_void,
            ..Default::default()
        };
        instance_nexts.push(msg_create_info);
    }

    let app_info = vk::ApplicationInfo::default()
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let mut instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers)
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    for next in instance_nexts.iter_mut() {
        instance_info = instance_info.push_next(next);
    }

    let instance = match unsafe { entry.create_instance(&instance_info, None) } {
        Ok(inst) => inst,
        Err(result) => return Err(VkContextError::InstanceCreate(result))
    };

    Ok(instance)
}

fn create_memory_allocator(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> Result<vk_mem::Allocator, VkContextError> {
    let info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
    match unsafe {
        vk_mem::Allocator::new(info)
    } {
        Ok(allocator) => Ok(allocator),
        Err(err) => Err(VkContextError::MemoryAllocatorCreate(err))
    }
}

fn get_physical_device(
    instance: &ash::Instance,
    required_extensions: &Vec<&str>,
    required_features: &VkPhysicalDeviceFeatures
) -> Result<VkPhysicalDevice, VkContextError> {
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(result) => return Err(VkContextError::PhysicalDevicesNotFound(result))
    };

    let device_opt = devices.into_iter()
        .map(|device| {
            VkPhysicalDevice::new(instance, device)
        })
        .filter(|physical_device| {
            physical_device.features.contains(required_features) &&
                physical_device.subgroup_properties.supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC) &&
                physical_device.subgroup_properties.supported_stages.contains(vk::ShaderStageFlags::COMPUTE) &&
                required_extensions.iter().all(|&req_ext| physical_device.supported_extensions.contains(&req_ext.to_string()) )
        })
        .min_by_key(|physical_device| {
            match physical_device.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                vk::PhysicalDeviceType::OTHER => 4,
                _ => 5
            }
        });
    if let None = device_opt {
        return Err(VkContextError::PhysicalDeviceSuitableNotFound)
    }

    Ok(device_opt.unwrap())
}

fn get_logical_device(
    instance: &ash::Instance,
    physical_device: &VkPhysicalDevice,
    required_extensions: &Vec<&str>,
    required_features: &VkPhysicalDeviceFeatures
) -> Result<(ash::Device, u32), VkContextError> {
    // find queue family index
    let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device.device) };
    let mut queue_family_index = u32::MAX;
    for (i, properties) in queue_family_properties.iter().enumerate() {
        if properties.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER) {
            queue_family_index = i as u32;
            break;
        }
    }
    if queue_family_index == u32::MAX {
        return Err(VkContextError::PhysicalDeviceQueueNotFound);
    }

    let queue_priorities = [1.0_f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // prepare extensions
    let mut device_extensions: Vec<*const c_char> = Vec::new();
    for &ext in required_extensions {
        device_extensions.push(ffi::str_to_ptr_const_char(ext));
    }

    // (https://vulkan.lunarg.com/doc/view/1.4.321.0/mac/antora/spec/latest/chapters/devsandqueues.html#VUID-VkDeviceCreateInfo-pProperties-04451
    if physical_device.supported_extensions.contains(&khr::portability_subset::NAME.to_str().unwrap().to_string()) {
        device_extensions.push(khr::portability_subset::NAME.as_ptr())
    }

    // prepare features
    let vk10_features = vk::PhysicalDeviceFeatures::default()
        .shader_int16(required_features.shader_int16);
    let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default()
        .storage_buffer16_bit_access(required_features.storage_push_constant16)
        .storage_push_constant16(required_features.storage_push_constant16);
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
        .shader_float16(required_features.shader_float16)
        .shader_subgroup_extended_types(required_features.shader_subgroup_extended_types);
    let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default();

    // prepare device
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extensions)
        .enabled_features(&vk10_features)
        .push_next(&mut vk11_features)
        .push_next(&mut vk12_features)
        .push_next(&mut vk13_features);
    let device = match unsafe { instance.create_device(physical_device.device, &device_create_info, None) } {
        Ok(dev) => dev,
        Err(result) => return Err(VkContextError::DeviceCreateError(result))
    };

    Ok((device, queue_family_index))
}

fn is_layer_supported(entry: &ash::Entry, layer: &str) -> bool {
    let layer_properties = unsafe { entry.enumerate_instance_layer_properties() }
        .unwrap_or(Vec::new());
    if layer_properties.is_empty() {
        return false;
    }

    for prop in layer_properties.iter() {
        let layer_name = ffi::c_char_slice_to_string(&prop.layer_name);
        if layer_name == layer {
            return true
        }
    }

    false
}