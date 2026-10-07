use std::ffi::{CStr, c_char, c_void};

use ash::{ext, khr, vk};
use parking_lot::{Mutex, MutexGuard};

use super::{
    Error, VkCommandBufferResources, VkContextCreateInfo, VkContextError, VkLogger, VkPhysicalDevice,
    VkPhysicalDeviceFeatures,
};

const VK_LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
const HOST_COHERENT_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);
/// https://docs.vulkan.org/refpages/latest/refpages/index.html
pub struct VkContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    create_info: Box<VkContextCreateInfo>,
    physical_device: VkPhysicalDevice,
    command_buffer_resources: Mutex<Vec<VkCommandBufferResources>>,
    debug_utils: ext::debug_utils::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    timestamp_valid_bits: u32,
    device: ash::Device,
    memory_allocator: Option<vk_mem::Allocator>,
    queue: Mutex<vk::Queue>,
    queue_family_index: u32,
}
impl VkContext {
    pub fn new(create_info: VkContextCreateInfo) -> Result<Self, VkContextError> {
        let create_info = Box::new(create_info);
        let api_version = vk::API_VERSION_1_3;
        let required_extensions = vec![
            khr::shader_float_controls::NAME.to_str().unwrap(),
            khr::shader_float16_int8::NAME.to_str().unwrap(),
            khr::shader_subgroup_extended_types::NAME.to_str().unwrap(),
        ];
        let required_features = VkPhysicalDeviceFeatures {
            shader_int16: true,
            shader_float16: true,
            shader_subgroup_extended_types: true,
            storage_buffer16_bit_access: true,
            storage_push_constant16: true,
            buffer_device_address: true,
            host_query_reset: true,
            maintenance4: true,
            synchronization2: true,
        };

        let entry = get_entry()?;
        let instance = create_instance(&entry, api_version, &create_info)?;
        let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = if create_info.with_validation {
            Some(
                unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info(&create_info), None) }
                    .inspect_err(|_| unsafe { instance.destroy_instance(None) })?,
            )
        } else {
            None
        };
        let dispose_instance = || unsafe {
            if let Some(messenger) = debug_messenger {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            instance.destroy_instance(None);
        };
        let physical_device = get_physical_device(&instance, &required_extensions, &required_features)
            .inspect_err(|_| dispose_instance())?;
        let (device, queue_family_index, timestamp_valid_bits) =
            get_logical_device(&instance, &physical_device, &required_extensions, &required_features)
                .inspect_err(|_| dispose_instance())?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let memory_allocator =
            create_memory_allocator(&instance, &device, physical_device.device).inspect_err(|_| unsafe {
                device.destroy_device(None);
                dispose_instance();
            })?;

        Ok(Self {
            _entry: entry,
            instance,
            create_info,
            physical_device,
            command_buffer_resources: Mutex::new(Vec::new()),
            debug_utils,
            debug_messenger,
            timestamp_valid_bits,
            device,
            memory_allocator: Some(memory_allocator),
            queue: Mutex::new(queue),
            queue_family_index,
        })
    }

    /// Returns reset command resources: a cached set when one is idle, otherwise a new set.
    pub fn acquire_command_buffer_resources(&self) -> Result<VkCommandBufferResources, Error> {
        let Some(resources) = self.command_buffer_resources.lock().pop() else {
            return VkCommandBufferResources::new(&self.device, self.queue_family_index);
        };
        if let Err(error) = resources.reset(&self.device) {
            unsafe { resources.destroy(&self.device) };
            return Err(error);
        }
        Ok(resources)
    }

    /// # Safety
    /// The resources must come from this context and must not be pending on the GPU.
    pub unsafe fn recycle_command_buffer_resources(
        &self,
        resources: VkCommandBufferResources,
    ) {
        self.command_buffer_resources.lock().push(resources);
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn memory_allocator(&self) -> &vk_mem::Allocator {
        self.memory_allocator.as_ref().unwrap()
    }

    /// Hold the guard only for queue submission: Vulkan requires external queue synchronization.
    pub fn queue(&self) -> MutexGuard<'_, vk::Queue> {
        self.queue.lock()
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn timestamp_valid_bits(&self) -> u32 {
        self.timestamp_valid_bits
    }

    pub fn physical_device(&self) -> &VkPhysicalDevice {
        &self.physical_device
    }

    pub fn logger(&self) -> &dyn VkLogger {
        self.create_info.logger.as_ref()
    }
}
impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            for resources in self.command_buffer_resources.get_mut().drain(..) {
                resources.destroy(&self.device);
            }
            self.memory_allocator = None;
            self.device.destroy_device(None);
            if let Some(messenger) = self.debug_messenger {
                self.debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
fn get_entry() -> Result<ash::Entry, VkContextError> {
    #[cfg(target_os = "macos")]
    // default loader tries to load lib from /usr/lib/, but on macOS this folder is protected by SIP
    let entry_result = unsafe { ash::Entry::load_from("/usr/local/lib/libvulkan.dylib") };

    #[cfg(not(target_os = "macos"))]
    let entry_result = unsafe { ash::Entry::load() };

    match entry_result {
        Ok(entry) => Ok(entry),
        Err(err) => Err(VkContextError::EntryLoadingError(err)),
    }
}
fn create_instance(
    entry: &ash::Entry,
    api_version: u32,
    create_info: &VkContextCreateInfo,
) -> Result<ash::Instance, VkContextError> {
    let mut instance_extensions: Vec<*const c_char> = Vec::new();
    instance_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
    instance_extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());

    let mut instance_layers: Vec<*const c_char> = Vec::new();
    let mut instance_nexts = Vec::new();

    if create_info.with_validation {
        if !is_layer_supported(entry, VK_LAYER_KHRONOS_VALIDATION)? {
            return Err(VkContextError::ValidationNotSupported);
        }
        instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        instance_layers.push(VK_LAYER_KHRONOS_VALIDATION.as_ptr());

        let msg_create_info = debug_messenger_info(create_info);
        instance_nexts.push(msg_create_info);
    }

    let app_info = vk::ApplicationInfo::default().api_version(api_version);

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
        Err(result) => return Err(VkContextError::InstanceCreate(result)),
    };

    Ok(instance)
}
fn create_memory_allocator(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> Result<vk_mem::Allocator, VkContextError> {
    let mut info = vk_mem::AllocatorCreateInfo::new(instance, device, physical_device);
    info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
    info.vulkan_api_version = vk::API_VERSION_1_3;
    match unsafe { vk_mem::Allocator::new(info) } {
        Ok(allocator) => Ok(allocator),
        Err(err) => Err(VkContextError::MemoryAllocatorCreate(err)),
    }
}
fn get_physical_device(
    instance: &ash::Instance,
    required_extensions: &[&str],
    required_features: &VkPhysicalDeviceFeatures,
) -> Result<VkPhysicalDevice, VkContextError> {
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(result) => return Err(VkContextError::PhysicalDevicesNotFound(result)),
    };

    let device_opt = devices
        .into_iter()
        .map(|device| VkPhysicalDevice::new(instance, device))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|physical_device| {
            physical_device.properties.api_version >= vk::API_VERSION_1_3
                && physical_device.features.contains(required_features)
                && physical_device
                    .subgroup_properties
                    .supported_operations
                    .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                && physical_device.subgroup_properties.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && physical_device
                    .memory_properties
                    .memory_types_as_slice()
                    .iter()
                    .any(|memory_type| memory_type.property_flags.contains(HOST_COHERENT_MEMORY))
                && required_extensions
                    .iter()
                    .all(|&req_ext| physical_device.supported_extensions.contains(&req_ext.to_string()))
        })
        .min_by_key(|physical_device| match physical_device.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            vk::PhysicalDeviceType::OTHER => 4,
            _ => 5,
        });
    device_opt.ok_or(VkContextError::PhysicalDeviceSuitableNotFound)
}
fn get_logical_device(
    instance: &ash::Instance,
    physical_device: &VkPhysicalDevice,
    required_extensions: &[&str],
    required_features: &VkPhysicalDeviceFeatures,
) -> Result<(ash::Device, u32, u32), VkContextError> {
    // find queue family index
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device.device) };
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
    let queue_create_info =
        vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(&queue_priorities);

    // prepare extensions
    let extension_names =
        required_extensions.iter().map(|name| std::ffi::CString::new(*name).unwrap()).collect::<Vec<_>>();
    let mut device_extensions = extension_names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();

    // (https://vulkan.lunarg.com/doc/view/1.4.321.0/mac/antora/spec/latest/chapters/devsandqueues.html#VUID-VkDeviceCreateInfo-pProperties-04451
    if physical_device.supported_extensions.contains(&khr::portability_subset::NAME.to_str().unwrap().to_string()) {
        device_extensions.push(khr::portability_subset::NAME.as_ptr())
    }

    // prepare features
    let vk10_features = vk::PhysicalDeviceFeatures::default().shader_int16(required_features.shader_int16);
    let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default()
        .storage_buffer16_bit_access(required_features.storage_buffer16_bit_access)
        .storage_push_constant16(required_features.storage_push_constant16);
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
        .shader_float16(required_features.shader_float16)
        .shader_subgroup_extended_types(required_features.shader_subgroup_extended_types)
        .buffer_device_address(required_features.buffer_device_address)
        .host_query_reset(required_features.host_query_reset);
    let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
        .maintenance4(required_features.maintenance4)
        .synchronization2(required_features.synchronization2);

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
        Err(result) => return Err(VkContextError::DeviceCreateError(result)),
    };

    Ok((device, queue_family_index, queue_family_properties[queue_family_index as usize].timestamp_valid_bits))
}
fn is_layer_supported(
    entry: &ash::Entry,
    layer: &CStr,
) -> Result<bool, VkContextError> {
    let layer_properties = unsafe { entry.enumerate_instance_layer_properties() }?;
    Ok(layer_properties.iter().any(|properties| unsafe { CStr::from_ptr(properties.layer_name.as_ptr()) } == layer))
}

fn debug_messenger_info(create_info: &VkContextCreateInfo) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_message_callback))
        .user_data(create_info as *const VkContextCreateInfo as *mut c_void)
}

unsafe extern "system" fn debug_message_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    if p_user_data.is_null() {
        return vk::FALSE;
    }

    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let log_message = format!("{types}{:?}", message);

    let create_info = unsafe { &*(p_user_data as *const VkContextCreateInfo) };
    let logger = create_info.logger.as_ref();
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => logger.v(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => logger.i(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => logger.w(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => logger.e(log_message.as_str()),
        _ => logger.d(log_message.as_str()),
    }

    vk::FALSE
}
