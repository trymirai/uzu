use std::ffi::{CStr, c_char, c_void};

use ash::{ext, vk};
use parking_lot::Mutex;

use crate::backends::vulkan::{API_VERSION_1_4, VkContextError};

const VK_LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct VkInstance {
    entry: ash::Entry,
    handle: ash::Instance,
    debug_utils: ext::debug_utils::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    first_validation_error: Box<Mutex<Option<String>>>,
    debug_labels: bool,
}

impl VkInstance {
    pub fn new(validation: bool) -> Result<Self, VkContextError> {
        let entry = unsafe { ash::Entry::load() }.map_err(VkContextError::EntryLoad)?;
        let first_validation_error = Box::new(Mutex::new(None));
        let messages = &raw const *first_validation_error as *mut c_void;

        let mut layers: Vec<*const c_char> = Vec::new();
        if validation {
            let available =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(VkContextError::InstanceEnumeration)?;
            if !available
                .iter()
                .any(|layer| layer.layer_name_as_c_str().is_ok_and(|name| name == VK_LAYER_KHRONOS_VALIDATION))
            {
                return Err(VkContextError::ValidationNotSupported);
            }
            layers.push(VK_LAYER_KHRONOS_VALIDATION.as_ptr());
        }
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(VkContextError::InstanceEnumeration)?;
        let debug_labels = validation
            || available.iter().any(|extension| extension.extension_name_as_c_str() == Ok(vk::EXT_DEBUG_UTILS_NAME));
        let extensions: Vec<*const c_char> =
            debug_labels.then(|| vk::EXT_DEBUG_UTILS_NAME.as_ptr()).into_iter().collect();

        let app_info = vk::ApplicationInfo::default().api_version(API_VERSION_1_4);
        let mut messenger_info = debug_messenger_info(messages);
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        if validation {
            instance_info = instance_info.push_next(&mut messenger_info);
        }
        let handle = unsafe { entry.create_instance(&instance_info, None) }.map_err(VkContextError::InstanceCreate)?;

        let mut instance = Self {
            debug_utils: ext::debug_utils::Instance::new(&entry, &handle),
            entry,
            handle,
            debug_messenger: None,
            first_validation_error,
            debug_labels,
        };
        if validation {
            let messenger =
                unsafe { instance.debug_utils.create_debug_utils_messenger(&debug_messenger_info(messages), None) }
                    .map_err(VkContextError::DebugMessengerCreate)?;
            instance.debug_messenger = Some(messenger);
        }
        Ok(instance)
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn handle(&self) -> &ash::Instance {
        &self.handle
    }

    pub fn debug_labels(&self) -> bool {
        self.debug_labels
    }

    pub fn take_validation_error(&self) -> Option<String> {
        self.first_validation_error.lock().take()
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some(messenger) = self.debug_messenger {
                self.debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.handle.destroy_instance(None);
        }
    }
}

fn debug_messenger_info(messages: *mut c_void) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_message_callback))
        .user_data(messages)
}

unsafe extern "system" fn debug_message_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    let first_error = unsafe { &*(p_user_data as *const Mutex<Option<String>>) };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    first_error.lock().get_or_insert_with(|| format!("{message_type:?}: {}", message.to_string_lossy()));
    vk::FALSE
}
