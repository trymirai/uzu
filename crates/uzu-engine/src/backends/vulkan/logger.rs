use std::{ffi::CStr, os::raw::c_void};

use ash::vk;

use super::VkContextCreateInfo;

pub unsafe extern "system" fn debug_message_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    if _p_user_data.is_null() {
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

    let create_info = unsafe { &*(_p_user_data as *const VkContextCreateInfo) };
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
