use std::ffi::CStr;

use ash::vk;
use parking_lot::Mutex;

/// The receiver of validation warnings and errors: prints each one to stderr and keeps the first error until it is taken.
#[derive(Default)]
pub struct VkDebugMessages {
    first_error: Mutex<Option<String>>,
}

impl VkDebugMessages {
    pub fn receive(
        &self,
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        message: &CStr,
    ) {
        let types = match message_type {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
            _ => "",
        };
        let message = format!("{types}{message:?}");
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            eprintln!("[Error]: {message}");
            self.first_error.lock().get_or_insert(message);
        } else {
            eprintln!("[Warning]: {message}");
        }
    }

    /// The first error reported since the previous call.
    pub fn take_error(&self) -> Option<String> {
        self.first_error.lock().take()
    }
}
