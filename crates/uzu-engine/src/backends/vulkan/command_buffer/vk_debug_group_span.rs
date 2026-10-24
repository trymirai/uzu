use std::{
    ffi::CString,
    ops::{Deref, DerefMut},
};

use ash::vk;

use crate::backends::{common::CommandBufferEncoding, vulkan::VkCommandBufferEncoding};

pub struct VkDebugGroupSpan<'a> {
    encoding: &'a mut VkCommandBufferEncoding,
}

impl<'a> VkDebugGroupSpan<'a> {
    pub fn new(
        encoding: &'a mut VkCommandBufferEncoding,
        name: &str,
    ) -> Self {
        if let Some(debug_utils) = encoding.context().debug_utils()
            && encoding.is_recording()
        {
            let name = CString::new(name).expect("debug group name contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(encoding.command_buffer(), &label) };
        }
        encoding.start_timestamp_span(name);
        Self {
            encoding,
        }
    }
}

impl Deref for VkDebugGroupSpan<'_> {
    type Target = VkCommandBufferEncoding;

    fn deref(&self) -> &Self::Target {
        self.encoding
    }
}

impl DerefMut for VkDebugGroupSpan<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoding
    }
}

impl Drop for VkDebugGroupSpan<'_> {
    fn drop(&mut self) {
        self.encoding.end_timestamp_span();
        if let Some(debug_utils) = self.encoding.context().debug_utils()
            && self.encoding.is_recording()
        {
            unsafe { debug_utils.cmd_end_debug_utils_label(self.encoding.command_buffer()) };
        }
    }
}
