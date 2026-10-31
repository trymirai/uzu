use std::ops::{Deref, DerefMut};

use crate::backends::vulkan::VkCommandBufferEncoding;

pub struct VkDebugGroupSpan<'a> {
    encoding: &'a mut VkCommandBufferEncoding,
}

impl<'a> VkDebugGroupSpan<'a> {
    pub fn new(
        encoding: &'a mut VkCommandBufferEncoding,
        name: &str,
    ) -> Self {
        encoding.begin_debug_label(name);
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
        self.encoding.end_debug_label();
    }
}
