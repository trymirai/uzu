use std::{ffi::CStr, sync::Arc};

use bytemuck::{AnyBitPattern, NoUninit};

use super::validation_logger::ValidationLogger;
use crate::backends::vulkan::{
    VkBuffer, VkCommandBufferCompleted, VkCommandBufferEncoding, VkContext, VkContextCreateInfo,
};

/// A validated Vulkan context with the buffer and submission helpers shared by the Vulkan tests.
pub struct KernelFixture {
    pub context: Arc<VkContext>,
    logger: ValidationLogger,
}

impl KernelFixture {
    pub fn new() -> Self {
        let logger = ValidationLogger::default();
        let context = VkContext::new(VkContextCreateInfo {
            with_validation: true,
            logger: Box::new(logger.clone()),
        })
        .expect("Vulkan context");
        let properties = &context.physical_device().properties;
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        eprintln!("Vulkan test device: {} ({:?})", name.to_string_lossy(), properties.device_type);
        Self {
            context: Arc::new(context),
            logger,
        }
    }

    pub fn buffer<T: NoUninit>(
        &self,
        values: &[T],
    ) -> Arc<VkBuffer> {
        let bytes = bytemuck::cast_slice::<T, u8>(values);
        let mut buffer = VkBuffer::new(self.context.clone(), bytes.len() as u64).expect("buffer");
        buffer.fill(bytes).expect("host fill");
        Arc::new(buffer)
    }

    pub fn encoding(&self) -> VkCommandBufferEncoding {
        VkCommandBufferEncoding::new(self.context.clone()).expect("command buffer")
    }

    pub fn complete(encoding: VkCommandBufferEncoding) -> VkCommandBufferCompleted {
        encoding.end_encoding().expect("end encoding").submit().wait_until_completed().expect("completion")
    }

    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn read<T: NoUninit + AnyBitPattern>(buffer: &VkBuffer) -> Vec<T> {
        bytemuck::pod_collect_to_vec(&unsafe { buffer.get_bytes() })
    }

    pub fn assert_clean(&self) {
        self.logger.assert_clean();
    }
}
