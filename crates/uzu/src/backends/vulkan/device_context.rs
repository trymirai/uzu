use std::sync::Arc;
use ash::vk;
use crate::{array_size_in_bytes, DataType, DeviceContext};
use crate::backends::vulkan::array::VkArray;
use crate::backends::vulkan::buffer::{VkBuffer, VkBufferCreateInfo};
use crate::backends::vulkan::context::VkContext;

pub struct VkDeviceContext {
    context: Arc<VkContext>
}

impl VkDeviceContext {
    pub fn new(context: Arc<VkContext>) -> Self {
        Self { context }
    }
}

impl DeviceContext for VkDeviceContext {
    type DeviceArray = VkArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String
    ) -> Self::DeviceArray {
        let size = array_size_in_bytes(&shape, data_type) as vk::DeviceSize;
        let buffer_info = VkBufferCreateInfo::new(size, false, false);
        let mut buffer = VkBuffer::new_with_info(self.context.clone(), &buffer_info)
            .expect("Failed to create VkBuffer");
        buffer.set_name(label.as_str());
        VkArray::new_with_offset_and_label(buffer, shape, data_type, 0usize, label)
    }
}