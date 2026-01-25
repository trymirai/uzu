use std::sync::Arc;
use ash::vk;
use uzu::backends::vulkan::buffer::{VkBuffer, VkBufferCreateInfo};
use uzu::backends::vulkan::context::VkContext;
use uzu::backends::vulkan::context::VkContextCreateInfo;

fn main() {
    let ctx = Arc::new(
        VkContext::new(VkContextCreateInfo::default())
            .expect("Failed to create Vulkan context"),
    );
    println!("Vulkan context created");

    let mut buffer_data = vec![0u8; 4096];
    for i in 0..buffer_data.len() {
        buffer_data.push((i % 100) as u8);
    }

    let buffer_info = VkBufferCreateInfo::new(
        buffer_data.len() as vk::DeviceSize,
        false,
        true
    );
    let mut buffer = VkBuffer::new_with_info(ctx, &buffer_info)
        .expect("Failed to create VkBuffer");
    buffer.fill(&buffer_data)
        .expect("Failed to fill buffer");
    println!("Vulkan buffer created and filled");
}