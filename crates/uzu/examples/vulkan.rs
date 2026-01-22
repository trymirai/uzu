use uzu::backends::vulkan::context::VkContext;
use uzu::backends::vulkan::context::VkContextCreateInfo;

fn main() {
    let _ctx = VkContext::new(VkContextCreateInfo::default()).unwrap();
    println!("Vulkan context created");
}