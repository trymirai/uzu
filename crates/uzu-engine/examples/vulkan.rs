use std::sync::Arc;

use uzu_engine::backends::vulkan::{
    Error, VkBuffer, VkBufferCreateInfo, VkBufferError, VkContext, VkContextCreateInfo,
};

fn main() -> Result<(), Error> {
    let context = Arc::new(VkContext::new(VkContextCreateInfo::default())?);
    let name = unsafe { std::ffi::CStr::from_ptr(context.physical_device().properties.device_name.as_ptr()) };
    println!("Vulkan context created: {}", name.to_string_lossy());
    let data = (0..4096).map(|index| (index % 100) as u8).collect::<Box<[u8]>>();
    let info = VkBufferCreateInfo::new(data.len() as u64, true, true);
    let mut buffer = VkBuffer::new_with_info(context, &info)?;
    buffer.fill(&data)?;
    assert_eq!(buffer.get_bytes()?.as_ref(), data.as_ref());
    assert!(matches!(buffer.fill(&[0; 4097]), Err(VkBufferError::SizeOutOfBounds { .. })));
    println!("Vulkan buffer round-trip verified: {} bytes", data.len());
    Ok(())
}
