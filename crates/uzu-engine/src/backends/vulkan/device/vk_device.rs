use std::{mem::ManuallyDrop, ops::Deref};

use ash::ext;

use crate::backends::vulkan::{API_VERSION_1_4, VkContextError, VkInstance, VkPhysicalDevice};

// The logical device and what it is created from. Everything made from the device holds an `Arc` of this.
pub struct VkDevice {
    handle: ash::Device,
    pub memory_allocator: ManuallyDrop<vk_mem::Allocator>,
    pub debug_labels: Option<ext::debug_utils::Device>,
    pub physical_device: VkPhysicalDevice,
    pub instance: VkInstance,
}

impl VkDevice {
    pub fn new(validation: bool) -> Result<Self, VkContextError> {
        let instance = VkInstance::new(validation)?;
        let physical_device = VkPhysicalDevice::select(instance.entry(), instance.handle())?;
        let handle = physical_device.create_device(instance.handle())?;
        let debug_labels = instance.debug_labels().then(|| ext::debug_utils::Device::new(instance.handle(), &handle));
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(instance.handle(), &handle, physical_device.handle);
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        allocator_info.vulkan_api_version = API_VERSION_1_4;
        let memory_allocator = unsafe { vk_mem::Allocator::new(allocator_info) }
            .inspect_err(|_| unsafe { handle.destroy_device(None) })
            .map_err(VkContextError::MemoryAllocatorCreate)?;
        Ok(Self {
            handle,
            memory_allocator: ManuallyDrop::new(memory_allocator),
            debug_labels,
            physical_device,
            instance,
        })
    }
}

impl Deref for VkDevice {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.handle
    }
}

impl Drop for VkDevice {
    fn drop(&mut self) {
        // The allocator frees its memory through the device, and the instance field drops after this body.
        unsafe {
            ManuallyDrop::drop(&mut self.memory_allocator);
            self.handle.destroy_device(None);
        }
    }
}
