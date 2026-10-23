use bitflags::bitflags;

bitflags! {
    /// The optional device features the backend requires; Vulkan 1.4 makes every other feature it enables mandatory.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VkPhysicalDeviceFeatures: u32 {
        /// Slang declares StoragePushConstant8 for modules with byte pointers, as StoragePushConstant16 for 16-bit.
        const STORAGE_PUSH_CONSTANT8 = 1 << 0;
        const STORAGE_PUSH_CONSTANT16 = 1 << 1;
        const SHADER_FLOAT16 = 1 << 2;
    }
}
