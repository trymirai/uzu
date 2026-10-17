use bitflags::bitflags;

bitflags! {
    /// The device features the backend requires.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VkPhysicalDeviceFeatures: u32 {
        // Version 1.0
        const SHADER_INT16 = 1 << 0;

        // Version 1.1
        const STORAGE_BUFFER_16BIT_ACCESS = 1 << 1;
        const STORAGE_PUSH_CONSTANT16 = 1 << 2;

        // Version 1.2
        const SHADER_INT8 = 1 << 3;
        const STORAGE_BUFFER_8BIT_ACCESS = 1 << 4;
        /// Slang declares StoragePushConstant8 for modules with byte pointers, as StoragePushConstant16 for 16-bit.
        const STORAGE_PUSH_CONSTANT8 = 1 << 5;
        const SHADER_FLOAT16 = 1 << 6;
        const SHADER_SUBGROUP_EXTENDED_TYPES = 1 << 7;
        const BUFFER_DEVICE_ADDRESS = 1 << 8;
        const HOST_QUERY_RESET = 1 << 9;
        /// Slang emits every SPIR-V 1.6 module with the Vulkan memory model.
        const VULKAN_MEMORY_MODEL = 1 << 10;
        const VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 1 << 11;

        // Version 1.3
        const MAINTENANCE4 = 1 << 12;
        const SYNCHRONIZATION2 = 1 << 13;
    }
}
