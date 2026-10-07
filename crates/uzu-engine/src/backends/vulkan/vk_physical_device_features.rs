/// Here is only features that required by app
pub struct VkPhysicalDeviceFeatures {
    // Version 1.0
    pub shader_int16: bool,

    // Version 1.1
    pub storage_buffer16_bit_access: bool,
    pub storage_push_constant16: bool,

    // Version 1.2
    pub shader_float16: bool,
    pub shader_subgroup_extended_types: bool,
}
impl VkPhysicalDeviceFeatures {
    pub fn contains(
        &self,
        other: &Self,
    ) -> bool {
        (self.shader_int16 || !other.shader_int16)
            && (self.storage_buffer16_bit_access || !other.storage_buffer16_bit_access)
            && (self.storage_push_constant16 || !other.storage_push_constant16)
            && (self.shader_float16 || !other.shader_float16)
            && (self.shader_subgroup_extended_types || !other.shader_subgroup_extended_types)
    }
}
impl Default for VkPhysicalDeviceFeatures {
    fn default() -> Self {
        Self {
            shader_int16: true,
            storage_buffer16_bit_access: true,
            storage_push_constant16: true,
            shader_float16: true,
            shader_subgroup_extended_types: true,
        }
    }
}
