use std::{
    ffi::{CStr, c_void},
    mem::offset_of,
    ptr,
};

use ash::vk;

/// VK_KHR_shader_bfloat16's feature structure, absent from Ash 0.38's registry.
#[repr(C)]
pub struct VkShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub shader_bfloat16_type: vk::Bool32,
    pub shader_bfloat16_dot_product: vk::Bool32,
    pub shader_bfloat16_cooperative_matrix: vk::Bool32,
}

const _: () = {
    assert!(offset_of!(VkShaderBfloat16Features, s_type) == offset_of!(vk::BaseOutStructure, s_type));
    assert!(offset_of!(VkShaderBfloat16Features, p_next) == offset_of!(vk::BaseOutStructure, p_next));
    assert!(offset_of!(VkShaderBfloat16Features, shader_bfloat16_type) == size_of::<vk::BaseOutStructure>());
};

impl VkShaderBfloat16Features {
    pub const EXTENSION_NAME: &CStr = c"VK_KHR_shader_bfloat16";
}

impl Default for VkShaderBfloat16Features {
    fn default() -> Self {
        Self {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        }
    }
}

// SAFETY: repr(C), the guarded chain header and Vulkan's registered structure tag match the extension ABI.
unsafe impl vk::TaggedStructure for VkShaderBfloat16Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_141_000);
}

// SAFETY: VK_KHR_shader_bfloat16 permits this structure in both chains.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for VkShaderBfloat16Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for VkShaderBfloat16Features {}
