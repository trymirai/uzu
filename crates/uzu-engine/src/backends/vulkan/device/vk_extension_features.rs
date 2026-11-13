use std::{
    ffi::{CStr, c_void},
    mem::offset_of,
    ptr,
};

use ash::vk;

#[repr(C)]
pub struct VkShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    pub shader_bfloat16_cooperative_matrix: vk::Bool32,
}

const _: () = {
    assert!(offset_of!(VkShaderBfloat16Features, s_type) == offset_of!(vk::BaseOutStructure, s_type));
    assert!(offset_of!(VkShaderBfloat16Features, p_next) == offset_of!(vk::BaseOutStructure, p_next));
    assert!(offset_of!(VkShaderBfloat16Features, shader_bfloat16_type) == size_of::<vk::BaseOutStructure>());
};

impl VkShaderBfloat16Features {
    pub const EXTENSION_NAME: &CStr = c"VK_KHR_shader_bfloat16";
    pub const COMPONENT_TYPE: vk::ComponentTypeKHR = vk::ComponentTypeKHR::from_raw(1_000_141_000);
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

unsafe impl vk::TaggedStructure for VkShaderBfloat16Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_141_000);
}

unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for VkShaderBfloat16Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for VkShaderBfloat16Features {}

#[repr(C)]
pub struct VkCooperativeMatrix2Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    cooperative_matrix_workgroup_scope: vk::Bool32,
    cooperative_matrix_flexible_dimensions: vk::Bool32,
    cooperative_matrix_reductions: vk::Bool32,
    cooperative_matrix_conversions: vk::Bool32,
    pub cooperative_matrix_per_element_operations: vk::Bool32,
    cooperative_matrix_tensor_addressing: vk::Bool32,
    cooperative_matrix_block_loads: vk::Bool32,
}

const _: () = {
    assert!(offset_of!(VkCooperativeMatrix2Features, s_type) == offset_of!(vk::BaseOutStructure, s_type));
    assert!(offset_of!(VkCooperativeMatrix2Features, p_next) == offset_of!(vk::BaseOutStructure, p_next));
    assert!(
        offset_of!(VkCooperativeMatrix2Features, cooperative_matrix_workgroup_scope)
            == size_of::<vk::BaseOutStructure>()
    );
    assert!(
        offset_of!(VkCooperativeMatrix2Features, cooperative_matrix_block_loads)
            == size_of::<vk::BaseOutStructure>() + 6 * size_of::<vk::Bool32>()
    );
};

impl VkCooperativeMatrix2Features {
    pub const EXTENSION_NAME: &CStr = c"VK_NV_cooperative_matrix2";
}

impl Default for VkCooperativeMatrix2Features {
    fn default() -> Self {
        Self {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: ptr::null_mut(),
            cooperative_matrix_workgroup_scope: vk::FALSE,
            cooperative_matrix_flexible_dimensions: vk::FALSE,
            cooperative_matrix_reductions: vk::FALSE,
            cooperative_matrix_conversions: vk::FALSE,
            cooperative_matrix_per_element_operations: vk::FALSE,
            cooperative_matrix_tensor_addressing: vk::FALSE,
            cooperative_matrix_block_loads: vk::FALSE,
        }
    }
}

unsafe impl vk::TaggedStructure for VkCooperativeMatrix2Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_000);
}

unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for VkCooperativeMatrix2Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for VkCooperativeMatrix2Features {}
