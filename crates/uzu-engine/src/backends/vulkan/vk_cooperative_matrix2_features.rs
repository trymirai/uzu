use std::{
    ffi::{CStr, c_void},
    mem::offset_of,
    ptr,
};

use ash::vk;

/// VK_NV_cooperative_matrix2's feature structure, absent from Ash 0.38's registry.
#[repr(C)]
pub struct VkCooperativeMatrix2Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub cooperative_matrix_workgroup_scope: vk::Bool32,
    pub cooperative_matrix_flexible_dimensions: vk::Bool32,
    pub cooperative_matrix_reductions: vk::Bool32,
    pub cooperative_matrix_conversions: vk::Bool32,
    pub cooperative_matrix_per_element_operations: vk::Bool32,
    pub cooperative_matrix_tensor_addressing: vk::Bool32,
    pub cooperative_matrix_block_loads: vk::Bool32,
}

const _: () = {
    assert!(offset_of!(VkCooperativeMatrix2Features, s_type) == offset_of!(vk::BaseOutStructure, s_type));
    assert!(offset_of!(VkCooperativeMatrix2Features, p_next) == offset_of!(vk::BaseOutStructure, p_next));
    assert!(
        offset_of!(VkCooperativeMatrix2Features, cooperative_matrix_workgroup_scope)
            == size_of::<vk::BaseOutStructure>()
    );
    assert!(
        size_of::<VkCooperativeMatrix2Features>()
            == size_of::<vk::BaseOutStructure>() + 7 * size_of::<vk::Bool32>() + 4
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

// SAFETY: repr(C) with the guarded chain header, fields in vulkan_core.h's order, and the registered structure tag.
unsafe impl vk::TaggedStructure for VkCooperativeMatrix2Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_000);
}

// SAFETY: VK_NV_cooperative_matrix2 permits this structure in both chains.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for VkCooperativeMatrix2Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for VkCooperativeMatrix2Features {}
