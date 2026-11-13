use bytemuck::NoUninit;

#[derive(Clone, Copy, Debug, Default, NoUninit)]
#[repr(C)]
pub struct QKVNormParams {
    pub enabled: [u32; 3],
    pub full_layer: [u32; 3],
    pub epsilon: [f32; 3],
    pub scale_offset: [f32; 3],
}
