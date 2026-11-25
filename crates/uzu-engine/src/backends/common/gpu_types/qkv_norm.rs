use bytemuck::NoUninit;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QKVL2Op {
    None,
    RmsNorm,
    Clip,
}

#[derive(Clone, Copy, Debug, Default, NoUninit)]
#[repr(C)]
pub struct QKVNormParams {
    pub op: [u32; 3],
    pub full_layer: [u32; 3],
    pub epsilon: [f32; 3],
    pub scale_offset: [f32; 3],
    pub clip_threshold: [f32; 3],
}
