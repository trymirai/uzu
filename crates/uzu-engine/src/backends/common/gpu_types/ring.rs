use bytemuck::NoUninit;

#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct RingParams {
    pub ring_offset: u32,
    pub ring_length: u32,
}
