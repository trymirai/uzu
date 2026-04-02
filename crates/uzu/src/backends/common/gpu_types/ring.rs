#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RingParams {
    pub ring_offset: u32,
    pub ring_length: u32,
}
