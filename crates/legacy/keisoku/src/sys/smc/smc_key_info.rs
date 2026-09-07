#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcKeyInfo {
    pub data_size: u32,
    pub data_type: u32,
    pub data_attributes: u8,
}
