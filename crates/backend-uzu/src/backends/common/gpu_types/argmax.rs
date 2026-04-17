#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ArgmaxPair {
    pub value: f32,
    pub index: u32,
}
