#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Swap {
    pub source: u32,
    pub destination: u32,
}
