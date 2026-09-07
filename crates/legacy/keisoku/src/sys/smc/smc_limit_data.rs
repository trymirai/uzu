#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcLimitData {
    version: u16,
    length: u16,
    cpu_plimit: u32,
    gpu_plimit: u32,
    mem_plimit: u32,
}
