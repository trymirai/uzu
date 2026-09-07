#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcVersion {
    major: u8,
    minor: u8,
    build: u8,
    reserved: u8,
    release: u16,
}
