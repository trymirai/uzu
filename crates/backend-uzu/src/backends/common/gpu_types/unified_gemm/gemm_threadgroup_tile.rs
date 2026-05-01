use debug_display::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmThreadgroupTile {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
