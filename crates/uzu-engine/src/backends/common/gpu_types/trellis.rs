use uzu_engine_macros::uzu_config;

pub const COLUMN_GROUP_COUNT: u32 = 4;
pub const MAX_DIMENSION: u32 = 20480;
pub const MIN_HADAMARD_SIZE: u32 = 512;
pub const MAX_HADAMARD_SIZE: u32 = 4096;

#[repr(C)]
#[derive(Copy)]
#[uzu_config]
pub struct TrellisCodebook {
    pub scale: f32,
    pub offsets: [f32; 4],
}
