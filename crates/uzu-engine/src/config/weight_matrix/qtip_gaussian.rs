use uzu_engine_macros::uzu_config;

use super::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
    pub layout: Layout,
    pub tape: u32,
}

pub const CODEBOOK_V2: [f32; 5] = [
    f32::from_bits(0x3d5583d1),
    f32::from_bits(0xbda85975),
    f32::from_bits(0xbd9f2d3c),
    f32::from_bits(0xbda85975),
    f32::from_bits(0xbd9f2d3c),
];

pub const CODEBOOK_V4: [f32; 5] = [
    f32::from_bits(0x3d552540),
    f32::from_bits(0xbda80ef5),
    f32::from_bits(0xbd9ee6bd),
    f32::from_bits(0xbda009bd),
    f32::from_bits(0xbda61735),
];
