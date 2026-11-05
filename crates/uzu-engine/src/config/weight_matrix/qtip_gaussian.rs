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

pub const CODEBOOK_V2: [u32; 5] = [0x3d5583d1, 0xbda85975, 0xbd9f2d3c, 0xbda85975, 0xbd9f2d3c];

pub const CODEBOOK_V4: [u32; 5] = [0x3d552540, 0xbda80ef5, 0xbd9ee6bd, 0xbda009bd, 0xbda61735];
