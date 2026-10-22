use uzu_engine_macros::uzu_config;

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
    /// Code tape version; absent means 0 (legacy). This build reads 2.
    #[serde(default)]
    pub tape: u32,
}

/// Each restart block's bit stream stored reversed, little-endian.
pub const REVERSED_TAPE: u32 = 2;
