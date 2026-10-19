use uzu_engine_macros::uzu_config;

/// Restart tapes (V4) store their codes as little-endian 32-bit words, every other tape as bytes. Determined by the
/// tape; carried so a package from before the word layout fails to load.
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum TrellisCodeWordOrder {
    Bytes,
    LittleEndianWords,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
    pub code_word_order: TrellisCodeWordOrder,
}
