use uzu_engine_macros::uzu_config;

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
}
