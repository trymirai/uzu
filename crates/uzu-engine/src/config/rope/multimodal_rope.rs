use uzu_engine_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct MultimodalRoPEConfig {
    pub frequency_pairs_per_axis: [u32; 3],
}
