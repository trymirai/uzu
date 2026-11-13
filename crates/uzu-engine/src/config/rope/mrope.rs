use uzu_engine_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MRoPELayout {
    /// Qwen2-VL / Qwen2.5-VL: contiguous blocks [T T T ... H H H ... W W W].
    Chunked,
    /// Qwen3-VL / Qwen3.5: round-robin [T H W T H W ... T T].
    Interleaved,
}

#[uzu_config(super::RoPEConfig)]
pub struct MRoPEConfig {
    /// Frequency pairs per position axis (time, height, width); sums to head_dim / 2.
    pub sections: Box<[u32]>,
    pub layout: MRoPELayout,
}
