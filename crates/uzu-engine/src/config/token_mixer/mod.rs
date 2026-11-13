use uzu_engine_macros::uzu_config_abstract;

pub mod attention;
pub mod convolutions;
pub mod delta_net;
pub mod short_conv;
pub mod sparse_attention;

#[uzu_config_abstract(
    attention::AttentionConfig,
    delta_net::DeltaNetConfig,
    short_conv::ShortConvConfig,
    sparse_attention::SparseAttentionConfig
)]
pub struct TokenMixerConfig;
