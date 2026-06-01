use proc_macros::uzu_config_abstract;

pub mod attention;
pub mod convolutions;
pub mod delta_net;
pub mod mamba2;
pub mod short_conv;

#[uzu_config_abstract(
    attention::AttentionConfig,
    delta_net::DeltaNetConfig,
    mamba2::Mamba2Config,
    short_conv::ShortConvConfig
)]
pub struct TokenMixerConfig;
