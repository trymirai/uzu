use uzu_engine_macros::uzu_config_abstract;

pub mod attention;
pub mod convolutions;
pub mod delta_net;
pub mod short_conv;

#[uzu_config_abstract(attention::AttentionConfig, delta_net::DeltaNetConfig, short_conv::ShortConvConfig)]
pub struct TokenMixerConfig;
