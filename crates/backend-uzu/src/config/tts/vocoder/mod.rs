use proc_macros::uzu_config_abstract;

pub mod noop_vocoder;

#[uzu_config_abstract(noop_vocoder::NoopVocoderConfig)]
pub struct VocoderConfig;
