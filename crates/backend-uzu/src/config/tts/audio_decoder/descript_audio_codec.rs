use proc_macros::uzu_config;

use crate::config::tts::audio_decoder::fish_audio_modules::{DACDecoderConfig, DownsampleResidualVectorQuantizeConfig};

#[uzu_config(super::TTSAudioDecoderConfig)]
pub struct DescriptAudioCodecConfig {
    pub quantizer_config: DownsampleResidualVectorQuantizeConfig,
    pub decoder_config: DACDecoderConfig,
    pub samplerate: u32,

    pub encoder_dim: usize,
    pub encoder_rates: Box<[usize]>,
    pub decoder_dim: usize,
    pub decoder_rates: Box<[usize]>,
    pub input_dim: usize,
    pub n_codebooks: usize,
    pub codebook_dim: usize,
    pub downsample_factor: Box<[usize]>,
    pub codebook_size: usize,
    pub semantic_codebook_size: usize,
}
