use proc_macros::uzu_config;

use crate::config::tts::{
    audio_decoder::AnyTTSAudioDecoderConfig, text_decoder::AnyTTSTextDecoderConfig, vocoder::AnyVocoderConfig,
};

pub mod audio_decoder;
pub mod text_decoder;
pub mod vocoder;

#[uzu_config]
pub struct TTSConfig {
    pub text_decoder_config: AnyTTSTextDecoderConfig,
    pub audio_decoder_config: AnyTTSAudioDecoderConfig,
    pub vocoder_config: AnyVocoderConfig,
}
