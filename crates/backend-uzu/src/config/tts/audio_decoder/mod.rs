use proc_macros::uzu_config_abstract;

pub mod common;
pub mod descript_audio_codec;
pub mod fish_audio_modules;

#[uzu_config_abstract(descript_audio_codec::DescriptAudioCodecConfig)]
pub struct TTSAudioDecoderConfig;
