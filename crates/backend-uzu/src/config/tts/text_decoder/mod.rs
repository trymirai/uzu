use proc_macros::uzu_config_abstract;

pub mod fish_audio_text_decoder;

#[uzu_config_abstract(fish_audio_text_decoder::FishAudioTextDecoderConfig)]
pub struct TTSTextDecoderConfig;
