use proc_macros::uzu_config_abstract;

pub mod chat_codec;
pub mod tts_codec;

#[uzu_config_abstract(chat_codec::ChatCodecConfig, tts_codec::TTSCodecConfig)]
pub struct TokenCodecConfig;
