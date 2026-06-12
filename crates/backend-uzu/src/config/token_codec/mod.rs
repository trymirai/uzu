use proc_macros::uzu_config_abstract;

pub mod chat_codec;
pub mod raw_text_codec;
pub mod tts_codec;

#[uzu_config_abstract(chat_codec::ChatCodecConfig, tts_codec::TTSCodecConfig, raw_text_codec::RawTextCodecConfig)]
pub struct TokenCodecConfig;
