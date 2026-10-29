use shoji::types::session::classification::{ChatTokenCodecConfig, TokenCodecConfig as ShojiTokenCodecConfig};
use uzu_engine_macros::uzu_config_abstract;

pub mod chat_codec;
pub mod raw_text_codec;

#[uzu_config_abstract(chat_codec::ChatCodecConfig, raw_text_codec::RawTextCodecConfig)]
pub struct TokenCodecConfig;

impl From<&AnyTokenCodecConfig> for ShojiTokenCodecConfig {
    fn from(config: &AnyTokenCodecConfig) -> Self {
        match config {
            AnyTokenCodecConfig::ChatCodecConfig(config) => ShojiTokenCodecConfig::Chat(ChatTokenCodecConfig {
                prompt_template: config.prompt_template.clone(),
                output_parser_regex: config.output_parser_regex.clone(),
                system_role_name: config.system_role_name.clone(),
                user_role_name: config.user_role_name.clone(),
                assistant_role_name: config.assistant_role_name.clone(),
                eos_token: config.eos_token.clone(),
                bos_token: config.bos_token.clone(),
                end_of_thinking_tag: config.end_of_thinking_tag.clone(),
                default_system_prompt: config.default_system_prompt.clone(),
            }),
            AnyTokenCodecConfig::RawTextCodecConfig(_) => ShojiTokenCodecConfig::RawText,
        }
    }
}
