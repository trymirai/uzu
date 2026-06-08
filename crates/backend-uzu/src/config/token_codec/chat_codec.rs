use proc_macros::uzu_config;

#[uzu_config(super::TokenCodecConfig)]
pub struct ChatCodecConfig {
    pub prompt_template: String,
    pub output_parser_regex: Option<String>,
    pub system_role_name: String,
    pub user_role_name: String,
    pub assistant_role_name: String,
    pub eos_token: Option<String>,
    pub bos_token: Option<String>,
    pub end_of_thinking_tag: Option<String>,
    pub default_system_prompt: Option<String>,
}
