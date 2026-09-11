use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenCodecConfig {
    Chat(ChatTokenCodecConfig),
    RawText,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTokenCodecConfig {
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
