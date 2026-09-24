use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct ReasoningConfig {
    pub default_reasoning_effort: String,
    pub field_name: String,
    pub reasoning_effort_to_field_value: serde_json::Map<String, serde_json::Value>,
}

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
    pub reasoning_config: Option<ReasoningConfig>,
}
