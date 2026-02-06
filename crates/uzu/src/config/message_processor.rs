use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum ToolCallFormat {
    #[serde(rename = "json")]
    Json {
        name_key: String,
        arguments_key: String,
        separator: Option<String>,
    },
    #[serde(rename = "pythonic")]
    Pythonic {
        function_regex: String,
        argument_separator: String,
        string_token: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MessageProcessorConfig {
    pub prompt_template: String,
    pub output_parser_regex: Option<String>,
    pub tool_call_format: Option<ToolCallFormat>,
    pub system_role_name: String,
    pub user_role_name: String,
    pub assistant_role_name: String,
    pub bos_token: Option<String>,
}
