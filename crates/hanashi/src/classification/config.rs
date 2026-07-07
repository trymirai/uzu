use std::{io, io::ErrorKind};

use serde_json::Value;

pub enum TokenCodecConfig {
    Chat(ChatTokenCodecConfig),
    RawText,
}

impl TokenCodecConfig {
    pub fn from_value(value: Value) -> Result<Self, io::Error> {
        let typ = value.get("type");
        match typ {
            Some(Value::String(codec_type)) if codec_type == "RawTextCodecConfig" => Ok(Self::RawText),
            Some(Value::String(codec_type)) if codec_type == "ChatCodecConfig" => {
                let config = serde_json::from_value::<ChatTokenCodecConfig>(value)?;
                Ok(Self::Chat(config))
            },
            None => {
                let config = serde_json::from_value::<ChatTokenCodecConfig>(value)?;
                Ok(Self::Chat(config))
            },
            _ => Err(io::Error::new(ErrorKind::InvalidData, format!("Invalid config: unknown type \"{typ:?}\""))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
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
