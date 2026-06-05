use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[cfg(metal_backend)]
use crate::config::token_codec::tts_codec::TTSCodecConfig;
use crate::{
    config::token_codec::chat_codec::ChatCodecConfig,
    session::{parameter::ConfigResolvableValue, types::Role},
};

#[cfg(metal_backend)]
pub(crate) const DEFAULT_TTS_SPEAKER_ID: &str = "speaker:0";
#[cfg(metal_backend)]
pub(crate) const DEFAULT_TTS_STYLE: &str = "interleave";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(
        role: Role,
        content: String,
        reasoning_content: Option<String>,
    ) -> Self {
        Self {
            role,
            content,
            reasoning_content,
        }
    }

    pub fn system(content: String) -> Self {
        Self::new(Role::System, content, None)
    }

    pub fn user(content: String) -> Self {
        Self::new(Role::User, content, None)
    }

    pub fn assistant(
        content: String,
        reasoning_content: Option<String>,
    ) -> Self {
        Self::new(Role::Assistant, content, reasoning_content)
    }
}

impl ConfigResolvableValue<ChatCodecConfig, HashMap<String, String>> for Message {
    fn resolve(
        &self,
        config: &ChatCodecConfig,
    ) -> HashMap<String, String> {
        let role = match self.role {
            Role::System => config.system_role_name.clone(),
            Role::User => config.user_role_name.clone(),
            Role::Assistant => config.assistant_role_name.clone(),
        };
        let content = self.content.clone();
        let mut result = HashMap::from([(String::from("role"), role), (String::from("content"), content)]);
        if let Some(reasoning_content) = self.reasoning_content.clone() {
            result.insert(String::from("reasoning_content"), reasoning_content);
        }
        result
    }
}

#[cfg(metal_backend)]
impl ConfigResolvableValue<TTSCodecConfig, HashMap<String, String>> for Message {
    fn resolve(
        &self,
        _config: &TTSCodecConfig,
    ) -> HashMap<String, String> {
        HashMap::from([
            (String::from("content"), self.content.clone()),
            (String::from("speaker_id"), String::from(DEFAULT_TTS_SPEAKER_ID)),
            (String::from("style"), String::from(DEFAULT_TTS_STYLE)),
        ])
    }
}
