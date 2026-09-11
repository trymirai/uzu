mod content;
mod section;

use std::{collections::HashMap, str::FromStr};

pub use content::Content;
pub use section::Section;
use serde::{Deserialize, Deserializer};
use shoji::types::session::chat::{ChatMessage as CommonMessage, ChatMessageMetadata, ChatRole};

#[derive(Deserialize)]
pub struct Message {
    #[serde(deserialize_with = "deserialize_trimmed_role")]
    pub role: ChatRole,
    pub content: Option<Content>,
}

fn deserialize_trimmed_role<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ChatRole, D::Error> {
    let name = String::deserialize(deserializer)?;
    ChatRole::from_str(name.trim()).map_err(serde::de::Error::custom)
}

impl From<Message> for CommonMessage {
    fn from(message: Message) -> Self {
        CommonMessage {
            content: match message.content {
                Some(content) => content.blocks(&message.role),
                None => Vec::new(),
            },
            role: message.role,
            metadata: ChatMessageMetadata {
                values: HashMap::new(),
            },
        }
    }
}
