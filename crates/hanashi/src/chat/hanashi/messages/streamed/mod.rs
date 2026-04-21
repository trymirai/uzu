mod content;
mod section;

use std::{collections::HashMap, str::FromStr};

pub use content::Content;
pub use section::Section;
use serde::{Deserialize, Deserializer};
use shoji::types::session::chat::{Message as CommonMessage, Role};

#[derive(Deserialize)]
pub struct Message {
    #[serde(deserialize_with = "deserialize_trimmed_role")]
    pub role: Role,
    pub content: Option<Content>,
}

fn deserialize_trimmed_role<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Role, D::Error> {
    let name = String::deserialize(deserializer)?;
    Role::from_str(name.trim()).map_err(serde::de::Error::custom)
}

impl From<Message> for CommonMessage {
    fn from(message: Message) -> Self {
        CommonMessage {
            content: match message.content {
                Some(content) => content.blocks(&message.role),
                None => Vec::new(),
            },
            role: message.role,
            metadata: HashMap::new(),
        }
    }
}
