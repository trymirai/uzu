use enumset::EnumSetType;
use serde::{Deserialize, Serialize};

/// Full or part of content
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    // Audio, Image, etc
}

impl ContentPart {
    fn kind(&self) -> ContentKind {
        match self {
            ContentPart::Text {
                ..
            } => ContentKind::Text,
        }
    }
}

#[derive(EnumSetType)]
pub enum ContentKind {
    Text,
}
