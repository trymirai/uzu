use serde::{Deserialize, Serialize};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranslationPayload {
    Text {
        text: String,
    },
    Image {
        url: String,
    },
}
