use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ChatTranslationInput")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranslationInput {
    Text {
        text: String,
    },
    Image {
        url: String,
    },
}
