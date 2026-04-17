use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Specialization {
    Chat,
    Classification,
    TextToSpeech,
    Translation,
    Speculation,
}
