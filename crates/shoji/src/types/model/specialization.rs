use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ModelSpecialization")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Specialization {
    Chat,
    Classification,
    TextToSpeech,
    Translation,
    Speculation,
}
