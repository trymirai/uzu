use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSpecialization {
    Chat {},
    Classification {},
    TextToSpeech {},
    Translation {},
    Speculation {},
}
