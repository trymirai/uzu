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

#[bindings::export(Implementation)]
impl ModelSpecialization {
    #[bindings::export(Method(Getter))]
    pub fn name(&self) -> String {
        match self {
            ModelSpecialization::Chat {} => "Chat".to_string(),
            ModelSpecialization::Classification {} => "Classification".to_string(),
            ModelSpecialization::TextToSpeech {} => "Text to Speech".to_string(),
            ModelSpecialization::Translation {} => "Translation".to_string(),
            ModelSpecialization::Speculation {} => "Speculation".to_string(),
        }
    }
}
