pub enum ModelCapability {
    Chat,
    Classification,
    TextToSpeech,
}

impl ModelCapability {
    pub fn name(&self) -> String {
        match self {
            ModelCapability::Chat {} => "Chat".to_string(),
            ModelCapability::Classification {} => "Classification".to_string(),
            ModelCapability::TextToSpeech {} => "Text to Speech".to_string(),
        }
    }
}
