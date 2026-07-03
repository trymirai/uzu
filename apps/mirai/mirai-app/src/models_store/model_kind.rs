use uzu::types::model::Model;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Chat,

    CloudChat,
    Classification,
    TextToSpeech,
}

impl ModelKind {
    pub(super) fn matches(
        self,
        model: &Model,
    ) -> bool {
        match self {
            ModelKind::Chat => model.is_chat_capable() && model.is_local(),
            ModelKind::CloudChat => model.is_chat_capable() && model.is_remote(),
            ModelKind::Classification => model.is_classification_capable(),
            ModelKind::TextToSpeech => model.is_text_to_speech_capable(),
        }
    }
}
