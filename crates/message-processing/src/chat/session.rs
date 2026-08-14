use std::{ops::Deref, sync::Arc};

use uzu_types::{
    core::{InferenceModel, InferenceState},
    session::chat::{ChatMessage, ChatSessionConfig},
};

use crate::chat::error::ChatSessionError;

pub struct ChatSessionResult {}

pub struct ChatSession {
    model: Arc<dyn InferenceModel>,
    state: Box<dyn InferenceState>,
}

impl ChatSession {
    pub fn new(
        model: Arc<dyn InferenceModel>,
        config: ChatSessionConfig,
    ) -> Result<Self, ChatSessionError> {
        Ok(Self {
            model: model.clone(),
            state: model.create_empty_state(),
        })
    }

    pub fn reply(
        &self,
        input: &[ChatMessage],
    ) -> ChatSessionResult {
        self.model.reply(input, self.state);
        ChatSessionResult {}
    }
}
