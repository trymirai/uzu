use serde::{Deserialize, Serialize};
use shoji::types::{basic::Token, session::chat::ChatMessage};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SynchronizationError {
    #[error("Stream desynchronization")]
    Desynchronization,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SynchronizationResult {
    Updated,
    Inserted,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct State {
    pub tokens: Vec<Token>,
    pub messages: Vec<ChatMessage>,
}

impl State {
    pub fn text(&self) -> String {
        self.tokens.iter().map(|token| token.value.clone()).collect()
    }

    pub fn synchronize_messages(
        &mut self,
        streamed_messages: &[ChatMessage],
    ) -> Result<SynchronizationResult, SynchronizationError> {
        let last_message = self.messages.last().ok_or(SynchronizationError::Desynchronization)?;
        let last_matched_index = streamed_messages
            .iter()
            .rposition(|streamed_message| streamed_message.role == last_message.role)
            .ok_or(SynchronizationError::Desynchronization)?;

        let remaining_count = streamed_messages.len() - 1 - last_matched_index;
        let last_streamed = streamed_messages.last().ok_or(SynchronizationError::Desynchronization)?;

        match remaining_count {
            0 => {
                let position = self.messages.len() - 1;
                self.messages[position] = last_streamed.clone();
                Ok(SynchronizationResult::Updated)
            },
            1 => {
                self.messages.push(last_streamed.clone());
                Ok(SynchronizationResult::Inserted)
            },
            _ => Err(SynchronizationError::Desynchronization),
        }
    }
}
