use shoji::{
    traits::backend::chat_message::ToolCallState as OutputToolCallState,
    types::{basic::Value, session::chat::ToolCall},
};

use crate::openai::stream_state::ToolCallChunk;

#[derive(Debug, Default)]
pub struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallState {
    pub fn merge(
        &mut self,
        chunk: ToolCallChunk,
    ) {
        if let Some(id) = chunk.id {
            self.id.push_str(&id);
        }
        if let Some(name) = chunk.name {
            self.name.push_str(&name);
        }
        if let Some(arguments) = chunk.arguments {
            self.arguments.push_str(&arguments);
        }
    }

    pub fn build(
        &self,
        finalize: bool,
    ) -> OutputToolCallState {
        if finalize {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&self.arguments) {
                return OutputToolCallState::Finished(ToolCall {
                    identifier: Some(self.id.clone()),
                    name: self.name.clone(),
                    arguments: Value {
                        json: parsed.to_string(),
                    },
                });
            }
        }
        OutputToolCallState::Candidate(self.arguments.clone())
    }
}
