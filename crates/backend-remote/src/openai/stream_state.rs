use indexmap::IndexMap;
use shoji::{traits::backend::chat_message::Output, types::session::chat::FinishReason};

use crate::openai::tool_call_state::ToolCallState;

#[derive(Debug)]
pub struct ToolCallChunk {
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Default)]
pub struct StreamChunk {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallChunk>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Default)]
pub struct StreamState {
    text: Option<String>,
    reasoning: Option<String>,
    tool_calls: IndexMap<u32, ToolCallState>,
    finish_reason: Option<FinishReason>,
}

impl StreamState {
    pub fn process_chunk(
        &mut self,
        chunk: StreamChunk,
    ) -> Option<Output> {
        let mut processed = false;

        if let Some(content) = chunk.content {
            self.text.get_or_insert_with(String::new).push_str(&content);
            processed = true;
        }

        if let Some(reasoning) = chunk.reasoning {
            self.reasoning.get_or_insert_with(String::new).push_str(&reasoning);
            processed = true;
        }

        if !chunk.tool_calls.is_empty() {
            for call in chunk.tool_calls {
                self.tool_calls.entry(call.index).or_default().merge(call);
            }
            processed = true;
        }

        if let Some(reason) = chunk.finish_reason {
            self.finish_reason = Some(reason);
            processed = true;
        }

        if processed {
            Some(self.build())
        } else {
            None
        }
    }

    fn build(&self) -> Output {
        let finalize = self.finish_reason.is_some();
        Output {
            reasoning: self.reasoning.clone(),
            text: self.text.clone(),
            tool_calls: self.tool_calls.values().map(|state| state.build(finalize)).collect(),
            finish_reason: self.finish_reason.clone(),
        }
    }
}
