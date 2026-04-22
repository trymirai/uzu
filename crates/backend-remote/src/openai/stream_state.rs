use std::time::Instant;

use indexmap::IndexMap;
use shoji::{
    traits::backend::chat_message::Output,
    types::session::chat::{ChatFinishReason, ChatStats},
};

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
    pub finish_reason: Option<ChatFinishReason>,
    pub tokens_input: Option<u32>,
    pub tokens_output: Option<u32>,
}

#[derive(Debug)]
pub struct StreamState {
    text: Option<String>,
    reasoning: Option<String>,
    tool_calls: IndexMap<u32, ToolCallState>,
    finish_reason: Option<ChatFinishReason>,
    start_moment: Instant,
    first_token_moment: Option<Instant>,
    tokens_input: Option<u32>,
    tokens_output: Option<u32>,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            text: None,
            reasoning: None,
            tool_calls: IndexMap::new(),
            finish_reason: None,
            start_moment: Instant::now(),
            first_token_moment: None,
            tokens_input: None,
            tokens_output: None,
        }
    }

    pub fn process_chunk(
        &mut self,
        chunk: StreamChunk,
    ) -> Option<Output> {
        let mut processed = false;
        let produced_output = chunk.content.is_some() || chunk.reasoning.is_some() || !chunk.tool_calls.is_empty();

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

        if produced_output && self.first_token_moment.is_none() {
            self.first_token_moment = Some(Instant::now());
        }

        if let Some(tokens) = chunk.tokens_input {
            self.tokens_input = Some(tokens);
            processed = true;
        }

        if let Some(tokens) = chunk.tokens_output {
            self.tokens_output = Some(tokens);
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
            stats: self.stats(),
        }
    }

    fn stats(&self) -> ChatStats {
        let duration = self.start_moment.elapsed().as_secs_f64();
        let time_to_first_token =
            self.first_token_moment.map(|first| first.duration_since(self.start_moment).as_secs_f64());
        let prefill_tokens_per_second = match (self.tokens_input, time_to_first_token) {
            (Some(tokens_input), Some(time_to_first_token)) if time_to_first_token > 0.0 => {
                Some(tokens_input as f64 / time_to_first_token)
            },
            _ => None,
        };
        let generate_tokens_per_second = match (self.tokens_output, time_to_first_token) {
            (Some(tokens_output), Some(time_to_first_token))
                if (tokens_output > 0) && (duration > time_to_first_token) =>
            {
                Some((tokens_output - 1) as f64 / (duration - time_to_first_token))
            },
            _ => None,
        };
        ChatStats {
            duration,
            time_to_first_token,
            prefill_tokens_per_second,
            generate_tokens_per_second,
            tokens_count_input: self.tokens_input,
            tokens_count_output: self.tokens_output,
        }
    }
}
