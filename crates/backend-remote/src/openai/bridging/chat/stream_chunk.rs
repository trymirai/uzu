use async_openai::types::chat::FinishReason;
use serde::Deserialize;

use crate::openai::{
    bridging::chat::finish_reason,
    stream_state::{StreamChunk, ToolCallChunk},
};

#[derive(Debug, Deserialize)]
pub struct Response {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionCallStream>,
}

#[derive(Debug, Deserialize)]
struct FunctionCallStream {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

pub fn build(response: Response) -> Option<StreamChunk> {
    let choice = response.choices.into_iter().next()?;
    let delta = choice.delta;
    let tool_calls = delta
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(position, call)| {
            let (name, arguments) = call.function.map_or((None, None), |f| (f.name, f.arguments));
            ToolCallChunk {
                index: call.index.unwrap_or(position as u32),
                id: call.id,
                name,
                arguments,
            }
        })
        .collect();
    Some(StreamChunk {
        content: delta.content,
        reasoning: None,
        tool_calls,
        finish_reason: choice.finish_reason.map(finish_reason::build),
    })
}
