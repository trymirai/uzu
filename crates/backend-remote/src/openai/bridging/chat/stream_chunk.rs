use async_openai::types::chat::FinishReason;
use serde::Deserialize;

use crate::openai::{
    bridging::chat::finish_reason,
    stream_state::{StreamChunk, ToolCallChunk},
};

#[derive(Debug, Deserialize)]
pub struct Response {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
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

#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
}

pub fn build(response: Response) -> Option<StreamChunk> {
    let usage = response.usage;
    let choice = response.choices.into_iter().next();
    let tokens_input = usage.as_ref().and_then(|usage| usage.prompt_tokens);
    let tokens_output = usage.as_ref().and_then(|usage| usage.completion_tokens);

    let (content, tool_calls, finish_reason) = match choice {
        Some(choice) => {
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
            (delta.content, tool_calls, choice.finish_reason.map(finish_reason::build))
        },
        None => (None, Vec::new(), None),
    };

    if content.is_none()
        && tool_calls.is_empty()
        && finish_reason.is_none()
        && tokens_input.is_none()
        && tokens_output.is_none()
    {
        return None;
    }

    Some(StreamChunk {
        content,
        reasoning: None,
        tool_calls,
        finish_reason,
        tokens_input,
        tokens_output,
    })
}
