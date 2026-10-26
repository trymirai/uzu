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
    reasoning_content: Option<String>,
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
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

pub fn build(response: Response) -> Option<StreamChunk> {
    let usage = response.usage;
    let choice = response.choices.into_iter().next();
    let tokens_input_cached =
        usage.as_ref().and_then(|usage| usage.prompt_tokens_details.as_ref()).and_then(|details| details.cached_tokens);
    let tokens_input = usage
        .as_ref()
        .and_then(|usage| usage.prompt_tokens)
        .map(|tokens| tokens.saturating_sub(tokens_input_cached.unwrap_or(0)));
    let tokens_output = usage.as_ref().and_then(|usage| usage.completion_tokens);

    let (content, reasoning, tool_calls, finish_reason) = match choice {
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
            (delta.content, delta.reasoning_content, tool_calls, choice.finish_reason.map(finish_reason::build))
        },
        None => (None, None, Vec::new(), None),
    };

    if content.is_none()
        && reasoning.is_none()
        && tool_calls.is_empty()
        && finish_reason.is_none()
        && tokens_input.is_none()
        && tokens_input_cached.is_none()
        && tokens_output.is_none()
    {
        return None;
    }

    Some(StreamChunk {
        content,
        reasoning,
        tool_calls,
        finish_reason,
        tokens_input,
        tokens_input_cached,
        tokens_output,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Response, build};
    use crate::openai::stream_state::{StreamChunk, StreamState};

    /// Builds the chunk from the provider's wire shape rather than from the decoder
    /// struct, so the field names the mapping depends on stay pinned to what the
    /// API actually sends.
    fn chunk(delta: serde_json::Value) -> Option<StreamChunk> {
        let response: Response = serde_json::from_value(json!({
            "choices": [{ "delta": delta }],
        }))
        .expect("response fixture should deserialize");
        build(response)
    }

    #[test]
    fn reasoning_only_delta_yields_a_chunk() {
        let chunk = chunk(json!({ "reasoning_content": "thinking" })).expect("a reasoning-only delta yields a chunk");

        assert_eq!(chunk.reasoning.as_deref(), Some("thinking"));
        assert_eq!(chunk.content, None);
    }

    #[test]
    fn content_and_reasoning_both_survive() {
        let chunk = chunk(json!({ "content": "answer", "reasoning_content": "thinking" }))
            .expect("a delta with content and reasoning yields a chunk");

        assert_eq!(chunk.content.as_deref(), Some("answer"));
        assert_eq!(chunk.reasoning.as_deref(), Some("thinking"));
    }

    #[test]
    fn content_only_delta_has_no_reasoning() {
        let chunk = chunk(json!({ "content": "answer" })).expect("a content-only delta yields a chunk");

        assert_eq!(chunk.content.as_deref(), Some("answer"));
        assert_eq!(chunk.reasoning, None);
    }

    #[test]
    fn empty_delta_yields_no_chunk() {
        assert!(chunk(json!({})).is_none());
        assert!(chunk(json!({ "role": "assistant" })).is_none());
    }

    #[test]
    fn tool_calls_parse_alongside_reasoning() {
        let chunk = chunk(json!({
            "reasoning_content": "thinking",
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": { "name": "get_weather", "arguments": "{\"city\":" },
            }],
        }))
        .expect("a delta with reasoning and tool calls yields a chunk");

        assert_eq!(chunk.reasoning.as_deref(), Some("thinking"));
        assert_eq!(chunk.tool_calls.len(), 1);
        assert_eq!(chunk.tool_calls[0].index, 0);
        assert_eq!(chunk.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(chunk.tool_calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(chunk.tool_calls[0].arguments.as_deref(), Some("{\"city\":"));
    }

    /// Drives the real accumulation path, so reasoning streamed a token at a time
    /// reaches the reply the CLI server renders.
    #[test]
    fn successive_reasoning_chunks_accumulate() {
        let mut state = StreamState::new();
        let mut output = None;

        for reasoning in ["first ", "second"] {
            let chunk = chunk(json!({ "reasoning_content": reasoning })).expect("each delta yields a chunk");
            output = state.process_chunk(chunk);
        }

        let output = output.expect("the last chunk produces output");
        assert_eq!(output.reasoning.as_deref(), Some("first second"));
    }
}
