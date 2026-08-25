use async_openai::types::responses::{OutputItem, ResponseStreamEvent, ResponseUsage};

use crate::openai::{
    bridging::responses::finish_reason,
    stream_state::{StreamChunk, ToolCallChunk},
};

fn usage_parts(usage: &ResponseUsage) -> (Option<u32>, Option<u32>, Option<u32>) {
    let cached = usage.input_tokens_details.cached_tokens;
    (Some(usage.input_tokens.saturating_sub(cached)), Some(cached), Some(usage.output_tokens))
}

pub fn build(event: ResponseStreamEvent) -> Option<StreamChunk> {
    match event {
        ResponseStreamEvent::ResponseOutputTextDelta(event) => Some(StreamChunk {
            content: Some(event.delta),
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseReasoningTextDelta(event) => Some(StreamChunk {
            reasoning: Some(event.delta),
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(event) => Some(StreamChunk {
            reasoning: Some(event.delta),
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseOutputItemAdded(event) => match event.item {
            OutputItem::FunctionCall(call) => Some(StreamChunk {
                tool_calls: vec![ToolCallChunk {
                    index: event.output_index,
                    id: Some(call.call_id),
                    name: Some(call.name),
                    arguments: None,
                }],
                ..StreamChunk::default()
            }),
            _ => None,
        },
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(event) => Some(StreamChunk {
            tool_calls: vec![ToolCallChunk {
                index: event.output_index,
                id: None,
                name: None,
                arguments: Some(event.delta),
            }],
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseCompleted(event) => {
            let (tokens_input, tokens_input_cached, tokens_output) =
                event.response.usage.as_ref().map(usage_parts).unwrap_or_default();
            Some(StreamChunk {
                finish_reason: finish_reason::build(&event.response.status),
                tokens_input,
                tokens_input_cached,
                tokens_output,
                ..StreamChunk::default()
            })
        },
        ResponseStreamEvent::ResponseFailed(event) => {
            let (tokens_input, tokens_input_cached, tokens_output) =
                event.response.usage.as_ref().map(usage_parts).unwrap_or_default();
            Some(StreamChunk {
                finish_reason: finish_reason::build(&event.response.status),
                tokens_input,
                tokens_input_cached,
                tokens_output,
                ..StreamChunk::default()
            })
        },
        ResponseStreamEvent::ResponseIncomplete(event) => {
            let (tokens_input, tokens_input_cached, tokens_output) =
                event.response.usage.as_ref().map(usage_parts).unwrap_or_default();
            Some(StreamChunk {
                finish_reason: finish_reason::build(&event.response.status),
                tokens_input,
                tokens_input_cached,
                tokens_output,
                ..StreamChunk::default()
            })
        },
        _ => None,
    }
}
