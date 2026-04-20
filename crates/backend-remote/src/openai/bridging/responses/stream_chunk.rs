use async_openai::types::responses::{OutputItem, ResponseStreamEvent};

use crate::openai::{
    bridging::responses::finish_reason,
    stream_state::{StreamChunk, ToolCallChunk},
};

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
        ResponseStreamEvent::ResponseCompleted(event) => Some(StreamChunk {
            finish_reason: finish_reason::build(&event.response.status),
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseFailed(event) => Some(StreamChunk {
            finish_reason: finish_reason::build(&event.response.status),
            ..StreamChunk::default()
        }),
        ResponseStreamEvent::ResponseIncomplete(event) => Some(StreamChunk {
            finish_reason: finish_reason::build(&event.response.status),
            ..StreamChunk::default()
        }),
        _ => None,
    }
}
