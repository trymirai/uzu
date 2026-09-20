use shoji::{
    traits::backend::chat_message::{Output, ToolCallState},
    types::{
        basic::{ToolCall, Value},
        session::chat::{ChatReplyFinishReason, ChatReplyStats},
    },
};

use crate::error::Error;

pub fn empty_stop_output(duration: f64) -> Output {
    Output {
        reasoning: None,
        text: None,
        tool_calls: vec![],
        finish_reason: Some(ChatReplyFinishReason::Stop),
        stats: stats(duration, None, None),
    }
}

pub fn map_envelope(
    envelope: serde_json::Value,
    duration: f64,
    is_grammar: bool,
) -> Result<Output, Error> {
    if envelope.get("success").and_then(serde_json::Value::as_bool) == Some(false) {
        return Err(complete_error(&envelope, 0));
    }
    if nonempty(envelope.get("error")) {
        return Err(complete_error(&envelope, 0));
    }

    let function_calls = envelope.get("function_calls").cloned().unwrap_or(serde_json::Value::Array(vec![]));
    let calls = function_calls.as_array().cloned().unwrap_or_default();
    let suppressed = envelope.get("suppressed_calls");
    let ungrounded = envelope.pointer("/validation/ungrounded");
    let withheld = nonempty(suppressed) || nonempty(ungrounded);

    let mut reasoning = envelope.get("reasoning").and_then(serde_json::Value::as_str).map(str::to_string);
    if let Some(confidence) = envelope.get("confidence").and_then(serde_json::Value::as_f64) {
        let prefix = format!("[confidence={confidence}] ");
        reasoning = Some(match reasoning {
            Some(text) if !text.is_empty() => format!("{prefix}{text}"),
            _ => prefix.trim_end().to_string(),
        });
    }

    if withheld {
        let mut extra = Vec::new();
        if nonempty(suppressed) {
            extra.push(format!("suppressed={}", suppressed.unwrap()));
        }
        if nonempty(ungrounded) {
            extra.push(format!("ungrounded={}", ungrounded.unwrap()));
        }
        let withheld_text = format!("[needle withheld] {}", extra.join(" "));
        reasoning = Some(match reasoning {
            Some(text) if !text.is_empty() => format!("{text} {withheld_text}"),
            _ => withheld_text,
        });
        return Ok(Output {
            reasoning,
            text: None,
            tool_calls: vec![],
            finish_reason: Some(ChatReplyFinishReason::Rejected),
            stats: stats_from_envelope(&envelope, duration),
        });
    }

    let mut tool_calls = Vec::new();
    let mut extract_text = None;
    for (index, call) in calls.iter().enumerate() {
        let name = call.get("name").and_then(serde_json::Value::as_str).unwrap_or("").to_string();
        let arguments = call.get("arguments").cloned().unwrap_or(serde_json::Value::Object(Default::default()));
        if is_grammar && index == 0 {
            extract_text = Some(serde_json::to_string_pretty(&arguments)?);
        }
        tool_calls.push(ToolCallState::Finished(ToolCall {
            identifier: Some(format!("needle-{index}")),
            name,
            arguments: Value {
                json: serde_json::to_string(&arguments)?,
            },
        }));
    }

    let finish_reason = if withheld {
        ChatReplyFinishReason::Rejected
    } else if is_grammar
        || tool_calls.is_empty()
        || envelope.get("type").and_then(serde_json::Value::as_str) == Some("respond")
    {
        ChatReplyFinishReason::Stop
    } else {
        ChatReplyFinishReason::ToolCalls
    };

    Ok(Output {
        reasoning,
        text: extract_text,
        tool_calls,
        finish_reason: Some(finish_reason),
        stats: stats_from_envelope(&envelope, duration),
    })
}

fn stats_from_envelope(
    envelope: &serde_json::Value,
    duration: f64,
) -> ChatReplyStats {
    stats(
        duration,
        envelope.get("prefill_tps").and_then(serde_json::Value::as_f64),
        envelope.get("decode_tps").and_then(serde_json::Value::as_f64),
    )
}

fn stats(
    duration: f64,
    prefill: Option<f64>,
    decode: Option<f64>,
) -> ChatReplyStats {
    ChatReplyStats {
        duration,
        time_to_first_token: Some(duration),
        prefill_tokens_per_second: prefill,
        generate_tokens_per_second: decode,
        ..ChatReplyStats::default()
    }
}

fn nonempty(value: Option<&serde_json::Value>) -> bool {
    match value {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Array(items)) => !items.is_empty(),
        Some(serde_json::Value::Object(map)) => !map.is_empty(),
        Some(serde_json::Value::String(text)) => !text.is_empty(),
        Some(serde_json::Value::Bool(false)) => false,
        Some(_) => true,
    }
}

fn complete_error(
    envelope: &serde_json::Value,
    code: i32,
) -> Error {
    let message = envelope
        .get("error")
        .map(|error| match error {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .filter(|text| text != "null" && !text.is_empty())
        .unwrap_or_else(|| "needle engine error".to_string());
    Error::CompleteFailed {
        code,
        message,
    }
}
