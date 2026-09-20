use backend_needle::{
    Error, NEEDLE3_TAG,
    mapping::{bind_tools, complete_inputs, map_envelope, messages::CompleteInput, system_text, tools::tools_json},
};
use shoji::types::{
    basic::{Grammar, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatReplyFinishReason, ChatRole},
};

fn function_tool(
    name: &str,
    description: &str,
) -> ToolDescription {
    ToolDescription::Function {
        tool_function: ToolFunction {
            name: name.to_string(),
            description: description.to_string(),
            parameters: Some(Value {
                json: r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#.to_string(),
            }),
            return_definition: None,
        },
    }
}

fn tools_message(tools: Vec<ToolDescription>) -> ChatMessage {
    ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
        name: "default".to_string(),
        description: None,
        tools,
    }])
}

fn tool_result(json: &str) -> ChatMessage {
    ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier: Some("needle-0".to_string()),
        name: Some("get_weather".to_string()),
        value: Value {
            json: json.to_string(),
        },
    })
}

#[test]
fn tools_json_flattens_and_sorts() {
    let messages =
        vec![tools_message(vec![function_tool("get_weather", "weather"), function_tool("set_lights", "lights")])];
    let json = tools_json(&messages).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed[0]["name"], "get_weather");
    assert_eq!(parsed[1]["name"], "set_lights");
    assert_eq!(parsed[0]["parameters"]["properties"]["city"]["type"], "string");
}

#[test]
fn tools_json_rejects_duplicate_names() {
    let messages = vec![ChatMessage::developer().with_tool_namespaces(vec![
        ToolNamespace {
            name: "a".to_string(),
            description: None,
            tools: vec![function_tool("dup", "one")],
        },
        ToolNamespace {
            name: "b".to_string(),
            description: None,
            tools: vec![function_tool("dup", "two")],
        },
    ])];
    let error = tools_json(&messages).unwrap_err();
    assert!(matches!(error, Error::DuplicateToolName { name } if name == "dup"));
}

#[test]
fn complete_inputs_batches_tool_results_as_array() {
    let messages = vec![
        ChatMessage::user().with_text("weather in Lagos".to_string()),
        ChatMessage::assistant().with_text(String::new()),
        tool_result(r#"{"city":"Lagos","temp_c":27}"#),
        tool_result(r#"{"ok":true}"#),
    ];
    let inputs = complete_inputs(&messages).unwrap();
    assert_eq!(
        inputs,
        vec![
            CompleteInput::User("weather in Lagos".to_string()),
            CompleteInput::ToolResults(vec![
                serde_json::json!({"city":"Lagos","temp_c":27}),
                serde_json::json!({"ok":true}),
            ]),
        ]
    );
    assert_eq!(inputs[1].as_complete_text().unwrap(), r#"[{"city":"Lagos","temp_c":27},{"ok":true}]"#);
}

#[test]
fn complete_inputs_skips_assistant_and_joins_system() {
    let messages = vec![
        ChatMessage::system().with_text("date: 2026-09-20".to_string()),
        ChatMessage::system().with_text("locale: en-US".to_string()),
        ChatMessage::user().with_text("hello".to_string()),
        ChatMessage::assistant().with_tool_call(shoji::types::basic::ToolCall {
            identifier: Some("needle-0".to_string()),
            name: "get_weather".to_string(),
            arguments: Value {
                json: "{}".to_string(),
            },
        }),
        tool_result(r#"{"city":"Lagos"}"#),
        ChatMessage::user().with_text("thanks".to_string()),
    ];
    assert_eq!(system_text(&messages), "date: 2026-09-20; locale: en-US");
    let inputs = complete_inputs(&messages).unwrap();
    assert_eq!(
        inputs,
        vec![
            CompleteInput::User("hello".to_string()),
            CompleteInput::ToolResults(vec![serde_json::json!({"city":"Lagos"})]),
            CompleteInput::User("thanks".to_string()),
        ]
    );
}

#[test]
fn single_tool_result_is_still_an_array() {
    let messages = vec![tool_result(r#"{"city":"Lagos"}"#)];
    let inputs = complete_inputs(&messages).unwrap();
    assert_eq!(inputs[0].as_complete_text().unwrap(), r#"[{"city":"Lagos"}]"#);
}

#[test]
fn rejects_image_content() {
    let messages = vec![ChatMessage::user().with_block(ChatContentBlock::Image {
        url: "https://example.com/a.png".to_string(),
    })];
    assert!(matches!(complete_inputs(&messages), Err(Error::UnsupportedContent)));
}

#[test]
fn envelope_call_maps_to_tool_calls() {
    let envelope = serde_json::json!({
        "type": "call",
        "success": true,
        "function_calls": [{"name": "get_weather", "arguments": {"city": "Lagos"}}],
        "reasoning": "Lagos -> city",
        "confidence": 0.94,
        "prefill_tps": 100.0,
        "decode_tps": 50.0
    });
    let output = map_envelope(envelope, 0.01, false).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::ToolCalls));
    assert_eq!(output.tool_calls.len(), 1);
    assert!(output.reasoning.unwrap().starts_with("[confidence=0.94] "));
    assert!(output.text.is_none());
}

#[test]
fn envelope_empty_calls_are_stop() {
    let envelope = serde_json::json!({
        "type": "call",
        "success": true,
        "function_calls": [],
        "reasoning": "off topic"
    });
    let output = map_envelope(envelope, 0.01, false).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::Stop));
    assert!(output.tool_calls.is_empty());
}

#[test]
fn envelope_suppressed_is_rejected_without_tool_calls() {
    let envelope = serde_json::json!({
        "type": "call",
        "success": true,
        "function_calls": [{"name": "get_weather", "arguments": {"city": "Lagos"}}],
        "suppressed_calls": [{"name": "get_weather", "arguments": {"city": "Lagos"}}]
    });
    let output = map_envelope(envelope, 0.01, false).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::Rejected));
    assert!(output.tool_calls.is_empty());
    assert!(output.reasoning.unwrap().contains("[needle withheld]"));
}

#[test]
fn envelope_ungrounded_is_rejected() {
    let envelope = serde_json::json!({
        "type": "call",
        "success": true,
        "function_calls": [{"name": "pay", "arguments": {"amount": 5}}],
        "validation": { "ungrounded": ["pay.amount"] }
    });
    let output = map_envelope(envelope, 0.01, false).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::Rejected));
    assert!(output.tool_calls.is_empty());
}

#[test]
fn envelope_error_is_failure() {
    let envelope = serde_json::json!({
        "success": false,
        "error": "boom"
    });
    let error = map_envelope(envelope, 0.01, false).unwrap_err();
    assert!(matches!(error, Error::CompleteFailed { message, .. } if message == "boom"));
}

#[test]
fn grammar_bind_uses_title_and_maps_stop_text() {
    let schema =
        r#"{"title":"Invoice","type":"object","properties":{"vendor":{"type":"string"}},"required":["vendor"]}"#;
    let bound = bind_tools(
        &[],
        Some(&Grammar::JsonSchema {
            schema: schema.to_string(),
        }),
    )
    .unwrap();
    assert!(bound.is_grammar);
    let tools: serde_json::Value = serde_json::from_str(&bound.json).unwrap();
    assert_eq!(tools[0]["name"], "Invoice");

    let envelope = serde_json::json!({
        "type": "call",
        "success": true,
        "function_calls": [{"name": "Invoice", "arguments": {"vendor": "Acme"}}]
    });
    let output = map_envelope(envelope, 0.01, true).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::Stop));
    assert!(output.text.unwrap().contains("Acme"));
    assert_eq!(output.tool_calls.len(), 1);
}

#[test]
fn regex_grammar_is_unsupported() {
    let error = bind_tools(
        &[],
        Some(&Grammar::Regex {
            pattern: ".*".to_string(),
        }),
    )
    .unwrap_err();
    assert!(matches!(error, Error::UnsupportedGrammar));
}

#[test]
fn needle3_tag_roundtrip() {
    let tag = u32::from_le_bytes([
        (NEEDLE3_TAG & 0xff) as u8,
        ((NEEDLE3_TAG >> 8) & 0xff) as u8,
        ((NEEDLE3_TAG >> 16) & 0xff) as u8,
        ((NEEDLE3_TAG >> 24) & 0xff) as u8,
    ]);
    assert_eq!(tag, NEEDLE3_TAG);
}

#[test]
fn custom_role_is_skipped() {
    let messages = vec![
        ChatMessage::for_role(ChatRole::Custom {
            name: "note".to_string(),
        })
        .with_text("ignore me".to_string()),
        ChatMessage::user().with_text("hi".to_string()),
    ];
    let inputs = complete_inputs(&messages).unwrap();
    assert_eq!(inputs, vec![CompleteInput::User("hi".to_string())]);
}

#[test]
fn tools_json_defaults_missing_parameters() {
    let messages = vec![ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
        name: "default".to_string(),
        description: None,
        tools: vec![ToolDescription::Function {
            tool_function: ToolFunction {
                name: "ping".to_string(),
                description: "ping".to_string(),
                parameters: None,
                return_definition: None,
            },
        }],
    }])];
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&tools_json(&messages).unwrap()).unwrap();
    assert_eq!(parsed[0]["parameters"]["type"], "object");
    assert_eq!(parsed[0]["parameters"]["properties"], serde_json::json!({}));
}

#[test]
fn grammar_without_title_uses_extract() {
    let schema = r#"{"type":"object","properties":{"vendor":{"type":"string"}}}"#;
    let bound = bind_tools(
        &[],
        Some(&Grammar::JsonSchema {
            schema: schema.to_string(),
        }),
    )
    .unwrap();
    let tools: serde_json::Value = serde_json::from_str(&bound.json).unwrap();
    assert_eq!(tools[0]["name"], "extract");
}

#[test]
fn envelope_respond_is_stop() {
    let envelope = serde_json::json!({
        "type": "respond",
        "success": true,
        "function_calls": [{"name": "get_weather", "arguments": {"city": "Lagos"}}]
    });
    let output = map_envelope(envelope, 0.01, false).unwrap();
    assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::Stop));
}
