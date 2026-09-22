use serde_json::json;
use uzu::types::{
    basic::ToolDescription,
    session::chat::{ChatContentBlock, ChatRole},
};

use super::*;

fn task(messages: serde_json::Value) -> serde_json::Value {
    json!({
        "identifier": "replay", "repo_id": "test/model", "number_of_runs": 2,
        "tokens_limit": 16, "greedy": true, "messages": messages
    })
}

#[test]
fn legacy_task_still_converts_without_tools() {
    let input = task(json!([{"role": "user", "content": "Hello"}]));
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, ChatRole::User {});
    assert_eq!(messages[0].text().as_deref(), Some("Hello"));
}

#[test]
fn assistant_tool_call_accepts_null_content() {
    let input = task(json!([{"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": "{}"}}
    ]}]));
    let task: BenchTask = serde_json::from_value(input.clone()).unwrap();
    let echoed = serde_json::to_value(&task).unwrap();
    assert_eq!(echoed["messages"][0]["content"], serde_json::Value::Null);
    assert_eq!(echoed["messages"][0]["tool_calls"], input["messages"][0]["tool_calls"]);
    let messages = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap();
    assert_eq!(messages[0].role, ChatRole::Assistant {});
    assert_eq!(messages[0].text(), None);
    assert_eq!(messages[0].tool_calls()[0].identifier.as_deref(), Some("call_clock"));
}

#[test]
fn tool_replay_preserves_wire_input_and_builds_complete_context() {
    let mut input = task(json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": "Checking", "reasoning_content": "Use the clock",
         "tool_calls": [{"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": "{\"zone\":\"UTC\"}"}}]},
        {"role": "tool", "tool_call_id": "call_clock", "content": "{\"time\":\"12:00\"}"},
        {"role": "user", "content": "Summarize"}
    ]));
    input["tools"] = json!([{"type": "function", "function": {"name": "clock", "description": "Get time", "strict": true,
        "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "additionalProperties": false}}}]);
    let task: BenchTask = serde_json::from_value(input.clone()).unwrap();
    let echoed = serde_json::to_value(&task).unwrap();
    assert_eq!(echoed["tools"], input["tools"]);
    assert_eq!(echoed["messages"][2]["tool_calls"], input["messages"][2]["tool_calls"]);
    assert_eq!(echoed["messages"][3]["tool_call_id"], "call_clock");
    let messages = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap();
    assert_eq!(messages.len(), 6);
    assert_eq!(messages[0].role, ChatRole::System {});
    assert_eq!(messages[1].role, ChatRole::Developer {});
    let namespaces = messages[1].tool_namespaces();
    assert_eq!(namespaces[0].tools.len(), 1);
    let ToolDescription::Function {
        tool_function,
    } = &namespaces[0].tools[0];
    assert_eq!(tool_function.name, "clock");
    assert_eq!(tool_function.description, "Get time");
    assert_eq!(serde_json::to_value(&tool_function.parameters).unwrap(), input["tools"][0]["function"]["parameters"]);
    assert!(matches!(messages[3].content[0], ChatContentBlock::Reasoning { .. }));
    assert_eq!(messages[3].text().as_deref(), Some("Checking"));
    let calls = messages[3].tool_calls();
    assert_eq!(calls[0].identifier.as_deref(), Some("call_clock"));
    assert_eq!(serde_json::from_str::<serde_json::Value>(&calls[0].arguments.json).unwrap(), json!({"zone": "UTC"}));
    assert_eq!(messages[4].role, ChatRole::Tool {});
    let results = messages[4].tool_call_results();
    assert_eq!(results[0].0.as_deref(), Some("call_clock"));
    assert_eq!(results[0].1.as_deref(), Some("clock"));
    assert_eq!(serde_json::from_str::<serde_json::Value>(&results[0].2.json).unwrap(), json!("{\"time\":\"12:00\"}"));
}

#[test]
fn tool_choice_selects_benchmark_tools_and_is_preserved_in_echo() {
    for (choice, expected) in
        [(json!("none"), vec![]), (json!({"type": "function", "function": {"name": "clock"}}), vec!["clock"])]
    {
        let mut input = task(json!([{"role": "user", "content": "Hello"}]));
        input["tools"] = json!([
            {"type": "function", "function": {"name": "clock"}},
            {"type": "function", "function": {"name": "weather"}}
        ]);
        input["tool_choice"] = choice.clone();
        let task: BenchTask = serde_json::from_value(input).unwrap();
        let messages = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap();
        let names: Vec<_> = messages
            .iter()
            .flat_map(|message| message.tool_namespaces())
            .flat_map(|namespace| namespace.tools)
            .map(|tool| match tool {
                ToolDescription::Function {
                    tool_function,
                } => tool_function.name,
            })
            .collect();
        assert_eq!(names, expected);
        assert_eq!(serde_json::to_value(&task).unwrap()["tool_choice"], choice);
    }
}

#[test]
fn invalid_tool_choice_is_rejected_without_exposing_private_values() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["tool_choice"] = json!("PRIVATE_SENTINEL");
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let error = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap_err();
    assert!(error.to_string().contains("tool_choice"));
    assert!(!format!("{error:?}").contains("PRIVATE_SENTINEL"));
}

#[test]
fn malformed_tool_definition_returns_contextual_error() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["tools"] = json!([{"type": "function", "function": {"parameters": {}}}]);
    let task: BenchTask = serde_json::from_value(input).unwrap();
    assert!(task.to_chat_messages(ThinkingSupport::Unsupported).unwrap_err().to_string().contains("benchmark tools"));
}

#[test]
fn malformed_tool_call_returns_contextual_error() {
    let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": {}}}
    ]}]));
    let task: BenchTask = serde_json::from_value(input).unwrap();
    assert!(
        task.to_chat_messages(ThinkingSupport::Unsupported).unwrap_err().to_string().contains("benchmark messages")
    );
}

#[test]
fn invalid_tool_arguments_are_rejected_without_rewriting_history() {
    for arguments in ["", "not json", "[]", "null", "1", "\"text\"", "{\"x\":"] {
        let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": arguments}}
        ]}]));
        let task: BenchTask = serde_json::from_value(input).unwrap();
        assert!(
            task.to_chat_messages(ThinkingSupport::Unsupported).unwrap_err().to_string().contains("tool arguments")
        );
    }
}

#[test]
fn nested_object_tool_arguments_are_preserved() {
    let arguments = json!({"options": {"zones": ["UTC", "GMT"], "enabled": true}, "offset": null});
    let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": arguments.to_string()}}
    ]}]));
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&messages[0].tool_calls()[0].arguments.json).unwrap(),
        arguments
    );
}

#[test]
fn conversion_errors_do_not_include_private_values_in_debug_chain() {
    let mut tools_input = task(json!([{"role": "user", "content": "Hello"}]));
    tools_input["tools"] = json!(["PRIVATE_SENTINEL"]);
    let calls_input = task(json!([{"role": "assistant", "content": "", "tool_calls": ["PRIVATE_SENTINEL"]}]));
    for input in [tools_input, calls_input] {
        let task: BenchTask = serde_json::from_value(input).unwrap();
        let error = task.to_chat_messages(ThinkingSupport::Unsupported).unwrap_err();
        assert!(!format!("{error:?}").contains("PRIVATE_SENTINEL"));
    }
}

#[test]
fn benchmark_echo_preserves_requested_execution_settings() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["reasoning"] = json!(false);
    input["context_size"] = json!("auto");
    input["context_padding"] = json!(64);
    input["generation_config"] = json!({"temperature": 1.0, "top_p": 1.0, "top_k": 20,
        "stop_token_ids": [1, 2]});
    let parsed: BenchTask = serde_json::from_value(input.clone()).unwrap();
    let echoed = serde_json::to_value(parsed).unwrap();
    for key in ["reasoning", "context_size", "context_padding", "generation_config"] {
        assert_eq!(echoed[key], input[key], "{key}");
    }
}

#[test]
fn reasoning_respects_capabilities_and_preserves_leading_system() {
    let mut input = task(json!([{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Hello"}]));
    input["reasoning"] = json!(false);
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages(ThinkingSupport::Toggle(true)).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text().as_deref(), Some("Instructions"));
    assert!(messages[0].content.iter().any(|block| matches!(
        block,
        ChatContentBlock::ReasoningEffort {
            value: ReasoningEffort::Disabled
        }
    )));
    assert!(task.to_chat_messages(ThinkingSupport::AlwaysOn).is_err());
    assert!(task.to_chat_messages(ThinkingSupport::Unsupported).is_ok());
}

#[test]
fn reasoning_true_inserts_system_and_rejects_unsupported_models() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["reasoning"] = json!(true);
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages(ThinkingSupport::Toggle(false)).unwrap();
    assert_eq!(messages[0].role, ChatRole::System {});
    assert!(task.to_chat_messages(ThinkingSupport::Unsupported).is_err());
}

#[test]
fn context_resolution_checks_capacity_and_overflow() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["context_size"] = json!("auto");
    let task: BenchTask = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(task.resolve_context_size(100).unwrap(), Some(116));
    input["context_padding"] = json!(64);
    let task: BenchTask = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(task.resolve_context_size(100).unwrap(), Some(180));
    assert!(task.context_length().is_err());
    assert!(task.resolve_context_size(u64::MAX).is_err());
    input["context_size"] = json!(128);
    let task: BenchTask = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(task.resolve_context_size(100).unwrap(), Some(128));
    assert!(task.resolve_context_size(120).is_err());
    input["context_size"] = json!(0);
    assert!(serde_json::from_value::<BenchTask>(input).unwrap().validate().is_err());
}

#[test]
fn generation_config_overrides_sampling_and_rejects_unsupported_effects() {
    let value = json!({"temperature": 1.0, "top_p": 1.0, "top_k": 20, "stop_token_ids": [1, 2]});
    let config: BenchGenerationConfig = serde_json::from_value(value.clone()).unwrap();
    config.validate().unwrap();
    assert!(matches!(
        config.sampling_method(),
        SamplingMethod::Stochastic {
            temperature: Some(1.0),
            top_p: Some(1.0),
            top_k: Some(20),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None
        }
    ));
    for (key, invalid) in [
        ("temperature", json!(0)),
        ("temperature", json!(1e-40)),
        ("top_p", json!(-0.1)),
        ("min_p", json!(1.1)),
        ("presence_penalty", json!(1.0)),
        ("frequency_penalty", json!(1.0)),
        ("banned_tokens", json!([1])),
        ("repetition_penalty", json!(1.2)),
        ("suffix_repetition_length", json!(32)),
    ] {
        let mut input = value.clone();
        input[key] = invalid;
        assert!(serde_json::from_value::<BenchGenerationConfig>(input).unwrap().validate().is_err(), "{key}");
    }
    assert!(serde_json::from_value::<BenchGenerationConfig>(json!({"top_p": 1.0})).is_err());
    let mut input = value;
    input["top_k"] = json!(0);
    input["repetition_penalty"] = json!(1.0);
    let config: BenchGenerationConfig = serde_json::from_value(input).unwrap();
    config.validate().unwrap();
    assert!(matches!(
        config.sampling_method(),
        SamplingMethod::Stochastic {
            top_k: None,
            repetition_penalty: None,
            ..
        }
    ));
}

#[test]
fn generation_stop_overrides_cannot_be_silently_ignored() {
    let config: BenchGenerationConfig = serde_json::from_value(json!({
        "temperature": 1.0, "top_p": 1.0, "top_k": 20, "stop_token_ids": [1, 2]
    }))
    .unwrap();
    config.validate_stop_tokens(Some(&[1, 2])).unwrap();
    config.validate_stop_tokens(Some(&[2, 1])).unwrap();
    assert!(config.validate_stop_tokens(Some(&[1])).is_err());
    assert!(config.validate_stop_tokens(None).is_err());
}

#[test]
fn disabled_reasoning_reaches_the_qwen38_prompt_renderer() {
    use hanashi::chat::hanashi::{config::HanashiConfig, renderer::Renderer};

    let renderer = Renderer::new(HanashiConfig::Qwen38.resolve().unwrap().rendering);
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["reasoning"] = json!(false);
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages(ThinkingSupport::Toggle(true)).unwrap();
    let rendered = renderer.render(&messages, true, None, None, None).unwrap();
    assert_eq!(rendered, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

#[test]
fn invalid_context_types_are_rejected() {
    for context_size in [json!(-1), json!(true), json!("maximum"), json!(4294967296_u64), json!(1.5)] {
        let mut input = task(json!([{"role": "user", "content": "Hello"}]));
        input["context_size"] = context_size;
        assert!(serde_json::from_value::<BenchTask>(input).is_err());
    }
}
