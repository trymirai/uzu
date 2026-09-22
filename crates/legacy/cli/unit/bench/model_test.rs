use serde_json::json;
use uzu::types::session::chat::{ChatContentBlock, ChatRole};

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
    let messages = task.to_chat_messages().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, ChatRole::User {});
    assert_eq!(messages[0].text().as_deref(), Some("Hello"));
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
    let messages = task.to_chat_messages().unwrap();
    assert_eq!(messages.len(), 6);
    assert_eq!(messages[0].role, ChatRole::System {});
    assert_eq!(messages[1].role, ChatRole::Developer {});
    assert_eq!(messages[1].tool_namespaces()[0].tools.len(), 1);
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
fn malformed_tool_definition_returns_contextual_error() {
    let mut input = task(json!([{"role": "user", "content": "Hello"}]));
    input["tools"] = json!([{"type": "function", "function": {"parameters": {}}}]);
    let task: BenchTask = serde_json::from_value(input).unwrap();
    assert!(task.to_chat_messages().unwrap_err().to_string().contains("benchmark tools"));
}

#[test]
fn malformed_tool_call_returns_contextual_error() {
    let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": {}}}
    ]}]));
    let task: BenchTask = serde_json::from_value(input).unwrap();
    assert!(task.to_chat_messages().unwrap_err().to_string().contains("benchmark messages"));
}

#[test]
fn invalid_tool_arguments_are_rejected_without_rewriting_history() {
    for arguments in ["", "not json", "[]", "null", "1", "\"text\"", "{\"x\":"] {
        let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": arguments}}
        ]}]));
        let task: BenchTask = serde_json::from_value(input).unwrap();
        assert!(task.to_chat_messages().unwrap_err().to_string().contains("tool arguments"));
    }
}

#[test]
fn nested_object_tool_arguments_are_preserved() {
    let arguments = json!({"options": {"zones": ["UTC", "GMT"], "enabled": true}, "offset": null});
    let input = task(json!([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": arguments.to_string()}}
    ]}]));
    let task: BenchTask = serde_json::from_value(input).unwrap();
    let messages = task.to_chat_messages().unwrap();
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
        let error = task.to_chat_messages().unwrap_err();
        assert!(!format!("{error:?}").contains("PRIVATE_SENTINEL"));
    }
}
