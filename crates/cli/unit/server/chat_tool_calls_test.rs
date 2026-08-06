use super::*;
use crate::server::chat_completions::{ChatCompletionRequest, build_messages};

fn request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("valid request json")
}

fn messages(json: &str) -> Vec<ChatMessage> {
    build_messages(&request(json)).expect("valid request")
}

#[test]
fn tools_become_developer_message_after_system() {
    let messages = messages(
        r#"{
            "messages":[{"role":"system","content":"be nice"},{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{}}}}]
        }"#,
    );

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, ChatRole::Developer {});
    let namespaces = messages[1].tool_namespaces();
    assert_eq!(namespaces.len(), 1);
    assert_eq!(namespaces[0].name, "functions");
    let ToolDescription::Function {
        tool_function,
    } = &namespaces[0].tools[0];
    assert_eq!(tool_function.name, "get_time");
}

#[test]
fn tool_call_round_trip_maps_to_chat_blocks() {
    let messages = messages(
        r#"{
            "messages":[
                {"role":"user","content":"what time is it?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"{\"time\":\"17:03\"}"}
            ]
        }"#,
    );

    assert_eq!(messages.len(), 3);
    let tool_calls = messages[1].tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].identifier.as_deref(), Some("call_1"));
    assert_eq!(tool_calls[0].name, "get_time");

    assert_eq!(messages[2].role, ChatRole::Tool {});
    let results = messages[2].tool_call_results();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.as_deref(), Some("call_1"));
    assert_eq!(results[0].1.as_deref(), Some("get_time"), "result name should be backfilled from the matching call");
}

#[test]
fn invalid_tool_call_arguments_stay_serializable() {
    let call = |arguments: &str| {
        to_tool_call(&OaiToolCall {
            index: None,
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: OaiFunctionCall {
                name: "get_time".to_string(),
                arguments: arguments.to_string(),
            },
        })
    };

    assert_eq!(call(r#"{"a":1}"#).arguments.json, r#"{"a":1}"#);
    assert_eq!(call("").arguments.json, "{}");
    assert_eq!(call(r#"{"a":"#).arguments.json, r#""{\"a\":""#);
    for arguments in ["", r#"{"a":"#] {
        serde_json::to_value(&call(arguments).arguments).expect("arguments should stay serializable");
    }
}

#[test]
fn tool_choice_controls_exposed_tools() {
    const TOOLS: &str = r#""tools":[
        {"type":"function","function":{"name":"get_time","description":"Get time"}},
        {"type":"function","function":{"name":"get_weather","description":"Get weather"}}
    ]"#;
    let with_choice =
        |choice: &str| format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{TOOLS},"tool_choice":{choice}}}"#);
    let exposed = |choice: &str| -> Vec<String> {
        messages(&with_choice(choice))
            .iter()
            .flat_map(|message| message.tool_namespaces())
            .flat_map(|namespace| namespace.tools)
            .map(
                |ToolDescription::Function {
                     tool_function,
                 }| tool_function.name,
            )
            .collect()
    };

    assert_eq!(exposed(r#""auto""#), ["get_time", "get_weather"]);
    assert_eq!(exposed(r#""required""#), ["get_time", "get_weather"]);
    assert_eq!(exposed(r#""none""#), Vec::<String>::new());
    assert_eq!(exposed(r#"{"type":"function","function":{"name":"get_weather"}}"#), ["get_weather"]);

    let unknown_function = with_choice(r#"{"type":"function","function":{"name":"missing"}}"#);
    assert!(build_messages(&request(&unknown_function)).is_err(), "undeclared forced function should be rejected");
    let bogus_mode = with_choice(r#""sometimes""#);
    assert!(build_messages(&request(&bogus_mode)).is_err(), "unrecognized tool_choice should be rejected");
}

#[test]
fn json_looking_stream_text_is_withheld_while_tools_are_declared() {
    // Undecided (empty / leading whitespace) or JSON-looking text may still be
    // reclassified into a tool call, so it must not stream.
    assert!(withhold_stream_text(true, ""));
    assert!(withhold_stream_text(true, "  "));
    assert!(withhold_stream_text(true, r#" {"name": "get_time", "parameters"#));

    // Prose can never be rewritten; without tools no rewrite happens at all.
    assert!(!withhold_stream_text(true, "The time is"));
    assert!(!withhold_stream_text(false, r#"{"name": "get_time"}"#));
}

#[test]
fn reply_tool_calls_serialize_in_openai_format() {
    let tool_call = ToolCall {
        identifier: Some("call_1".to_string()),
        name: "get_time".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    };

    let response = serde_json::to_value(oai_tool_call(None, &tool_call)).expect("serializable tool call");
    assert_eq!(
        response,
        serde_json::json!({"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{}"}})
    );

    let delta = serde_json::to_value(oai_tool_call(Some(0), &tool_call)).expect("serializable tool call delta");
    assert_eq!(delta["index"], 0);
}
