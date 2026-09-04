use super::*;
use crate::{
    common::thinking::ThinkingSupport,
    server::chat_completions::{ChatCompletionRequest, build_messages},
};

fn request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("valid request json")
}

fn messages(json: &str) -> Vec<ChatMessage> {
    build_messages(&request(json), ThinkingSupport::Unsupported).expect("valid request")
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
fn json_looking_tool_result_content_remains_text() {
    for content in ["26", r#"{"number":1}"#, "[hello]", "[1,2,3]", r#"[{"number":228}]"#, r#"[{"text":"hi"}]"#] {
        let block = tool_call_result_block("call_1", content.to_string());
        let ChatContentBlock::ToolCallResult {
            value,
            ..
        } = block
        else {
            panic!("expected tool call result");
        };

        let serialized = serde_json::to_value(value).expect("tool result should be serializable");
        assert_eq!(serialized.as_str(), Some(content));
    }
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
    assert_eq!(call(r#"{"a":"#).arguments.json, r#"{"__uzu_unparsed_arguments":"{\"a\":"}"#);
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
    assert!(
        build_messages(&request(&unknown_function), ThinkingSupport::Unsupported).is_err(),
        "undeclared forced function should be rejected"
    );
    let bogus_mode = with_choice(r#""sometimes""#);
    assert!(
        build_messages(&request(&bogus_mode), ThinkingSupport::Unsupported).is_err(),
        "unrecognized tool_choice should be rejected"
    );
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

#[test]
fn json_candidate_announces_name_once_it_is_complete() {
    let mut streamer = ToolCallStreamer::new();
    assert!(streamer.update(0, r#"{"name": "wr"#, &ToolParameterTypes::default()).is_empty());
    assert!(streamer.update(0, r#"{"arguments": {}}"#, &ToolParameterTypes::default()).is_empty());

    let deltas = streamer.update(1, r#"{"name": "write", "arguments": {"path":"#, &ToolParameterTypes::default());
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0].index, Some(1));
    assert_eq!(deltas[0].function.name, "write");
    assert_eq!(deltas[0].function.arguments, "");

    // Metadata is emitted exactly once, in the announcement delta.
    assert!(!deltas[0].id.is_empty());
    let call = ToolCall {
        identifier: Some("server-side-id-must-not-leak".to_string()),
        name: "write".to_string(),
        arguments: Value {
            json: r#"{"path":"/tmp/a"}"#.to_string(),
        },
    };

    let finish = serde_json::to_value(streamer.finish(1, &call)).expect("serializable final delta");
    assert_eq!(
        finish,
        serde_json::json!({
            "index": 1,
            "function": {"arguments": r#"{"path":"/tmp/a"}"#}
        })
    );

    assert!(
        streamer.update(1, r#"{"name": "write", "arguments": {"path": "/"#, &ToolParameterTypes::default()).is_empty()
    );
}

#[test]
fn framed_argument_deltas_do_not_repeat_announced_metadata() {
    let mut streamer = ToolCallStreamer::new();
    let deltas = streamer.update(0, "<function=write>\n<parameter=path>\n/tmp/a", &ToolParameterTypes::default());
    assert_eq!(deltas.len(), 2);

    let announcement = serde_json::to_value(&deltas[0]).expect("serializable announcement");
    assert_eq!(announcement["type"], "function");
    assert_eq!(announcement["function"]["name"], "write");

    let arguments = serde_json::to_value(&deltas[1]).expect("serializable argument delta");
    assert_eq!(
        arguments,
        serde_json::json!({
            "index": 0,
            "function": {"arguments": r#"{"path":"/tmp/a"#}
        })
    );
}

#[test]
fn finish_emits_metadata_when_no_announcement_was_possible() {
    let mut streamer = ToolCallStreamer::new();
    let call = ToolCall {
        identifier: None,
        name: "write".to_string(),
        arguments: Value {
            json: r#"{"path":"/tmp/a"}"#.to_string(),
        },
    };

    let finish = serde_json::to_value(streamer.finish(0, &call)).expect("serializable final delta");
    assert_eq!(finish["type"], "function");
    assert_eq!(finish["function"]["name"], "write");
    assert!(!finish["id"].as_str().expect("string id").is_empty());
}

#[test]
fn framed_tool_call_streams_arguments_incrementally() {
    let final_markup = "\n<function=write>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=content>\nhello world\n</parameter>\n</function>\n";
    let final_arguments = serde_json::json!({"path": "/tmp/a.txt", "content": "hello world"});
    let call = ToolCall {
        identifier: Some("c1".to_string()),
        name: "write".to_string(),
        arguments: Value {
            json: final_arguments.to_string(),
        },
    };

    let mut streamer = ToolCallStreamer::new();
    let mut fragments = String::new();
    let mut announcements = 0;
    let chars = final_markup.chars().collect::<Vec<_>>();
    for i in 0..=chars.len() {
        let partial: String = chars[..i].iter().collect();
        for delta in streamer.update(0, &partial, &ToolParameterTypes::default()) {
            if !delta.function.name.is_empty() {
                announcements += 1;
                assert_eq!(delta.function.name, "write");
            }
            fragments.push_str(&delta.function.arguments);
        }
    }
    assert_eq!(announcements, 1);
    fragments.push_str(&streamer.finish(0, &call).function.arguments);

    let parsed: serde_json::Value = serde_json::from_str(&fragments).expect("assembled arguments parse");
    assert_eq!(parsed, final_arguments);
}

#[test]
fn framed_tool_call_withholds_typed_parameter_until_complete() {
    let mut streamer = ToolCallStreamer::new();
    let deltas = streamer.update(0, "\n<function=read>\n<parameter=options>\n{\"a\"", &ToolParameterTypes::default());
    assert!(deltas.iter().all(|delta| delta.function.arguments.is_empty()));

    let deltas = streamer.update(
        0,
        "\n<function=read>\n<parameter=options>\n{\"a\": 1}\n</parameter>\n</function>\n",
        &ToolParameterTypes::default(),
    );
    let fragment: String = deltas.iter().map(|delta| delta.function.arguments.as_str()).collect();
    assert_eq!(fragment, r#"{"options":{"a":1}"#);

    let call = ToolCall {
        identifier: None,
        name: "read".to_string(),
        arguments: Value {
            json: r#"{"options":{"a":1}}"#.to_string(),
        },
    };
    assert_eq!(streamer.finish(0, &call).function.arguments, "}");
}

#[test]
fn framed_tool_call_streams_multibyte_content_without_panicking() {
    let final_markup = "\n<function=write>\n<parameter=content>\nLondon — a city\n</parameter>\n</function>\n";
    let call = ToolCall {
        identifier: None,
        name: "write".to_string(),
        arguments: Value {
            json: serde_json::json!({ "content": "London — a city" }).to_string(),
        },
    };

    let mut streamer = ToolCallStreamer::new();
    let mut fragments = String::new();
    let chars = final_markup.chars().collect::<Vec<_>>();
    for i in 0..=chars.len() {
        let partial: String = chars[..i].iter().collect();
        for delta in streamer.update(0, &partial, &ToolParameterTypes::default()) {
            fragments.push_str(&delta.function.arguments);
        }
    }
    fragments.push_str(&streamer.finish(0, &call).function.arguments);
    let parsed: serde_json::Value = serde_json::from_str(&fragments).expect("assembled arguments parse");
    assert_eq!(parsed, serde_json::json!({ "content": "London — a city" }));
}

#[test]
fn framed_tool_call_with_array_parameter_assembles_exactly() {
    // the edit-call shape: a parameter whose value is a JSON array must never be
    // string-streamed before its type is known, and the finish must not double it
    let final_markup = "\n<function=edit>\n<parameter=path>\nlonodn.md\n</parameter>\n<parameter=edits>\n[{\"oldText\": \"London\", \"newText\": \"Londinium\"}]\n</parameter>\n</function>\n";
    let final_arguments = serde_json::json!({
        "path": "lonodn.md",
        "edits": [{"oldText": "London", "newText": "Londinium"}],
    });
    let call = ToolCall {
        identifier: Some("c1".to_string()),
        name: "edit".to_string(),
        arguments: Value {
            json: final_arguments.to_string(),
        },
    };

    let mut streamer = ToolCallStreamer::new();
    let mut fragments = String::new();
    let mut fragment_count = 0;
    let chars = final_markup.chars().collect::<Vec<_>>();
    for i in 0..=chars.len() {
        let partial: String = chars[..i].iter().collect();
        for delta in streamer.update(0, &partial, &ToolParameterTypes::default()) {
            fragment_count += 1;
            fragments.push_str(&delta.function.arguments);
        }
    }
    fragments.push_str(&streamer.finish(0, &call).function.arguments);

    assert!(fragment_count > 2, "fragments: {fragments:?}");
    let parsed: serde_json::Value = serde_json::from_str(&fragments).expect("assembled arguments parse");
    assert_eq!(parsed, final_arguments);
}

fn parameter_types(tools_json: &str) -> ToolParameterTypes {
    let tools: Vec<OaiTool> = serde_json::from_str(tools_json).expect("valid tools json");
    ToolParameterTypes::from_tools(Some(&tools))
}

const SEARCH_TOOL: &str = r#"[{"type":"function","function":{"name":"search","description":"Search",
    "parameters":{"type":"object","properties":{
        "query":{"type":"string"},
        "limit":{"type":"integer"},
        "ratio":{"type":"number"},
        "metric":{"type":"boolean"},
        "note":{"type":["string","null"]},
        "padded_id":{"type":"string"}
    },"required":["query"]}}}]"#;

#[test]
fn declared_types_restore_scalar_arguments() {
    let types = parameter_types(SEARCH_TOOL);
    let call = ToolCall {
        identifier: None,
        name: "search".to_string(),
        arguments: Value {
            json: r#"{"query":"cats","limit":"5","ratio":"-0.75","metric":"true","note":"null","padded_id":"00123","unknown":"7"}"#.to_string(),
        },
    };

    let coerced: serde_json::Value =
        serde_json::from_str(&coerce_tool_call(&call, &types).arguments.json).expect("coerced arguments parse");
    assert_eq!(
        coerced,
        serde_json::json!({
            "query": "cats",
            "limit": 5,
            "ratio": -0.75,
            "metric": true,
            // a union that includes "string" keeps the string: it is already schema-valid
            "note": "null",
            // declared strings are never retyped, so padded ids survive
            "padded_id": "00123",
            // undeclared parameters are left alone
            "unknown": "7"
        })
    );
}

#[test]
fn python_style_booleans_coerce_to_declared_boolean() {
    let types = parameter_types(SEARCH_TOOL);
    // qwen3.5 writes Python-style booleans into its tool markup
    let call = ToolCall {
        identifier: None,
        name: "search".to_string(),
        arguments: Value {
            json: r#"{"metric":"True","query":"False","limit":"True"}"#.to_string(),
        },
    };

    let coerced: serde_json::Value =
        serde_json::from_str(&coerce_tool_call(&call, &types).arguments.json).expect("coerced arguments parse");
    assert_eq!(
        coerced,
        serde_json::json!({
            "metric": true,
            // declared strings are never retyped
            "query": "False",
            // a boolean does not read as the declared integer
            "limit": "True"
        })
    );
}

#[test]
fn framed_streamer_agrees_with_coerced_boolean_finish() {
    let types = parameter_types(SEARCH_TOOL);
    let final_markup =
        "<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=metric>\nTrue\n</parameter>\n</function>";
    // what coerce_tool_call produces from the parser's Python-style boolean
    let call = ToolCall {
        identifier: Some("c1".to_string()),
        name: "search".to_string(),
        arguments: Value {
            json: r#"{"query":"cats","metric":true}"#.to_string(),
        },
    };

    let mut streamer = ToolCallStreamer::new();
    let mut fragments = String::new();
    let chars = final_markup.chars().collect::<Vec<_>>();
    for i in 0..=chars.len() {
        let partial: String = chars[..i].iter().collect();
        for delta in streamer.update(0, &partial, &types) {
            fragments.push_str(&delta.function.arguments);
        }
    }
    fragments.push_str(&streamer.finish(0, &call).function.arguments);

    let parsed: serde_json::Value = serde_json::from_str(&fragments).expect("assembled arguments parse");
    assert_eq!(parsed, serde_json::json!({"query": "cats", "metric": true}));
}

#[test]
fn declared_string_keeps_json_shaped_text() {
    let types = parameter_types(
        r#"[{"type":"function","function":{"name":"write_file","description":"Write",
            "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
            "required":["path","content"]}}}]"#,
    );
    // the parser typed the JSON-shaped content by its braces
    let call = ToolCall {
        identifier: None,
        name: "write_file".to_string(),
        arguments: Value {
            json: r#"{"path":"data.json","content":{"a":1}}"#.to_string(),
        },
    };

    let coerced: serde_json::Value =
        serde_json::from_str(&coerce_tool_call(&call, &types).arguments.json).expect("coerced arguments parse");
    assert_eq!(coerced["content"], serde_json::json!(r#"{"a":1}"#));
}

#[test]
fn coercion_leaves_unparseable_scalars_and_wrapped_arguments_alone() {
    let types = parameter_types(SEARCH_TOOL);
    let call = |json: &str| ToolCall {
        identifier: None,
        name: "search".to_string(),
        arguments: Value {
            json: json.to_string(),
        },
    };

    // text that does not read as the declared type stays a string
    assert_eq!(coerce_tool_call(&call(r#"{"limit":"lots"}"#), &types).arguments.json, r#"{"limit":"lots"}"#);
    // a declared integer that reads as a float stays a string
    assert_eq!(coerce_tool_call(&call(r#"{"limit":"5.5"}"#), &types).arguments.json, r#"{"limit":"5.5"}"#);
    // wrapped unparseable arguments are not an object of parameters
    let wrapped = r#"{"__uzu_unparsed_arguments":"{\"a\":"}"#;
    assert_eq!(coerce_tool_call(&call(wrapped), &types).arguments.json, wrapped);
}

#[test]
fn framed_streamer_agrees_with_coerced_finish() {
    let types = parameter_types(SEARCH_TOOL);
    let final_markup =
        "<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>";
    // what coerce_tool_call produces from the parser's stringified scalars
    let call = ToolCall {
        identifier: Some("c1".to_string()),
        name: "search".to_string(),
        arguments: Value {
            json: r#"{"query":"cats","limit":5}"#.to_string(),
        },
    };

    let mut streamer = ToolCallStreamer::new();
    let mut fragments = String::new();
    let chars = final_markup.chars().collect::<Vec<_>>();
    for i in 0..=chars.len() {
        let partial: String = chars[..i].iter().collect();
        for delta in streamer.update(0, &partial, &types) {
            fragments.push_str(&delta.function.arguments);
        }
    }
    fragments.push_str(&streamer.finish(0, &call).function.arguments);

    let parsed: serde_json::Value = serde_json::from_str(&fragments).expect("assembled arguments parse");
    assert_eq!(parsed, serde_json::json!({"query": "cats", "limit": 5}));
}
