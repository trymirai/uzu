//! Chat Completions tool-call tests.

use uzu::types::{
    basic::{ToolCall, Value},
    session::chat::{ChatContentBlock, ChatRole},
};

use super::*;
use crate::server::chat_tool_calls::{INCOMPLETE_TOOL_CALL_BATCH_ERROR, has_incomplete_tool_call_batch};

fn request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("valid request json")
}

#[test]
fn tools_and_tool_history_map_to_chat_messages() {
    let request = request(
        r#"{
            "messages":[
                {"role":"system","content":"Be helpful"},
                {"role":"user","content":"Weather?"},
                {"role":"assistant","content":null,"tool_calls":[{
                    "id":"call_1","type":"function",
                    "function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}
                }]},
                {"role":"tool","tool_call_id":"call_1","content":"{\"degrees\":21}"}
            ],
            "tools":[{"type":"function","function":{
                "name":"weather","description":"Get weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
            }}]
        }"#,
    );

    let tools = select_tools(&request).expect("valid tool choice");
    let messages = to_chat_messages(&request.messages, tools);
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0].role, ChatRole::System {});
    assert_eq!(messages[1].role, ChatRole::Developer {});
    assert_eq!(messages[1].tool_namespaces()[0].tools.len(), 1);

    let call = &messages[3].tool_calls()[0];
    assert_eq!(call.identifier.as_deref(), Some("call_1"));
    assert_eq!(call.name, "weather");
    assert_eq!(call.arguments.json, r#"{"city":"Paris"}"#);

    let results = messages[4].tool_call_results();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.as_deref(), Some("call_1"));
    assert_eq!(results[0].1.as_deref(), Some("weather"));
    assert_eq!(results[0].2.json, r#"{"degrees":21}"#);
}

#[test]
fn tools_attach_to_existing_developer_message() {
    let request = request(
        r#"{
            "messages":[
                {"role":"system","content":"System instructions"},
                {"role":"developer","content":"Developer instructions"},
                {"role":"user","content":"Hello"}
            ],
            "tools":[{"type":"function","function":{"name":"lookup"}}]
        }"#,
    );

    let tools = select_tools(&request).expect("valid tool choice");
    let messages = to_chat_messages(&request.messages, tools);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, ChatRole::System {});
    assert_eq!(messages[1].role, ChatRole::Developer {});
    assert_eq!(messages[1].text().as_deref(), Some("Developer instructions"));
    assert_eq!(messages[1].tool_namespaces()[0].tools.len(), 1);
    assert_eq!(messages[2].role, ChatRole::User {});
}

#[test]
fn tool_choice_none_hides_all_tools() {
    let request = request(
        r#"{
            "messages":[{"role":"user","content":"Hello"}],
            "tools":[{"type":"function","function":{"name":"first"}}],
            "tool_choice":"none"
        }"#,
    );

    let tools = select_tools(&request).expect("supported tool choice");
    assert!(tools.is_empty());
    let messages = to_chat_messages(&request.messages, tools);
    assert!(messages.iter().all(|message| message.tool_namespaces().is_empty()));
}

#[test]
fn pinned_tool_choice_is_rejected_until_enforced() {
    let request = request(
        r#"{
            "messages":[{"role":"user","content":"Hello"}],
            "tools":[
                {"type":"function","function":{"name":"first"}},
                {"type":"function","function":{"name":"second"}}
            ],
            "tool_choice":{"type":"function","function":{"name":"second"}}
        }"#,
    );

    assert_eq!(
        select_tools(&request).expect_err("pinned calls cannot be enforced"),
        ToolChoiceError::Unsupported("function second".to_string())
    );
}

#[test]
fn unsupported_and_invalid_tool_choices_are_rejected() {
    let required = request(r#"{"messages":[],"tool_choice":"required"}"#);
    assert_eq!(
        select_tools(&required).expect_err("required cannot be enforced"),
        ToolChoiceError::Unsupported("required".to_string())
    );

    let missing = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"first"}}],
            "tool_choice":{"type":"function","function":{"name":"missing"}}
        }"#,
    );
    assert_eq!(
        select_tools(&missing).expect_err("unknown function should fail"),
        ToolChoiceError::UnknownFunction("missing".to_string())
    );
}

#[test]
fn selected_strict_tools_are_rejected_without_silent_downgrade() {
    let auto_request = request(
        r#"{
            "messages":[],
            "tools":[
                {"type":"function","function":{"name":"regular","strict":false}},
                {"type":"function","function":{"name":"strict","strict":true}}
            ]
        }"#,
    );
    let tools = select_tools(&auto_request).expect("default auto choice");
    assert_eq!(
        validate_selected_tools(&auto_request, &tools).expect_err("strict tool should fail"),
        ToolDefinitionError::StrictUnsupported {
            index: 1,
        }
    );

    let disabled = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"strict","strict":true}}],
            "tool_choice":"none"
        }"#,
    );
    let tools = select_tools(&disabled).expect("supported none choice");
    assert!(validate_selected_tools(&disabled, &tools).is_ok());
}

#[test]
fn disabled_parallel_tool_calls_are_rejected_when_tools_are_enabled() {
    let auto_request = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"lookup"}}],
            "parallel_tool_calls":false
        }"#,
    );
    let tools = select_tools(&auto_request).expect("default auto choice");
    assert_eq!(validate_parallel_tool_calls(&auto_request, &tools, true), Err(ParallelToolCallsError));
    assert!(validate_parallel_tool_calls(&auto_request, &tools, false).is_ok());

    let disabled = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"lookup"}}],
            "tool_choice":"none",
            "parallel_tool_calls":false
        }"#,
    );
    let tools = select_tools(&disabled).expect("supported none choice");
    assert!(validate_parallel_tool_calls(&disabled, &tools, true).is_ok());
}

#[test]
fn enabled_tools_require_model_tool_capability() {
    let auto_request = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"lookup"}}]
        }"#,
    );
    let tools = select_tools(&auto_request).expect("default auto choice");
    assert_eq!(validate_tool_capability(&tools, false), Err(ToolDefinitionError::ToolsUnsupported));
    assert!(validate_tool_capability(&tools, true).is_ok());

    let disabled = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"lookup"}}],
            "tool_choice":"none"
        }"#,
    );
    let tools = select_tools(&disabled).expect("supported none choice");
    assert!(validate_tool_capability(&tools, false).is_ok());
}

#[test]
fn unsupported_tool_kinds_are_rejected() {
    let auto_request = request(
        r#"{
            "messages":[],
            "tools":[{"type":"not_function","function":{"name":"lookup"}}]
        }"#,
    );
    let tools = select_tools(&auto_request).expect("default auto choice");
    assert_eq!(
        validate_selected_tools(&auto_request, &tools).expect_err("unsupported kind should fail"),
        ToolDefinitionError::UnsupportedKind {
            index: 0,
            kind: "not_function".to_string(),
        }
    );

    let disabled = request(
        r#"{
            "messages":[],
            "tools":[{"type":"not_function","function":{"name":"lookup"}}],
            "tool_choice":"none"
        }"#,
    );
    let tools = select_tools(&disabled).expect("supported none choice");
    assert!(matches!(
        validate_selected_tools(&disabled, &tools),
        Err(ToolDefinitionError::UnsupportedKind {
            index: 0,
            ..
        })
    ));
}

#[test]
fn non_streaming_message_serializes_tool_calls() {
    let message = ChatMessage::assistant().with_tool_call(ToolCall {
        identifier: Some("call_7".to_string()),
        name: "lookup".to_string(),
        arguments: Value {
            json: r#"{"query":"uzu"}"#.to_string(),
        },
    });
    let response = OaiMessage {
        role: "assistant".to_string(),
        content: message.text(),
        tool_calls: response_tool_calls(&message),
        tool_call_id: None,
    };

    let json = serde_json::to_value(response).expect("serializable response");
    assert!(json["content"].is_null());
    assert_eq!(json["tool_calls"][0]["id"], "call_7");
    assert_eq!(json["tool_calls"][0]["type"], "function");
    assert_eq!(json["tool_calls"][0]["function"]["name"], "lookup");
    assert_eq!(json["tool_calls"][0]["function"]["arguments"], r#"{"query":"uzu"}"#);
}

#[test]
fn streaming_tool_calls_are_emitted_once_with_indexes() {
    let calls = vec![
        ToolCall {
            identifier: Some("call_1".to_string()),
            name: "first".to_string(),
            arguments: Value {
                json: "{}".to_string(),
            },
        },
        ToolCall {
            identifier: Some("call_2".to_string()),
            name: "second".to_string(),
            arguments: Value {
                json: r#"{"value":2}"#.to_string(),
            },
        },
    ];
    let mut emitted = 0;
    let deltas = stream_tool_call_deltas(&calls, &mut emitted);
    let json = serde_json::to_value(deltas).expect("serializable deltas");

    assert_eq!(emitted, 2);
    assert_eq!(json[0]["index"], 0);
    assert_eq!(json[1]["index"], 1);
    assert_eq!(json[1]["function"]["arguments"], r#"{"value":2}"#);
    assert!(stream_tool_call_deltas(&calls, &mut emitted).is_empty());
}

#[test]
fn incomplete_tool_call_candidates_invalidate_the_entire_batch() {
    let valid_call = ToolCall {
        identifier: Some("call_1".to_string()),
        name: "first".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    };
    let valid = ChatMessage::assistant().with_tool_call(valid_call);
    assert!(!has_incomplete_tool_call_batch(&valid, Some(&ChatReplyFinishReason::ToolCalls)));

    let partial = valid.with_tool_call_candidate(Value::from(serde_json::json!({"name": "broken"})));
    assert!(has_incomplete_tool_call_batch(&partial, Some(&ChatReplyFinishReason::ToolCalls)));

    let interleaved = partial.with_tool_call(ToolCall {
        identifier: Some("call_2".to_string()),
        name: "second".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    });
    assert!(has_incomplete_tool_call_batch(&interleaved, Some(&ChatReplyFinishReason::ToolCalls)));
    assert!(
        normalize_tool_calls_for_capability(&interleaved, false)
            .content
            .iter()
            .all(|block| !matches!(block, ChatContentBlock::ToolCallCandidate { .. }))
    );
    assert!(has_incomplete_tool_call_batch(&ChatMessage::assistant(), Some(&ChatReplyFinishReason::ToolCalls)));
}

#[test]
fn generated_calls_must_match_selected_tools() {
    let message = ChatMessage::assistant().with_tool_call(ToolCall {
        identifier: Some("call_1".to_string()),
        name: "lookup".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    });
    assert!(
        tool_call_batch_error(&message, Some(&ChatReplyFinishReason::ToolCalls), &["lookup".to_string()], true)
            .is_none()
    );
    assert!(
        tool_call_batch_error(&message, Some(&ChatReplyFinishReason::ToolCalls), &[], true)
            .is_some_and(|error| error.contains("undeclared tool"))
    );
    assert!(
        tool_call_batch_error(&message, Some(&ChatReplyFinishReason::Stop), &["lookup".to_string()], true)
            .is_some_and(|error| error == INCOMPLETE_TOOL_CALL_BATCH_ERROR)
    );
    assert!(tool_call_batch_error(&message, None, &["lookup".to_string()], false).is_none());
    assert!(
        tool_call_batch_error(&message, None, &["lookup".to_string()], true)
            .is_some_and(|error| error == INCOMPLETE_TOOL_CALL_BATCH_ERROR)
    );

    let missing_id = ChatMessage::assistant().with_tool_call(ToolCall {
        identifier: None,
        name: "lookup".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    });
    assert!(
        tool_call_batch_error(&missing_id, Some(&ChatReplyFinishReason::ToolCalls), &["lookup".to_string()], true)
            .is_some_and(|error| error.contains("without an identifier"))
    );

    let invalid_arguments = ChatMessage::assistant().with_tool_call(ToolCall {
        identifier: Some("call_2".to_string()),
        name: "lookup".to_string(),
        arguments: Value {
            json: "not-json".to_string(),
        },
    });
    assert!(
        tool_call_batch_error(
            &invalid_arguments,
            Some(&ChatReplyFinishReason::ToolCalls),
            &["lookup".to_string()],
            true
        )
        .is_some_and(|error| error.contains("invalid JSON"))
    );

    let duplicate_id = message.with_tool_call(ToolCall {
        identifier: Some("call_1".to_string()),
        name: "lookup".to_string(),
        arguments: Value {
            json: "{}".to_string(),
        },
    });
    assert!(
        tool_call_batch_error(&duplicate_id, Some(&ChatReplyFinishReason::ToolCalls), &["lookup".to_string()], true)
            .is_some_and(|error| error.contains("duplicate"))
    );
}

#[test]
fn malformed_typed_request_fields_use_request_error_path() {
    for body in [
        r#"{"messages":[],"parallel_tool_calls":"false"}"#,
        r#"{"messages":[],"tools":[{"type":"function","function":{"name":"lookup","strict":"true"}}]}"#,
        r#"{"messages":[],"tools":[{"type":"function"}]}"#,
    ] {
        let value: serde_json::Value = serde_json::from_str(body).expect("valid JSON");
        assert!(parse_request_body(value).is_err(), "body: {body}");
    }
}

#[test]
fn tool_history_must_match_model_capabilities() {
    let history_request = request(
        r#"{
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":{"name":"first","arguments":"{}"}},
                    {"id":"call_2","type":"function","function":{"name":"second","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"call_1","content":"one"},
                {"role":"tool","tool_call_id":"call_2","content":"two"}
            ]
        }"#,
    );
    let error = validate_tool_history(&history_request.messages, false, false).expect_err("unsupported history");
    assert_eq!(error.param(), "messages[0].tool_calls");
    assert_eq!(error.code(), "unsupported_tool_history");
    let error = validate_tool_history(&history_request.messages, true, false).expect_err("parallel history");
    assert_eq!(error.param(), "messages[0].tool_calls");
    assert_eq!(error.code(), "unsupported_parallel_tool_history");
    assert!(validate_tool_history(&history_request.messages, true, true).is_ok());

    let tool_result = request(r#"{"messages":[{"role":"tool","tool_call_id":"call_1","content":"ok"}]}"#);
    let error = validate_tool_history(&tool_result.messages, true, true).expect_err("orphan tool result");
    assert_eq!(error.param(), "messages[0].tool_call_id");
    assert_eq!(error.code(), "invalid_tool_history");
}

#[test]
fn malformed_tool_history_is_rejected_before_conversion() {
    for (body, expected_param) in [
        (
            r#"{"messages":[{"role":"user","content":"hi","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}"#,
            "messages[0].tool_calls",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"other","function":{"name":"lookup","arguments":"{}"}}]}]}"#,
            "messages[0].tool_calls[0].type",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"not-json"}}]}]}"#,
            "messages[0].tool_calls[0].function.arguments",
        ),
        (r#"{"messages":[{"role":"tool","content":"ok"}]}"#, "messages[0].tool_call_id"),
        (r#"{"messages":[{"role":"bogus","content":"hi"}]}"#, "messages[0].role"),
        (r#"{"messages":[{"role":"system","content":null}]}"#, "messages[0].content"),
        (r#"{"messages":[{"role":"assistant","content":null}]}"#, "messages[0].content"),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":null}]}"#,
            "messages[1].content",
        ),
    ] {
        let history_request = request(body);
        let error = validate_tool_history(&history_request.messages, true, true).expect_err("invalid history");
        assert_eq!(error.param(), expected_param, "body: {body}");
    }
}

#[test]
fn single_call_capability_exposes_only_the_call_retained_in_history() {
    let message = ChatMessage::assistant()
        .with_tool_call(ToolCall {
            identifier: Some("call_1".to_string()),
            name: "first".to_string(),
            arguments: Value {
                json: "{}".to_string(),
            },
        })
        .with_tool_call(ToolCall {
            identifier: Some("call_2".to_string()),
            name: "second".to_string(),
            arguments: Value {
                json: "{}".to_string(),
            },
        });

    let normalized = normalize_tool_calls_for_capability(&message, false);
    let calls = normalized.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].identifier.as_deref(), Some("call_1"));
    assert_eq!(normalize_tool_calls_for_capability(&message, true).tool_calls().len(), 2);
}

#[rocket::get("/tool-choice-err")]
fn tool_choice_err_route() -> ChatCompletionResult {
    tool_choice_error_response(ToolChoiceError::Unsupported("required".to_string()))
}

#[rocket::get("/strict-tool-err")]
fn strict_tool_err_route() -> ChatCompletionResult {
    tool_definition_error_response(ToolDefinitionError::StrictUnsupported {
        index: 2,
    })
}

#[rocket::get("/tool-kind-err")]
fn tool_kind_err_route() -> ChatCompletionResult {
    tool_definition_error_response(ToolDefinitionError::UnsupportedKind {
        index: 1,
        kind: "not_function".to_string(),
    })
}

#[rocket::get("/unsupported-tools-err")]
fn unsupported_tools_err_route() -> ChatCompletionResult {
    tool_definition_error_response(ToolDefinitionError::ToolsUnsupported)
}

#[rocket::get("/parallel-tool-calls-err")]
fn parallel_tool_calls_err_route() -> ChatCompletionResult {
    parallel_tool_calls_error_response(ParallelToolCallsError)
}

#[test]
fn unsupported_tool_choice_yields_http_400_with_openai_body() {
    let client =
        rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![tool_choice_err_route]))
            .expect("rocket client");
    let response = client.get("/tool-choice-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "tool_choice");
    assert_eq!(body["error"]["code"], "unsupported_tool_choice");
}

#[test]
fn strict_tool_yields_http_400_with_precise_parameter() {
    let client =
        rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![strict_tool_err_route]))
            .expect("rocket client");
    let response = client.get("/strict-tool-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "tools[2].function.strict");
    assert_eq!(body["error"]["code"], "unsupported_strict_tool");
}

#[test]
fn unsupported_tool_kind_yields_http_400() {
    let client =
        rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![tool_kind_err_route]))
            .expect("rocket client");
    let response = client.get("/tool-kind-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "tools[1].type");
    assert_eq!(body["error"]["code"], "unsupported_tool_type");
}

#[test]
fn unsupported_model_tool_capability_yields_http_400() {
    let client = rocket::local::blocking::Client::tracked(
        rocket::build().mount("/", rocket::routes![unsupported_tools_err_route]),
    )
    .expect("rocket client");
    let response = client.get("/unsupported-tools-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "tools");
    assert_eq!(body["error"]["code"], "unsupported_tools");
}

#[test]
fn disabled_parallel_tool_calls_yield_http_400() {
    let client = rocket::local::blocking::Client::tracked(
        rocket::build().mount("/", rocket::routes![parallel_tool_calls_err_route]),
    )
    .expect("rocket client");
    let response = client.get("/parallel-tool-calls-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "parallel_tool_calls");
    assert_eq!(body["error"]["code"], "unsupported_parallel_tool_calls");
}
