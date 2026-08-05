use super::*;

fn request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("valid request json")
}

fn reply_config(json: &str) -> ChatReplyConfig {
    build_reply_config(&request(json)).expect("valid reply config")
}

#[cfg(not(feature = "capability-grammar"))]
fn reply_config_error(json: &str) -> ResponseFormatError {
    build_reply_config(&request(json)).expect_err("invalid reply config")
}

#[test]
fn response_format_maps_to_grammar() {
    assert!(reply_config(r#"{"messages":[]}"#).grammar.is_none());
    assert!(reply_config(r#"{"messages":[],"response_format":{"type":"text"}}"#).grammar.is_none());

    #[cfg(feature = "capability-grammar")]
    assert_eq!(
        reply_config(r#"{"messages":[],"response_format":{"type":"json_object"}}"#).grammar,
        Some(Grammar::JsonAny {})
    );
}

#[test]
fn response_format_rejects_grammar_without_capability() {
    #[cfg(not(feature = "capability-grammar"))]
    {
        assert_eq!(
            reply_config_error(r#"{"messages":[],"response_format":{"type":"json_object"}}"#),
            ResponseFormatError::GrammarUnsupported
        );
        assert_eq!(
            reply_config_error(
                r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}"#
            ),
            ResponseFormatError::GrammarUnsupported
        );
    }
}

#[test]
fn response_format_unrecognized_is_invalid() {
    let error = build_reply_config(&request(r#"{"messages":[],"response_format":{"type":"totally-bogus"}}"#))
        .expect_err("unrecognized response_format should be rejected");
    assert!(
        matches!(error, ResponseFormatError::InvalidResponseFormat(_)),
        "expected InvalidResponseFormat, got {error:?}"
    );
}

#[cfg(feature = "capability-grammar")]
#[test]
fn response_format_json_schema_maps_to_grammar() {
    let config = reply_config(
        r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object"}}}}"#,
    );
    assert_eq!(
        config.grammar,
        Some(Grammar::JsonSchema {
            schema: r#"{"type":"object"}"#.to_string(),
        })
    );
}

#[test]
fn response_format_json_schema_rejects_invalid_schema() {
    let error = build_reply_config(&request(
        r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"not-a-json-schema-type"}}}}"#,
    ))
    .expect_err("an invalid JSON Schema should be rejected");
    assert!(matches!(error, ResponseFormatError::InvalidJsonSchema(_)), "expected InvalidJsonSchema, got {error:?}");
    assert_eq!(error.code(), "invalid_json_schema");
}

#[test]
fn response_format_validation_errors_are_request_errors() {
    match request_error_response(ResponseFormatError::GrammarUnsupported) {
        ChatCompletionResult::Error(_) => {},
        ChatCompletionResult::Json(_) | ChatCompletionResult::Stream(_) => {
            panic!("response_format validation errors should be request errors")
        },
    }
}

#[test]
fn malformed_response_format_passes_json_extraction() {
    for body in [
        r#"{"messages":[],"response_format":{"type":"totally-bogus"}}"#,
        r#"{"messages":[],"response_format":"not-even-an-object"}"#,
    ] {
        serde_json::from_str::<ChatCompletionRequest>(body)
            .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
    }
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

    let messages = to_chat_messages(&request.messages, request.tools.as_deref());
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

#[rocket::get("/err")]
fn err_route() -> ChatCompletionResult {
    request_error_response(ResponseFormatError::InvalidResponseFormat("bad".to_string()))
}

#[test]
fn error_responder_yields_http_400_with_openai_body() {
    let client = rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![err_route]))
        .expect("rocket client");
    let response = client.get("/err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format");
    assert_eq!(body["error"]["code"], "invalid_response_format");
    assert!(
        body["error"]["message"].as_str().is_some_and(|message| !message.is_empty()),
        "expected a non-empty error message, got {body}"
    );
}

#[cfg(feature = "capability-grammar")]
#[test]
fn response_format_composes_with_sampling_options() {
    let stochastic = reply_config(
        r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"response_format":{"type":"json_object"}}"#,
    );
    assert_eq!(stochastic.grammar, Some(Grammar::JsonAny {}));
    assert_eq!(
        stochastic.sampling_policy,
        uzu::types::basic::SamplingPolicy::Custom {
            method: SamplingMethod::Stochastic {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: Some(0.9),
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            },
        }
    );

    let greedy = reply_config(r#"{"messages":[],"temperature":0,"response_format":{"type":"json_object"}}"#);
    assert_eq!(greedy.grammar, Some(Grammar::JsonAny {}));
    assert_eq!(
        greedy.sampling_policy,
        uzu::types::basic::SamplingPolicy::Custom {
            method: SamplingMethod::Greedy {},
        }
    );
}
