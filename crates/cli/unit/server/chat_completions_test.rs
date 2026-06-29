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

#[test]
fn tools_map_to_function_namespace() {
    let request = request(
        r#"{"messages":[],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object"}}}]}"#,
    );
    let namespaces = to_tool_namespaces(request.tools.as_deref().unwrap_or(&[]));
    assert_eq!(namespaces.len(), 1);
    assert_eq!(namespaces[0].name, "functions");
    assert_eq!(namespaces[0].tools.len(), 1);
    let ToolDescription::Function {
        tool_function,
    } = &namespaces[0].tools[0];
    assert_eq!(tool_function.name, "get_weather");
    assert_eq!(tool_function.description, "Get weather");
    assert!(tool_function.parameters.is_some());
}

#[test]
fn absent_tools_produce_no_namespaces() {
    assert!(to_tool_namespaces(request(r#"{"messages":[]}"#).tools.as_deref().unwrap_or(&[])).is_empty());
}

#[test]
fn tool_calls_map_to_openai_shape() {
    let tool_calls = vec![ToolCall {
        identifier: Some("call_1".to_string()),
        name: "get_weather".to_string(),
        arguments: Value::from(serde_json::json!({ "city": "Paris" })),
    }];
    let mapped = to_oai_tool_calls(&tool_calls).expect("tool calls present");
    assert_eq!(mapped.len(), 1);
    assert_eq!(mapped[0].id, "call_1");
    assert_eq!(mapped[0].kind, "function");
    assert_eq!(mapped[0].function.name, "get_weather");
    assert_eq!(mapped[0].function.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn no_tool_calls_serialize_to_none() {
    assert!(to_oai_tool_calls(&[]).is_none());
}

#[test]
fn tools_with_auto_tool_choice_are_supported() {
    assert!(
        ensure_tools_supported(&request(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#
        ))
        .is_ok()
    );
    assert!(
        ensure_tools_supported(&request(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#
        ))
        .is_ok()
    );
}

#[test]
fn absent_tools_skip_tool_validation() {
    // No tools: a stray tool_choice or stream must not trigger a tool error.
    assert!(ensure_tools_supported(&request(r#"{"messages":[],"tool_choice":"none","stream":true}"#)).is_ok());
}

#[test]
fn streaming_with_tools_is_rejected() {
    assert_eq!(
        ensure_tools_supported(&request(
            r#"{"messages":[],"stream":true,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#
        )),
        Err(ToolRequestError::StreamingUnsupported)
    );
}

#[test]
fn unsupported_tool_choice_is_rejected() {
    for body in [
        r#"{"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"none"}"#,
        r#"{"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"required"}"#,
        r#"{"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"f"}}}"#,
    ] {
        assert_eq!(
            ensure_tools_supported(&request(body)),
            Err(ToolRequestError::UnsupportedToolChoice),
            "expected UnsupportedToolChoice for {body}"
        );
    }
}

#[test]
fn tool_namespaces_attach_to_first_message() {
    let request = request(
        r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#,
    );
    let namespaces = to_tool_namespaces(request.tools.as_deref().unwrap_or(&[]));
    let messages = to_chat_messages(&request.messages, namespaces);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].tool_namespaces().len(), 1);
}
