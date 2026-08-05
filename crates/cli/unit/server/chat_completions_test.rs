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
        ToolDefinitionError {
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
    assert_eq!(validate_parallel_tool_calls(&auto_request, &tools), Err(ParallelToolCallsError));

    let disabled = request(
        r#"{
            "messages":[],
            "tools":[{"type":"function","function":{"name":"lookup"}}],
            "tool_choice":"none",
            "parallel_tool_calls":false
        }"#,
    );
    let tools = select_tools(&disabled).expect("supported none choice");
    assert!(validate_parallel_tool_calls(&disabled, &tools).is_ok());
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
    assert!(has_incomplete_tool_call_batch(&ChatMessage::assistant(), Some(&ChatReplyFinishReason::ToolCalls)));
}

#[rocket::get("/err")]
fn err_route() -> ChatCompletionResult {
    request_error_response(ResponseFormatError::InvalidResponseFormat("bad".to_string()))
}

#[rocket::get("/tool-choice-err")]
fn tool_choice_err_route() -> ChatCompletionResult {
    tool_choice_error_response(ToolChoiceError::Unsupported("required".to_string()))
}

#[rocket::get("/strict-tool-err")]
fn strict_tool_err_route() -> ChatCompletionResult {
    tool_definition_error_response(ToolDefinitionError {
        index: 2,
    })
}

#[rocket::get("/parallel-tool-calls-err")]
fn parallel_tool_calls_err_route() -> ChatCompletionResult {
    parallel_tool_calls_error_response(ParallelToolCallsError)
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
