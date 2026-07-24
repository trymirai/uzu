use uzu::types::session::chat::ChatMessageList;

use super::*;

fn request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("valid request json")
}

fn chat_messages(json: &str) -> Vec<ChatMessage> {
    let request = request(json);
    to_chat_messages(
        &request.messages,
        parse_reasoning_effort(request.reasoning_effort.as_ref()).expect("valid reasoning_effort"),
    )
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
fn reasoning_effort_is_optional() {
    let messages = chat_messages(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(messages.reasoning_effort(), None);
}

#[test]
fn reasoning_effort_applies_to_latest_message() {
    let messages = chat_messages(
        r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}],"reasoning_effort":"none"}"#,
    );

    assert_eq!(messages.reasoning_effort(), Some(ReasoningEffort::Disabled));
    assert_eq!(messages.first().and_then(ChatMessage::reasoning_effort), None);
    assert_eq!(messages.last().and_then(ChatMessage::reasoning_effort), Some(ReasoningEffort::Disabled));
}

#[test]
fn reasoning_effort_accepts_openai_values_and_uzu_aliases() {
    for (value, expected) in [
        ("none", ReasoningEffort::Disabled),
        ("disabled", ReasoningEffort::Disabled),
        ("default", ReasoningEffort::Default),
        ("low", ReasoningEffort::Low),
        ("medium", ReasoningEffort::Medium),
        ("high", ReasoningEffort::High),
    ] {
        let request = request(&format!(r#"{{"messages":[],"reasoning_effort":"{value}"}}"#));
        assert_eq!(parse_reasoning_effort(request.reasoning_effort.as_ref()), Ok(Some(expected)));
    }
}

#[test]
fn recognized_unsupported_reasoning_effort_is_request_error() {
    for value in ["minimal", "xhigh"] {
        let request = request(&format!(r#"{{"messages":[],"reasoning_effort":"{value}"}}"#));
        let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
            .expect_err("unsupported reasoning_effort should be rejected");
        assert_eq!(error, RequestValidationError::UnsupportedReasoningEffort(value));
        assert_eq!(error.param(), "reasoning_effort");
        assert_eq!(error.code(), "unsupported_reasoning_effort");
    }
}

#[test]
fn invalid_reasoning_effort_is_request_error() {
    let request = request(r#"{"messages":[],"reasoning_effort":"maximum"}"#);
    let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
        .expect_err("invalid reasoning_effort should be rejected");
    assert_eq!(error, RequestValidationError::InvalidReasoningEffort("maximum".to_string()));
    assert_eq!(error.param(), "reasoning_effort");
    assert_eq!(error.code(), "invalid_reasoning_effort");
}

#[test]
fn malformed_reasoning_effort_passes_json_extraction() {
    for body in
        [r#"{"messages":[],"reasoning_effort":123}"#, r#"{"messages":[],"reasoning_effort":{"level":"disabled"}}"#]
    {
        let request = serde_json::from_str::<ChatCompletionRequest>(body)
            .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
        let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
            .expect_err("malformed reasoning_effort should be rejected");
        assert_eq!(error.param(), "reasoning_effort");
        assert_eq!(error.code(), "invalid_reasoning_effort");
    }
}

#[test]
fn reasoning_effort_composes_with_sampling_options() {
    let stochastic =
        reply_config(r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"reasoning_effort":"none"}"#);
    let messages = chat_messages(
        r#"{"messages":[{"role":"user","content":"json please"}],"temperature":0.7,"top_p":0.9,"top_k":40,"reasoning_effort":"none"}"#,
    );
    assert_eq!(messages.reasoning_effort(), Some(ReasoningEffort::Disabled));
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

    let greedy = reply_config(r#"{"messages":[],"temperature":0,"reasoning_effort":"none"}"#);
    assert_eq!(
        greedy.sampling_policy,
        uzu::types::basic::SamplingPolicy::Custom {
            method: SamplingMethod::Greedy {},
        }
    );
}

#[rocket::get("/err")]
fn err_route() -> ChatCompletionResult {
    response_format_error_response(ResponseFormatError::InvalidResponseFormat("bad".to_string()))
}

#[rocket::get("/reasoning-err")]
fn reasoning_err_route() -> ChatCompletionResult {
    reasoning_effort_error_response(RequestValidationError::InvalidReasoningEffort("bad".to_string()))
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
fn reasoning_effort_error_responder_yields_http_400_with_openai_body() {
    let client =
        rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![reasoning_err_route]))
            .expect("rocket client");
    let response = client.get("/reasoning-err").dispatch();

    assert_eq!(response.status(), Status::BadRequest);
    let body: serde_json::Value = response.into_json().expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "reasoning_effort");
    assert_eq!(body["error"]["code"], "invalid_reasoning_effort");
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
