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
    let error = ResponseFormatError::GrammarUnsupported;
    match invalid_request_response("response_format", error.code(), error.message()) {
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

fn messages_with_support(
    json: &str,
    thinking_support: ThinkingSupport,
) -> Vec<ChatMessage> {
    build_messages(&request(json), thinking_support).expect("valid request")
}

#[test]
fn reasoning_effort_prepends_system_message_for_levels_model() {
    use uzu::types::session::chat::ChatMessageList;

    let messages = messages_with_support(
        r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}"#,
        ThinkingSupport::Levels(ReasoningEffort::Default),
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(messages.reasoning_effort(), Some(ReasoningEffort::High));
}

#[test]
fn reasoning_effort_merges_into_leading_system_message() {
    let messages = messages_with_support(
        r#"{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"reasoning_effort":"high"}"#,
        ThinkingSupport::Levels(ReasoningEffort::Default),
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, ChatRole::System {});
    assert_eq!(messages[0].reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(messages[0].text().as_deref(), Some("be brief"));
}

#[test]
fn reasoning_effort_default_emits_no_system_message() {
    for body in [
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"default"}"#,
    ] {
        let messages = messages_with_support(body, ThinkingSupport::Levels(ReasoningEffort::Default));
        assert_eq!(messages.len(), 1);
        assert!(messages.iter().all(|message| message.reasoning_effort().is_none()));
    }
}

#[test]
fn reasoning_effort_maps_levels_to_on_off_for_toggle_model() {
    let enabled = messages_with_support(r#"{"messages":[],"reasoning_effort":"low"}"#, ThinkingSupport::Toggle(true));
    assert_eq!(enabled[0].reasoning_effort(), Some(ReasoningEffort::Default));

    let disabled =
        messages_with_support(r#"{"messages":[],"reasoning_effort":"disabled"}"#, ThinkingSupport::Toggle(true));
    assert_eq!(disabled[0].reasoning_effort(), Some(ReasoningEffort::Disabled));
}

#[test]
fn reasoning_effort_satisfied_without_system_message_when_model_behavior_already_matches() {
    for (support, body) in [
        (ThinkingSupport::AlwaysOn, r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}"#),
        (
            ThinkingSupport::Unsupported,
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"disabled"}"#,
        ),
    ] {
        let messages = messages_with_support(body, support);
        assert_eq!(messages.len(), 1);
        assert!(messages.iter().all(|message| message.reasoning_effort().is_none()));
    }
}

#[test]
fn reasoning_effort_rejected_when_model_cannot_produce_it() {
    for (support, body) in [
        (ThinkingSupport::AlwaysOn, r#"{"messages":[],"reasoning_effort":"disabled"}"#),
        (ThinkingSupport::Unsupported, r#"{"messages":[],"reasoning_effort":"high"}"#),
    ] {
        let error =
            build_messages(&request(body), support).expect_err("unfulfillable reasoning_effort should be rejected");
        assert!(
            matches!(error, MessageBuildError::ReasoningEffort(_)),
            "expected ReasoningEffort error, got {error:?}"
        );
    }
}

#[test]
fn enable_thinking_honored_for_toggle_model() {
    let enabled = messages_with_support(r#"{"messages":[],"enable_thinking":true}"#, ThinkingSupport::Toggle(false));
    assert_eq!(enabled[0].reasoning_effort(), Some(ReasoningEffort::Default));

    let disabled = messages_with_support(
        r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":false}}"#,
        ThinkingSupport::Toggle(true),
    );
    assert_eq!(disabled[0].reasoning_effort(), Some(ReasoningEffort::Disabled));
}

#[test]
fn enable_thinking_honored_for_levels_model() {
    let disabled = messages_with_support(
        r#"{"messages":[],"enable_thinking":false}"#,
        ThinkingSupport::Levels(ReasoningEffort::Default),
    );
    assert_eq!(disabled[0].reasoning_effort(), Some(ReasoningEffort::Disabled));
}

#[test]
fn enable_thinking_rejected_when_model_cannot_produce_it() {
    let error = build_messages(&request(r#"{"messages":[],"enable_thinking":false}"#), ThinkingSupport::AlwaysOn)
        .expect_err("disabling an always-on model should be rejected");
    assert!(matches!(error, MessageBuildError::EnableThinking(_)), "expected EnableThinking error, got {error:?}");

    let error = build_messages(&request(r#"{"messages":[],"enable_thinking":true}"#), ThinkingSupport::Unsupported)
        .expect_err("enabling reasoning on an unsupported model should be rejected");
    assert!(matches!(error, MessageBuildError::EnableThinking(_)), "expected EnableThinking error, got {error:?}");
}

#[test]
fn enable_thinking_rejects_contradictions() {
    for body in [
        r#"{"messages":[],"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":false}}"#,
        r#"{"messages":[],"enable_thinking":false,"reasoning_effort":"high"}"#,
        r#"{"messages":[],"enable_thinking":true,"reasoning_effort":"disabled"}"#,
    ] {
        build_messages(&request(body), ThinkingSupport::Levels(ReasoningEffort::Default))
            .expect_err("contradictory thinking requests should be rejected");
    }
}

#[test]
fn enable_thinking_agrees_with_compatible_reasoning_effort() {
    let messages = messages_with_support(
        r#"{"messages":[],"enable_thinking":true,"reasoning_effort":"low"}"#,
        ThinkingSupport::Levels(ReasoningEffort::Default),
    );
    assert_eq!(messages[0].reasoning_effort(), Some(ReasoningEffort::Low));
}

#[test]
fn enable_thinking_rejects_malformed_values() {
    for body in [
        r#"{"messages":[],"enable_thinking":"yes"}"#,
        r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":1}}"#,
        r#"{"messages":[],"chat_template_kwargs":"nope"}"#,
    ] {
        build_messages(&request(body), ThinkingSupport::Toggle(true))
            .expect_err("malformed enable_thinking should be rejected");
    }
}

#[test]
fn chat_template_kwargs_ignores_unrelated_keys_and_null() {
    for body in [
        r#"{"messages":[],"chat_template_kwargs":{"some_other_kwarg":42}}"#,
        r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":null}}"#,
        r#"{"messages":[],"enable_thinking":null}"#,
    ] {
        let messages = messages_with_support(body, ThinkingSupport::Toggle(true));
        assert!(messages.iter().all(|message| message.reasoning_effort().is_none()));
    }
}

#[test]
fn malformed_enable_thinking_passes_json_extraction() {
    for body in [r#"{"messages":[],"enable_thinking":"yes"}"#, r#"{"messages":[],"chat_template_kwargs":"nope"}"#] {
        serde_json::from_str::<ChatCompletionRequest>(body)
            .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
    }
}

#[test]
fn reasoning_effort_rejects_unrecognized_values() {
    for body in [r#"{"messages":[],"reasoning_effort":"totally-bogus"}"#, r#"{"messages":[],"reasoning_effort":42}"#] {
        let error = build_messages(&request(body), ThinkingSupport::default())
            .expect_err("bad reasoning_effort should be rejected");
        assert!(
            matches!(error, MessageBuildError::ReasoningEffort(_)),
            "expected ReasoningEffort error, got {error:?}"
        );
        assert_eq!(error.param(), "reasoning_effort");
        assert_eq!(error.code(), "invalid_reasoning_effort");
    }
}

#[test]
fn malformed_reasoning_effort_passes_json_extraction() {
    for body in [r#"{"messages":[],"reasoning_effort":"totally-bogus"}"#, r#"{"messages":[],"reasoning_effort":42}"#] {
        serde_json::from_str::<ChatCompletionRequest>(body)
            .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
    }
}

#[rocket::get("/err")]
fn err_route() -> ChatCompletionResult {
    let error = ResponseFormatError::InvalidResponseFormat("bad".to_string());
    invalid_request_response("response_format", error.code(), error.message())
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
fn message_content_accepts_string_null_and_text_parts() {
    let messages = request(
        r#"{"messages":[
            {"role":"user","content":"plain"},
            {"role":"assistant","content":null},
            {"role":"user","content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}]}
        ]}"#,
    )
    .messages;
    assert_eq!(messages[0].content.as_deref(), Some("plain"));
    assert_eq!(messages[1].content, None);
    assert_eq!(messages[2].content.as_deref(), Some("Hello world"));

    let missing = request(r#"{"messages":[{"role":"user"}]}"#);
    assert_eq!(missing.messages[0].content, None);

    let image = serde_json::from_str::<ChatCompletionRequest>(
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}"#,
    );
    assert!(image.is_err());
}

#[test]
fn assistant_reasoning_content_round_trips_in_session_block_order() {
    let messages = messages_with_support(
        r#"{"messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","reasoning_content":"thoughts","content":"answer","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"assistant","reasoning_content":"","content":"no thinking"},
            {"role":"user","content":"again"}
        ]}"#,
        ThinkingSupport::Levels(ReasoningEffort::Default),
    );
    let assistant = &messages[1];
    let kinds = assistant
        .content
        .iter()
        .map(|block| match block {
            ChatContentBlock::Reasoning {
                ..
            } => "reasoning",
            ChatContentBlock::Text {
                ..
            } => "text",
            ChatContentBlock::ToolCall {
                ..
            } => "tool_call",
            _ => "other",
        })
        .collect::<Vec<_>>();
    assert_eq!(kinds, ["reasoning", "text", "tool_call"]);
    assert_eq!(assistant.reasoning().as_deref(), Some("thoughts"));

    // empty reasoning_content must not create a block, matching how replies are stored
    assert_eq!(messages[2].reasoning(), None);
    assert_eq!(messages[2].text().as_deref(), Some("no thinking"));
}

#[test]
fn prefix_match_ignores_tool_call_identifiers() {
    let assistant_call = |identifier: &str| {
        ChatMessage::assistant().with_tool_call(uzu::types::basic::ToolCall {
            identifier: Some(identifier.to_string()),
            name: "write".to_string(),
            arguments: uzu::types::basic::Value {
                json: r#"{"path":"/tmp/a"}"#.to_string(),
            },
        })
    };

    let current = vec![ChatMessage::user().with_text("hi".to_string()), assistant_call("nagare-uuid")];
    let extending = vec![
        ChatMessage::user().with_text("hi".to_string()),
        assistant_call("server-uuid"),
        ChatMessage::user().with_text("again".to_string()),
    ];
    assert!(messages_have_prefix(&extending, &current));

    let mut different = assistant_call("nagare-uuid");
    different.content.push(ChatContentBlock::Text {
        value: "extra".to_string(),
    });
    let not_matching = vec![ChatMessage::user().with_text("hi".to_string()), different];
    assert!(!messages_have_prefix(&not_matching, &current));
}
