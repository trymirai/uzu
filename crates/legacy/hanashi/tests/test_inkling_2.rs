mod helpers;

use std::sync::Arc;

use hanashi::{
    Encoding as _,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig, renderer::Renderer},
};
use helpers::{load_tokenizer, tokenizer_directory};
use shoji::types::{
    basic::{ReasoningEffort, ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

const MODEL_NAME: &str = "thinkingmachines_Inkling-2-Mini";

const TOOLS_DECLARATION: &str = concat!(
    "<|message_system|>tool_declare<|content_xml|>",
    r#"[{"description":"Get the current weather","name":"get_weather","parameters":{"properties":{"#,
    r#""city":{"description":"City name","type":"string"},"units":{"enum":["metric","imperial"],"type":"string"}},"#,
    r#""required":["city"],"type":"object"},"type":"function"}]"#,
    "<|end_message|>",
);

/// The messages of `conversation()` as the reference template renders them, after `TOOLS_DECLARATION`
/// (the reference template has no generation prompt).
const CONVERSATION_MESSAGES: &str = concat!(
    "<|message_system|><|content_text|>You are a helpful assistant.<|end_message|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
    "<|message_user|><|content_text|>What is the weather in Paris?<|end_message|>",
    "<|message_model|><|content_thinking|>The user wants the weather, so I call the tool.<|end_message|>",
    "<|message_model|>get_weather<|content_invoke_tool_json|>",
    r#"{"name":"get_weather","args":{"city":"Paris","options":{"hourly":false,"days":2},"units":"metric"}}"#,
    "<|end_message|><|content_model_end_sampling|>",
    "<|message_tool|>get_weather<|content_text|>Sunny, 21 °C<|end_message|>",
    "<|message_model|><|content_thinking|>The tool answered.<|end_message|>",
    "<|message_model|><|content_text|>It is sunny and 21 °C in Paris.<|end_message|><|content_model_end_sampling|>",
    "<|message_user|><|content_text|>Thanks!<|end_message|>",
);

const GENERATION_PROMPT: &str = "<|message_model|>";

fn renderer() -> Renderer {
    Renderer::new(HanashiConfig::Inkling2.resolve().unwrap().rendering)
}

fn render(messages: &[ChatMessage]) -> String {
    renderer().render(messages, true, None, None, None).unwrap()
}

/// A byte tokenizer with the model's special tokens exercises rendering and streaming without the model's assets.
fn encoding() -> HanashiEncodingImpl {
    let config = HanashiConfig::Inkling2.resolve().unwrap();
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: [(String, u32); 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value.to_string(), index as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let model = BPE::builder().vocab_and_merges(vocabulary, Vec::new()).build().unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    // Special tokens that frame nothing (here the padding token) must stay invisible to the parser.
    let mut special_tokens = config.parsing.framing_config().tokens;
    special_tokens.push("<|endoftext|>".to_string());
    tokenizer
        .add_special_tokens(&special_tokens.into_iter().map(|value| AddedToken::from(value, true)).collect::<Vec<_>>());
    HanashiEncodingImpl::new(
        HanashiConfig::Custom {
            config,
        },
        Arc::new(tokenizer),
    )
    .unwrap()
}

/// Feeds the completion one token at a time and returns the reply after every token.
fn stream(
    encoding: &mut HanashiEncodingImpl,
    completion: &str,
) -> Vec<ChatMessage> {
    let history_len = encoding.state().messages.len();
    let mut replies = Vec::new();
    for token in encoding.tokenize(completion).unwrap() {
        encoding.decode(vec![token]).unwrap();
        assert_eq!(encoding.state().messages.len(), history_len, "the reply must stay a single message");
        let reply = encoding.state().messages.last().unwrap().clone();
        assert_eq!(reply.role, ChatRole::Assistant {});
        replies.push(reply);
    }
    replies
}

fn candidates(message: &ChatMessage) -> usize {
    message.content.iter().filter(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. })).count()
}

fn weather_tools() -> ChatMessage {
    ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
        name: "functions".to_string(),
        description: None,
        tools: vec![ToolDescription::Function {
            tool_function: ToolFunction {
                name: "get_weather".to_string(),
                description: "Get the current weather".to_string(),
                // Unsorted on purpose: tools are declared with sorted keys.
                parameters: Some(Value::from(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["metric", "imperial"]}
                    },
                    "required": ["city"]
                }))),
                return_definition: None,
            },
        }],
    }])
}

fn weather_call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        identifier: Some("call_1".to_string()),
        name: "get_weather".to_string(),
        arguments: Value::from(arguments),
    }
}

fn weather_result(value: serde_json::Value) -> ChatMessage {
    ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier: Some("call_1".to_string()),
        name: Some("get_weather".to_string()),
        value: Value::from(value),
    })
}

fn conversation() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system().with_text("You are a helpful assistant.".to_string()),
        weather_tools(),
        ChatMessage::user().with_text("What is the weather in Paris?".to_string()),
        ChatMessage::assistant()
            .with_reasoning("The user wants the weather, so I call the tool.".to_string())
            // Top-level arguments are sorted, nested values keep their order.
            .with_tool_call(weather_call(serde_json::json!({
                "units": "metric",
                "city": "Paris",
                "options": {"hourly": false, "days": 2}
            }))),
        weather_result(serde_json::json!("Sunny, 21 °C")),
        ChatMessage::assistant()
            .with_reasoning("The tool answered.".to_string())
            .with_text("It is sunny and 21 °C in Paris.".to_string()),
        ChatMessage::user().with_text("Thanks!".to_string()),
    ]
}

#[test]
fn rendering_matches_the_reference_template() {
    let renderer = renderer();
    let messages = conversation();
    let reference = format!("{TOOLS_DECLARATION}{CONVERSATION_MESSAGES}");

    // Without the generation prompt the prompt is byte-identical to the reference template.
    assert_eq!(renderer.render(&messages, false, None, None, None).unwrap(), reference);
    // The model starts every reply with <|message_model|>, so the generation prompt opens the reply with it.
    assert_eq!(renderer.render(&messages, true, None, None, None).unwrap(), format!("{reference}{GENERATION_PROMPT}"));
}

#[test]
fn rendering_of_a_text_chat() {
    let messages = vec![
        ChatMessage::system().with_text("Be brief.".to_string()),
        ChatMessage::user().with_text("Hello!".to_string()),
        ChatMessage::assistant().with_text("Hi!".to_string()),
        ChatMessage::user().with_text("How are you?".to_string()),
    ];
    assert_eq!(
        render(&messages),
        concat!(
            "<|message_system|><|content_text|>Be brief.<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
            "<|message_user|><|content_text|>Hello!<|end_message|>",
            "<|message_model|><|content_text|>Hi!<|end_message|><|content_model_end_sampling|>",
            "<|message_user|><|content_text|>How are you?<|end_message|>",
            "<|message_model|>",
        )
    );
}

#[test]
fn reasoning_efforts_set_the_thinking_level() {
    for (effort, level) in [
        (None, "0.9"),
        (Some(ReasoningEffort::Default), "0.9"),
        (Some(ReasoningEffort::Disabled), "0"),
        (Some(ReasoningEffort::Low), "0.2"),
        (Some(ReasoningEffort::Medium), "0.7"),
        (Some(ReasoningEffort::High), "0.9"),
        (Some(ReasoningEffort::XHigh), "0.99"),
    ] {
        let mut messages = Vec::new();
        if let Some(effort) = effort {
            messages.push(ChatMessage::system().with_reasoning_effort(effort));
        }
        messages.push(ChatMessage::user().with_text("Hi".to_string()));
        assert_eq!(
            render(&messages),
            format!(
                "<|message_system|><|content_text|>Thinking effort level: {level}<|end_message|>\
                 <|message_user|><|content_text|>Hi<|end_message|><|message_model|>"
            ),
            "{effort:?}"
        );
    }

    let messages = vec![
        ChatMessage::system().with_text("Be brief.".to_string()).with_reasoning_effort(ReasoningEffort::Low),
        ChatMessage::user().with_text("Hi".to_string()),
    ];
    assert_eq!(
        render(&messages),
        concat!(
            "<|message_system|><|content_text|>Be brief.<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
            "<|message_user|><|content_text|>Hi<|end_message|><|message_model|>",
        )
    );
}

#[test]
fn capabilities_and_defaults() {
    let capabilities = HanashiConfig::Inkling2.capabilities().unwrap();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_disable_reasoning);
    assert!(capabilities.supports_tools);
    assert!(capabilities.supports_multiple_tool_calls);
    assert!(!capabilities.requires_tools);
    for effort in [
        ReasoningEffort::Disabled,
        ReasoningEffort::Default,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
    ] {
        assert!(capabilities.reasoning_efforts.contains(&effort), "missing {effort}");
    }
    assert_eq!(HanashiConfig::Inkling2.default_reasoning_effort().unwrap(), Some(ReasoningEffort::High));

    let config = HanashiConfig::Inkling2.resolve().unwrap();
    assert_eq!(config.tokens.eos_token_id, Some(200006));
    assert_eq!(config.tokens.stop_token_ids, vec![200006]);
    // Structured output starts with the answer, which follows the text marker.
    assert_eq!(config.rendering.grammar_trigger(&[]), Some("<|content_text|>"));
    assert_eq!(
        config.rendering.grammar_trigger(&[ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled)]),
        None
    );
}

#[test]
fn tool_results_render_as_text() {
    let call = ChatMessage::assistant().with_tool_call(weather_call(serde_json::json!({"city": "Paris"})));
    let prefix = concat!(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
        "<|message_user|><|content_text|>Weather?<|end_message|>",
        "<|message_model|>get_weather<|content_invoke_tool_json|>",
        r#"{"name":"get_weather","args":{"city":"Paris"}}"#,
        "<|end_message|><|content_model_end_sampling|>",
    );
    for (name, value, expected) in [
        (
            Some("get_weather"),
            serde_json::json!("Sunny"),
            "<|message_tool|>get_weather<|content_text|>Sunny<|end_message|>",
        ),
        (
            Some("get_weather"),
            serde_json::json!({"temperature": 21, "unit": "°C", "sky": {"clouds": null, "rain": false}}),
            r#"<|message_tool|>get_weather<|content_text|>{"temperature":21,"unit":"°C","sky":{"clouds":null,"rain":false}}<|end_message|>"#,
        ),
        (
            Some("get_weather"),
            serde_json::json!([1, "two", null]),
            r#"<|message_tool|>get_weather<|content_text|>[1,"two",null]<|end_message|>"#,
        ),
        (
            Some("get_weather"),
            serde_json::json!(21.5),
            "<|message_tool|>get_weather<|content_text|>21.5<|end_message|>",
        ),
        (
            Some("get_weather"),
            serde_json::json!(true),
            "<|message_tool|>get_weather<|content_text|>true<|end_message|>",
        ),
        (
            Some("get_weather"),
            serde_json::Value::Null,
            "<|message_tool|>get_weather<|content_text|>null<|end_message|>",
        ),
        // Without a name (the reference template looks it up from the call ID) the header stays bare.
        (None, serde_json::json!("Sunny"), "<|message_tool|><|content_text|>Sunny<|end_message|>"),
        (
            None,
            serde_json::json!({"temperature": 21}),
            r#"<|message_tool|><|content_text|>{"temperature":21}<|end_message|>"#,
        ),
    ] {
        let result = ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: Some("call_1".to_string()),
            name: name.map(str::to_string),
            value: Value::from(value.clone()),
        });
        let messages = vec![ChatMessage::user().with_text("Weather?".to_string()), call.clone(), result];
        assert_eq!(render(&messages), format!("{prefix}{expected}{GENERATION_PROMPT}"), "{value}");
    }
}

#[test]
fn unparsed_tool_calls_do_not_break_rendering() {
    let messages = vec![
        ChatMessage::user().with_text("Weather?".to_string()),
        ChatMessage::assistant()
            .with_text("Checking.".to_string())
            // A truncated call stays a candidate: it has no name and is not replayed.
            .with_tool_call_candidate(Value::from(serde_json::json!(r#"{"name":"get_wea"#)))
            .with_tool_call(weather_call(serde_json::json!("Paris")))
            .with_tool_call(weather_call(serde_json::json!({}))),
        ChatMessage::user().with_text("Well?".to_string()),
    ];
    assert_eq!(
        render(&messages),
        concat!(
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
            "<|message_user|><|content_text|>Weather?<|end_message|>",
            "<|message_model|><|content_text|>Checking.<|end_message|>",
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":"Paris"}<|end_message|>"#,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            "<|message_user|><|content_text|>Well?<|end_message|>",
            "<|message_model|>",
        )
    );
}

#[test]
fn decoding_reasoning_and_text() {
    let mut encoding = encoding();
    encoding.encode(vec![ChatMessage::user().with_text("Hi".to_string())]).unwrap();
    assert_eq!(
        encoding.state().text(),
        concat!(
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
            "<|message_user|><|content_text|>Hi<|end_message|>",
            "<|message_model|>",
        )
    );
    let reply = encoding.state().messages.last().unwrap();
    assert_eq!(reply.role, ChatRole::Assistant {});
    assert!(reply.content.is_empty());

    let reasoning = "The user greets me.\nI greet back.";
    let text = "Hello! How can I help?";
    let completion = format!(
        "<|content_thinking|>{reasoning}<|end_message|><|message_model|><|content_text|>{text}<|end_message|><|content_model_end_sampling|>"
    );
    let replies = stream(&mut encoding, &completion);
    for reply in &replies {
        assert!(reasoning.starts_with(&reply.reasoning().unwrap_or_default()), "{reply:?}");
        assert!(text.starts_with(&reply.text().unwrap_or_default()), "{reply:?}");
        assert!(reply.tool_calls().is_empty() && candidates(reply) == 0, "{reply:?}");
    }
    let reply = replies.last().unwrap();
    assert_eq!(reply.reasoning().as_deref(), Some(reasoning));
    assert_eq!(reply.text().as_deref(), Some(text));
    // No text is shown while the model is still reasoning.
    assert!(replies.iter().any(|reply| reply.reasoning().is_some() && reply.text().is_none()));

    let mut encoding = self::encoding();
    encoding.encode(vec![ChatMessage::user().with_text("Hi".to_string())]).unwrap();
    let reply =
        stream(&mut encoding, "<|content_text|>Hello!<|end_message|><|content_model_end_sampling|>").pop().unwrap();
    assert_eq!(reply.reasoning(), None);
    assert_eq!(reply.text().as_deref(), Some("Hello!"));
}

#[test]
fn decoding_tool_calls() {
    let mut encoding = encoding();
    encoding
        .encode(vec![weather_tools(), ChatMessage::user().with_text("What is the weather in Paris?".to_string())])
        .unwrap();
    assert!(encoding.state().text().starts_with(TOOLS_DECLARATION));

    let arguments = r#"{"units":"metric","city":"Paris"}"#;
    let call = format!(
        r#"<|message_model|>get_weather<|content_invoke_tool_json|>{{"name":"get_weather","args":{arguments}}}<|end_message|>"#
    );
    let completion =
        format!("<|content_thinking|>I need the weather.<|end_message|>{call}<|content_model_end_sampling|>");
    let replies = stream(&mut encoding, &completion);

    for reply in &replies {
        assert!(reply.text().is_none(), "the tool name must not leak into the text: {reply:?}");
        assert!("I need the weather.".starts_with(&reply.reasoning().unwrap_or_default()), "{reply:?}");
        assert!(reply.tool_calls().len() + candidates(reply) <= 1, "{reply:?}");
    }
    // While the JSON streams the call is a candidate; it becomes a call once its part ends.
    let replies_until = |end: usize| encoding.tokenize(&completion[..end]).unwrap().len();
    let call_start = replies_until(completion.find("<|content_invoke_tool_json|>").unwrap());
    let call_end = replies_until(completion.rfind("<|end_message|>").unwrap());
    assert!(replies[..call_start].iter().all(|reply| candidates(reply) == 0 && reply.tool_calls().is_empty()));
    let streaming = &replies[call_start..call_end];
    assert!(!streaming.is_empty());
    assert!(streaming.iter().all(|reply| candidates(reply) == 1 && reply.tool_calls().is_empty()), "{streaming:?}");
    assert!(replies[call_end..].iter().all(|reply| candidates(reply) == 0 && reply.tool_calls().len() == 1));

    let reply = replies.last().unwrap();
    assert_eq!(reply.reasoning().as_deref(), Some("I need the weather."));
    assert_eq!(reply.text(), None);
    assert_eq!(candidates(reply), 0);
    assert_eq!(
        reply.tool_calls(),
        vec![ToolCall {
            identifier: None,
            name: "get_weather".to_string(),
            arguments: Value::from(serde_json::from_str::<serde_json::Value>(arguments).unwrap()),
        }]
    );
}

#[test]
fn decoding_text_and_several_tool_calls() {
    let mut encoding = encoding();
    encoding.encode(vec![weather_tools(), ChatMessage::user().with_text("Paris and Rome?".to_string())]).unwrap();
    let completion = concat!(
        "<|content_text|>Checking both cities.<|end_message|>",
        r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>"#,
        r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Rome","units":"metric"}}<|end_message|>"#,
        "<|content_model_end_sampling|>",
    );
    let reply = stream(&mut encoding, completion).pop().unwrap();
    assert_eq!(reply.text().as_deref(), Some("Checking both cities."));
    let calls = reply.tool_calls();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.name == "get_weather" && call.identifier.is_none()));
    assert_eq!(calls[0].arguments, Value::from(serde_json::json!({"city": "Paris"})));
    assert_eq!(calls[1].arguments, Value::from(serde_json::json!({"city": "Rome", "units": "metric"})));
}

#[test]
fn decoding_tolerates_malformed_parts() {
    for (completion, reasoning, text, calls) in [
        // A part that is not terminated by <|end_message|>.
        (
            "<|content_thinking|>Thinking.<|message_model|><|content_text|>Answer.<|content_model_end_sampling|>",
            Some("Thinking."),
            Some("Answer."),
            0,
        ),
        // Padding is ignored.
        ("<|content_text|>Answer.<|endoftext|><|end_message|><|content_model_end_sampling|>", None, Some("Answer."), 0),
        // A call written with the usual `arguments` key and JSON that needs repair.
        (
            r#"get_weather<|content_invoke_tool_json|>{"name": "get_weather", "arguments": {"city": "Paris",}}<|end_message|><|content_model_end_sampling|>"#,
            None,
            None,
            1,
        ),
    ] {
        let mut encoding = encoding();
        encoding.encode(vec![weather_tools(), ChatMessage::user().with_text("Weather?".to_string())]).unwrap();
        let reply = stream(&mut encoding, completion).pop().unwrap();
        assert_eq!(reply.reasoning().as_deref(), reasoning, "{completion}");
        assert_eq!(reply.text().as_deref(), text, "{completion}");
        assert_eq!(reply.tool_calls().len(), calls, "{completion}");
        assert_eq!(candidates(&reply), 0, "{completion}");
        if calls == 1 {
            assert_eq!(reply.tool_calls()[0].arguments, Value::from(serde_json::json!({"city": "Paris"})));
        }
    }

    // A truncated call stays a candidate.
    let mut encoding = encoding();
    encoding.encode(vec![weather_tools(), ChatMessage::user().with_text("Weather?".to_string())]).unwrap();
    let reply = stream(&mut encoding, r#"get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"ci"#)
        .pop()
        .unwrap();
    assert!(reply.tool_calls().is_empty());
    assert_eq!(candidates(&reply), 1);
}

/// With reasoning disabled there is no grammar trigger, so a grammar constrains the reply from its first
/// token and the model cannot write <|content_text|>: a finished turn without content parts is its text.
#[test]
fn headerless_reply_is_text_once_the_turn_ends() {
    let mut encoding = encoding();
    encoding
        .encode(vec![
            ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
            ChatMessage::user().with_text("Answer in JSON.".to_string()),
        ])
        .unwrap();
    assert!(encoding.state().text().contains("Thinking effort level: 0<|end_message|>"));
    let replies = stream(&mut encoding, r#"{"answer": 42}<|content_model_end_sampling|>"#);
    let (reply, streaming) = replies.split_last().unwrap();
    // The header of a part may be a tool name, so it is not shown while it streams.
    assert!(streaming.iter().all(|reply| reply.content.is_empty()), "{streaming:?}");
    assert_eq!(reply.text().as_deref(), Some(r#"{"answer": 42}"#));
    assert!(reply.tool_calls().is_empty() && candidates(reply) == 0);
}

#[test]
fn tool_call_continuation_keeps_the_sampled_tokens() {
    let mut encoding = encoding();
    let mut history = vec![
        ChatMessage::system().with_text("You are a helpful assistant.".to_string()),
        weather_tools(),
        ChatMessage::user().with_text("What is the weather in Paris?".to_string()),
    ];
    encoding.encode(history.clone()).unwrap();
    // The sampled call has unsorted arguments: replaying the history renders them sorted.
    stream(
        &mut encoding,
        concat!(
            "<|content_thinking|>I need the weather.<|end_message|>",
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"units":"metric","city":"Paris"}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
        ),
    );
    let sampled = encoding.state().tokens.clone();
    let sampled_text = encoding.state().text();

    let mut assistant = encoding.state().messages.last().unwrap().clone();
    let Some(ChatContentBlock::ToolCall {
        value,
    }) = assistant.content.last_mut()
    else {
        panic!("expected a tool call: {assistant:?}");
    };
    value.identifier = Some("call_1".to_string());
    history.push(assistant);
    history.push(weather_result(serde_json::json!("Sunny, 21 °C")));

    let suffix = encoding.try_append(&history).unwrap().expect("must continue after the sampled tool call");
    assert!(!suffix.is_empty());
    assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
    assert_eq!(
        &encoding.state().text()[sampled_text.len()..],
        "<|message_tool|>get_weather<|content_text|>Sunny, 21 °C<|end_message|><|message_model|>"
    );
    assert_eq!(encoding.state().messages.len(), history.len() + 1);
    assert!(encoding.state().messages.last().unwrap().content.is_empty());

    let reply = stream(&mut encoding, "<|content_text|>It is sunny.<|end_message|><|content_model_end_sampling|>")
        .pop()
        .unwrap();
    assert_eq!(reply.text().as_deref(), Some("It is sunny."));
    assert_eq!(&encoding.state().messages[..history.len()], history);

    // A fresh session replays the canonical form, which differs from the sampled arguments.
    let mut replay = self::encoding();
    replay.encode(history.clone()).unwrap();
    assert!(!replay.state().text().starts_with(&sampled_text));
    assert!(replay.state().text().contains(r#"{"name":"get_weather","args":{"city":"Paris","units":"metric"}}"#));
}

#[test]
fn real_tokenizer_round_trip() {
    let tokenizer_directory = tokenizer_directory(MODEL_NAME);
    if !tokenizer_directory.exists() {
        return;
    }
    let tokenizer = Arc::new(load_tokenizer(MODEL_NAME));
    let config = HanashiConfig::Inkling2.resolve().unwrap();
    for (token, id) in [
        ("<|message_user|>", 200000),
        ("<|message_model|>", 200001),
        ("<|message_system|>", 200002),
        ("<|message_tool|>", 200003),
        ("<|content_text|>", 200004),
        ("<|content_model_end_sampling|>", 200006),
        ("<|content_thinking|>", 200008),
        ("<|end_message|>", 200010),
        ("<|content_xml|>", 200024),
        ("<|content_invoke_tool_json|>", 200049),
    ] {
        assert_eq!(tokenizer.token_to_id(token), Some(id), "{token}");
        assert!(tokenizer.get_added_vocabulary().is_special_token(token), "{token}");
    }
    assert_eq!(config.tokens.eos_token_id, tokenizer.token_to_id("<|content_model_end_sampling|>"));

    let mut encoding = HanashiEncodingImpl::new(HanashiConfig::Inkling2, tokenizer.clone()).unwrap();
    let mut history = conversation();
    history.truncate(3);
    encoding.encode(history.clone()).unwrap();
    let prompt_end = CONVERSATION_MESSAGES.find("<|message_model|>").unwrap();
    assert_eq!(
        encoding.state().text(),
        format!("{TOOLS_DECLARATION}{}{GENERATION_PROMPT}", &CONVERSATION_MESSAGES[..prompt_end])
    );
    assert_eq!(encoding.state().tokens.last().unwrap().id, 200001);

    let completion = concat!(
        "<|content_thinking|>The user wants the weather, so I call the tool.<|end_message|>",
        r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","units":"metric"}}<|end_message|>"#,
        "<|content_model_end_sampling|>",
    );
    for id in tokenizer.encode(completion, false).unwrap().get_ids() {
        encoding.decode(vec![*id]).unwrap();
    }
    let reply = encoding.state().messages.last().unwrap();
    assert_eq!(reply.reasoning().as_deref(), Some("The user wants the weather, so I call the tool."));
    assert_eq!(reply.tool_calls().len(), 1);
    assert_eq!(reply.tool_calls()[0].arguments, Value::from(serde_json::json!({"city": "Paris", "units": "metric"})));

    let mut assistant = reply.clone();
    if let Some(ChatContentBlock::ToolCall {
        value,
    }) = assistant.content.last_mut()
    {
        value.identifier = Some("call_1".to_string());
    }
    history.push(assistant);
    history.push(weather_result(serde_json::json!("Sunny, 21 °C")));
    encoding.try_append(&history).unwrap().expect("must continue after the sampled tool call");

    let completion = "<|content_text|>It is sunny and 21 °C in Paris.<|end_message|><|content_model_end_sampling|>";
    for id in tokenizer.encode(completion, false).unwrap().get_ids() {
        encoding.decode(vec![*id]).unwrap();
    }
    let reply = encoding.state().messages.last().unwrap();
    assert_eq!(reply.text().as_deref(), Some("It is sunny and 21 °C in Paris."));
}
