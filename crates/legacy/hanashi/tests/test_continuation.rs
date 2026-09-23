use std::sync::Arc;

use hanashi::{
    Encoding as _,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::{
    basic::{ReasoningEffort, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

fn encoding(config: HanashiConfig) -> HanashiEncodingImpl {
    let mut config = config.resolve().unwrap();
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
    tokenizer.add_tokens(
        &["system", "user", "assistant", "tool", "model", "inspect"].map(|value| AddedToken::from(value, false)),
    );
    let mut markers = config.parsing.framing_config().tokens;
    markers.extend(["<bos>".to_string(), "<eos>".to_string()]);
    tokenizer.add_special_tokens(&markers.into_iter().map(|value| AddedToken::from(value, true)).collect::<Vec<_>>());
    config.tokens.bos_token_id = tokenizer.token_to_id("<bos>");
    config.tokens.eos_token_id = tokenizer.token_to_id("<eos>");
    HanashiEncodingImpl::new(
        HanashiConfig::Custom {
            config,
        },
        Arc::new(tokenizer),
    )
    .unwrap()
}

fn decode(
    encoding: &mut HanashiEncodingImpl,
    text: &str,
) {
    for id in encoding.tokenize(text).unwrap() {
        // Sample a valid alternative tokenization, as an autoregressive model can.
        if encoding.tokenize("inspect").unwrap() == [id] {
            for character in "inspect".chars() {
                let id = encoding.tokenize(&character.to_string()).unwrap()[0];
                encoding.decode(vec![id]).unwrap();
            }
        } else {
            encoding.decode(vec![id]).unwrap();
        }
    }
}

fn tool_call(options: &str) -> String {
    format!("<tool_call>\n<function=inspect>\n<parameter=options>\n{options}\n</parameter>\n</function>\n</tool_call>")
}

fn history() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::developer().with_block(ChatContentBlock::Tools {
            namespaces: vec![ToolNamespace {
                name: "functions".to_string(),
                description: None,
                tools: vec![ToolDescription::Function {
                    tool_function: ToolFunction {
                        name: "inspect".to_string(),
                        description: String::new(),
                        parameters: Some(Value::from(serde_json::json!({"type": "object"}))),
                        return_definition: None,
                    },
                }],
            }],
        }),
        ChatMessage::user().with_text("Inspect the repository.".to_string()),
    ]
}

fn add_tool_result(
    encoding: &HanashiEncodingImpl,
    history: &mut Vec<ChatMessage>,
    options: &str,
) {
    let mut assistant = encoding.state().messages.last().unwrap().clone();
    let ChatContentBlock::ToolCall {
        value,
    } = assistant.content.last_mut().unwrap()
    else {
        panic!("expected a tool call");
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&value.arguments.json).unwrap(),
        serde_json::json!({"options": options})
    );
    // Nagare assigns an ID after parsing; it is absent from the sampled encoding.
    value.identifier = Some(format!("call-{}", history.len()));
    let identifier = value.identifier.clone();
    history.push(assistant);
    history.push(ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier,
        name: Some("inspect".to_string()),
        value: Value::from(serde_json::json!("done")),
    }));
}

#[test]
fn formatting_and_sampled_token_ids_survive_repeated_tool_continuations() {
    for config in [HanashiConfig::Qwen35, HanashiConfig::Qwen36, HanashiConfig::Qwen38] {
        for options in [r#"{"a":1,"b":2}"#, r#"{"a":1, "b": 2}"#, "{\n  \"b\": 2,\n  \"a\": 1\n}"] {
            let mut encoding = encoding(config.clone());
            let mut history = history();
            encoding.encode(history.clone()).unwrap();
            for _ in 0..3 {
                // Parameter text is preserved verbatim; extra message whitespace
                // still makes the sampled prefix differ from canonical rendering.
                decode(&mut encoding, &(tool_call(options) + "  <|im_end|>"));
                let sampled = encoding.state().tokens.clone();
                let sampled_text = encoding.state().text();
                assert_ne!(
                    sampled.iter().map(|token| token.id).collect::<Vec<_>>(),
                    encoding.tokenize(&sampled_text).unwrap()
                );
                add_tool_result(&encoding, &mut history, options);
                let canonical = canonical_text(config.clone(), history.clone());
                assert!(!canonical.starts_with(&sampled_text));

                let suffix = encoding.try_append(&history).unwrap().expect("must reuse the sampled prefix");

                assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
                assert_eq!(
                    encoding.state().tokens[sampled.len()..].iter().map(|token| token.id).collect::<Vec<_>>(),
                    suffix
                );
                assert_eq!(
                    &encoding.state().text()[sampled_text.len()..],
                    "\n<|im_start|>user\n<tool_response>\ndone\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                );
                assert!(matches!(encoding.state().messages.last().unwrap().role, ChatRole::Assistant {}));
            }
        }
    }
}

#[test]
fn edited_history_and_render_context_are_rejected_without_mutation() {
    let mut encoding = encoding(HanashiConfig::Qwen38);
    let mut history = history();
    encoding.encode(history.clone()).unwrap();
    let options = r#"{"a":1, "b": 2}"#;
    decode(&mut encoding, &(tool_call(options) + "<|im_end|>"));
    add_tool_result(&encoding, &mut history, options);
    let original = encoding.state().clone();
    let mut edited = history.clone();
    edited[2] = ChatMessage::user().with_text("Different request.".to_string());
    assert!(encoding.try_append(&edited).unwrap().is_none());
    assert_eq!(*encoding.state(), original);
    let mut edited = history.clone();
    edited[0] = ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default);
    assert!(encoding.try_append(&edited).unwrap().is_none());
    assert_eq!(*encoding.state(), original);
    let mut edited = history.clone();
    let ChatContentBlock::Tools {
        namespaces,
    } = &mut edited[1].content[0]
    else {
        unreachable!()
    };
    let ToolDescription::Function {
        tool_function,
    } = &mut namespaces[0].tools[0];
    tool_function.description = "Changed tool description.".to_string();
    assert!(encoding.try_append(&edited).unwrap().is_none());
    assert_eq!(*encoding.state(), original);
    // A rejected attempt must not poison a later valid continuation.
    assert!(encoding.try_append(&history).unwrap().is_some());
}

#[test]
fn partial_messages_and_unparsed_calls_are_rejected() {
    for tail in [tool_call(r#"{"a":1,"b":2}"#), "<tool_call>broken</tool_call><|im_end|>".to_string()] {
        let mut encoding = encoding(HanashiConfig::Qwen35);
        let mut history = history();
        encoding.encode(history.clone()).unwrap();
        decode(&mut encoding, &tail);
        history.push(encoding.state().messages.last().unwrap().clone());
        history.push(ChatMessage::user().with_text("Continue.".to_string()));
        let original = encoding.state().clone();
        assert!(encoding.try_append(&history).unwrap().is_none());
        assert_eq!(*encoding.state(), original);
    }
}

#[test]
fn different_message_delimiters_and_bos_are_supported() {
    let mut encoding = encoding(HanashiConfig::Gemma3);
    let mut history = vec![ChatMessage::user().with_text("Hello.".to_string())];
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, "  Hello!  <end_of_turn>");
    let sampled = encoding.state().tokens.clone();
    history.push(encoding.state().messages.last().unwrap().clone());
    history.push(ChatMessage::user().with_text("Continue.".to_string()));
    let canonical = canonical_text(HanashiConfig::Gemma3, history.clone());
    assert!(!canonical.starts_with(&encoding.state().text()));
    assert!(encoding.try_append(&history).unwrap().is_some());
    assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
    assert!(encoding.state().text().ends_with("\n<start_of_turn>user\nContinue.<end_of_turn>\n<start_of_turn>model\n"));
}

fn canonical_text(
    config: HanashiConfig,
    history: Vec<ChatMessage>,
) -> String {
    let mut encoding = encoding(config);
    encoding.encode(history).unwrap();
    encoding.state().text()
}

#[test]
fn context_dependent_rewriting_of_previous_reasoning_is_rejected() {
    let mut config = HanashiConfig::Qwen38.resolve().unwrap();
    config.rendering.jinja.template.insert_str(0, "{%- set preserve_thinking = false %}");
    let config = HanashiConfig::Custom {
        config,
    };
    let mut encoding = encoding(config.clone());
    let mut history = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default),
        ChatMessage::user().with_text("Say hello.".to_string()),
    ];
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, "Some thought.\n</think>\n\nHello.<|im_end|>");
    history.push(encoding.state().messages.last().unwrap().clone());
    history.push(ChatMessage::user().with_text("Another question.".to_string()));
    let original = encoding.state().clone();
    assert!(original.text().contains("Some thought."));
    assert!(!canonical_text(config, history.clone()).contains("Some thought."));
    assert!(encoding.try_append(&history).unwrap().is_none());
    assert_eq!(*encoding.state(), original);
}

#[test]
fn reasoning_mode_change_that_only_affects_suffix_can_reuse_prefix() {
    let mut encoding = encoding(HanashiConfig::Qwen35);
    let mut history = history();
    encoding.encode(history.clone()).unwrap();
    let options = r#"{"a":1, "b": 2}"#;
    decode(&mut encoding, &(tool_call(options) + "<|im_end|>"));
    add_tool_result(&encoding, &mut history, options);
    history[0] = ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default);
    let sampled = encoding.state().tokens.clone();
    assert!(encoding.try_append(&history).unwrap().is_some());
    assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
    assert!(encoding.state().text().ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn unicode_survives_prefix_boundary_detection() {
    let mut encoding = encoding(HanashiConfig::Qwen38);
    let mut history = history();
    history[2] = ChatMessage::user().with_text("Проверь репозиторий 🦀".to_string());
    encoding.encode(history.clone()).unwrap();
    let options = r#"{"a":1, "b": 2}"#;
    decode(&mut encoding, &(tool_call(options) + "<|im_end|>"));
    add_tool_result(&encoding, &mut history, options);
    let sampled = encoding.state().tokens.clone();
    assert!(encoding.try_append(&history).unwrap().is_some());
    assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
}
