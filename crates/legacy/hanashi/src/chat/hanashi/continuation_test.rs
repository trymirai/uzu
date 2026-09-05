use std::sync::Arc;

use shoji::types::{
    basic::{ReasoningEffort, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

use super::{HanashiEncodingImpl, config::HanashiConfig};
use crate::Encoding as _;

fn tokenizer() -> Arc<Tokenizer> {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocabulary: Vec<(String, u32)> =
        alphabet.into_iter().enumerate().map(|(index, character)| (character.to_string(), index as u32)).collect();
    vocabulary.push(("ab".to_string(), 256));
    vocabulary.push(("abc".to_string(), 257));
    let vocabulary: [(String, u32); 258] = vocabulary.try_into().unwrap();
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, vec![("a".to_string(), "b".to_string()), ("ab".to_string(), "c".to_string())])
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer
        .add_tokens(&["system", "developer", "user", "assistant", "tool"].map(|value| AddedToken::from(value, false)));
    tokenizer.add_special_tokens(
        &[
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ]
        .map(|value| AddedToken::from(value, true)),
    );
    Arc::new(tokenizer)
}

fn encoding() -> HanashiEncodingImpl {
    HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer()).unwrap()
}

fn prompt() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default),
        ChatMessage::user().with_text("Inspect the repository.".to_string()),
    ]
}

fn decode(
    encoding: &mut HanashiEncodingImpl,
    text: &str,
) {
    for token_id in encoding.tokenize(text).unwrap() {
        encoding.decode(vec![token_id]).unwrap();
    }
}

fn tool_call(argument: &str) -> String {
    format!("<tool_call>\n<function=inspect>\n<parameter=options>\n{argument}\n</parameter>\n</function>\n</tool_call>")
}

fn tool_result(value: &str) -> ChatMessage {
    ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier: Some(value.to_string()),
        name: Some("inspect".to_string()),
        value: Value {
            json: serde_json::to_string(value).unwrap(),
        },
    })
}

#[test]
fn continuation_preserves_sampled_formatting_across_three_turns() {
    let mut encoding = encoding();
    let mut history = prompt();
    encoding.encode(history.clone()).unwrap();
    let variants = [
        format!("Ready.\n</think>\n\n{}", tool_call(r#"{"a":1,"b":2}"#)),
        format!("Ready.\n\n</think>\n\n{}", tool_call(r#"{"html":"<div>"}"#)),
        format!("Ready.\n</think>\n\nWorking.\n{}\n", tool_call(r#"{"a":1}"#)),
    ];
    for generated in variants {
        decode(&mut encoding, &format!("{generated}<|im_end|>"));
        assert_eq!(encoding.state().messages.last().unwrap().tool_calls().len(), 1);
        history.push(encoding.state().messages.last().unwrap().clone());
        assert!(encoding.record_completion(history.clone()).unwrap());
        let old_tokens = encoding.state().tokens.clone();
        let old_messages = history.clone();
        history.push(tool_result("done"));
        let suffix = encoding.try_append(&history).unwrap().expect("completed tool turn should append");
        assert!(!suffix.is_empty());
        assert_eq!(&encoding.state().tokens[..old_tokens.len()], old_tokens);
        assert_eq!(&encoding.state().messages[..old_messages.len()], old_messages);
        assert_eq!(&encoding.state().messages[..history.len()], history);
        assert_eq!(encoding.state().messages.len(), history.len() + 1);
        assert_eq!(encoding.state().messages.last().unwrap().role, ChatRole::Assistant {});
        assert_eq!(
            encoding.state().tokens[old_tokens.len()..].iter().map(|token| token.id).collect::<Vec<_>>(),
            suffix
        );
    }
    decode(&mut encoding, "Complete.\n</think>\n\nFinished.<|im_end|>");
    assert_eq!(encoding.state().messages.last().unwrap().text().as_deref(), Some("Finished."));
}

#[test]
fn continuation_preserves_noncanonical_token_ids_and_split_unicode() {
    let mut encoding = encoding();
    let mut history = prompt();
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, "Ready.\n</think>\n\n");
    let sampled: Vec<_> = ["a", "b", "c"].into_iter().flat_map(|text| encoding.tokenize(text).unwrap()).collect();
    assert_ne!(sampled, encoding.tokenize("abc").unwrap());
    for token_id in sampled {
        encoding.decode(vec![token_id]).unwrap();
    }
    decode(&mut encoding, " café 🦀<|im_end|>");
    history.push(encoding.state().messages.last().unwrap().clone());
    assert!(encoding.record_completion(history.clone()).unwrap());
    let old_tokens = encoding.state().tokens.clone();
    history.push(ChatMessage::user().with_text("Continue café 🦀".to_string()));
    assert!(encoding.try_append(&history).unwrap().is_some());
    assert_eq!(&encoding.state().tokens[..old_tokens.len()], old_tokens);
    decode(&mut encoding, "Ready.\n</think>\n\nVoilà 🦀<|im_end|>");
    assert_eq!(encoding.state().messages.last().unwrap().text().as_deref(), Some("Voilà 🦀"));
}

#[test]
fn continuation_groups_tool_results_without_overwriting_logical_inputs() {
    let mut encoding = encoding();
    let mut history = prompt();
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, &format!("Ready.\n</think>\n\n{}\n{}<|im_end|>", tool_call("one"), tool_call("two")));
    history.push(encoding.state().messages.last().unwrap().clone());
    assert!(encoding.record_completion(history.clone()).unwrap());
    history.extend([tool_result("one"), tool_result("two")]);
    let suffix = encoding.try_append(&history).unwrap().unwrap();
    let text = encoding.tokenizer.decode(&suffix, false).unwrap();
    assert_eq!(text.matches("<|im_start|>user").count(), 1);
    assert_eq!(text.matches("<tool_response>").count(), 2);
    assert_eq!(&encoding.state().messages[..history.len()], history);
}

#[test]
fn continuation_rejects_logical_edits_and_new_controls_without_mutation() {
    let mut encoding = encoding();
    let mut history = prompt();
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, "Ready.\n</think>\n\nDone.<|im_end|>");
    history.push(encoding.state().messages.last().unwrap().clone());
    assert!(encoding.record_completion(history.clone()).unwrap());
    let original = encoding.state().clone();
    let mut edited = history.clone();
    edited[1] = ChatMessage::user().with_text(" Inspect the repository. ".to_string());
    edited.push(ChatMessage::user().with_text("Next".to_string()));
    assert!(encoding.try_append(&edited).unwrap().is_none());
    assert_eq!(encoding.state(), &original);
    for message in [
        ChatMessage::system().with_text("changed".to_string()),
        ChatMessage::developer().with_tool_namespaces(vec![]),
        ChatMessage::user().with_text("Next".to_string()).with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_block(ChatContentBlock::Image {
            url: "image".to_string(),
        }),
        ChatMessage::assistant().with_text("partial".to_string()),
    ] {
        let mut next = history.clone();
        next.push(message);
        assert!(encoding.try_append(&next).unwrap().is_none());
        assert_eq!(encoding.state(), &original);
    }
    assert!(encoding.try_append(&history).unwrap().is_none());
    history.push(ChatMessage::user().with_text("Next".to_string()));
    assert!(encoding.try_append(&history).unwrap().is_some());
    assert!(encoding.try_append(&history).unwrap().is_none());
}

#[test]
fn completion_rejects_implicitly_closed_bounded_sections() {
    for generated in [
        "unfinished reasoning<|im_end|>",
        "Ready.\n</think>\n\n<tool_call>\n<function=inspect>\n<|im_end|>",
        "Ready.\n</think>\n\nDone.",
    ] {
        let mut encoding = encoding();
        encoding.encode(prompt()).unwrap();
        decode(&mut encoding, generated);
        assert!(!encoding.record_completion(encoding.state().messages.clone()).unwrap());
    }
}

#[test]
fn continuation_is_limited_to_bundled_qwen35_and_reset_clears_completion() {
    let tokenizer = tokenizer();
    assert_ne!(tokenizer.token_to_id("<|im_end|>"), HanashiConfig::Qwen35.resolve().unwrap().tokens.eos_token_id);
    assert!(HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer.clone()).unwrap().supports_continuation());
    for config in [
        HanashiConfig::Qwen36,
        HanashiConfig::Custom {
            config: HanashiConfig::Qwen35.resolve().unwrap(),
        },
    ] {
        assert!(!HanashiEncodingImpl::new(config, tokenizer.clone()).unwrap().supports_continuation());
    }
    let mut encoding = encoding();
    encoding.encode(prompt()).unwrap();
    decode(&mut encoding, "Ready.\n</think>\n\nDone.<|im_end|>");
    let mut history = encoding.state().messages.clone();
    assert!(encoding.record_completion(history.clone()).unwrap());
    encoding.reset().unwrap();
    history.push(ChatMessage::user().with_text("Next".to_string()));
    assert!(encoding.try_append(&history).unwrap().is_none());
    assert!(encoding.state().tokens.is_empty());
}

#[test]
fn continuation_accepts_disabled_thinking_preamble_and_authoritative_call_ids() {
    let mut encoding = encoding();
    let mut history = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text("Inspect the repository.".to_string()),
    ];
    encoding.encode(history.clone()).unwrap();
    decode(&mut encoding, &format!("{}<|im_end|>", tool_call("src")));
    let mut assistant = encoding.state().messages.last().unwrap().clone();
    let ChatContentBlock::ToolCall {
        value,
    } = assistant.content.last_mut().unwrap()
    else {
        panic!("expected parsed tool call");
    };
    value.identifier = Some("assigned-by-session".to_string());
    history.push(assistant);
    assert!(encoding.record_completion(history.clone()).unwrap());
    assert_eq!(encoding.state().messages, history);
    history.push(tool_result("done"));
    assert!(encoding.try_append(&history).unwrap().is_some());
    decode(&mut encoding, "Finished.<|im_end|>");
    assert_eq!(encoding.state().messages.last().unwrap().text().as_deref(), Some("Finished."));
}
