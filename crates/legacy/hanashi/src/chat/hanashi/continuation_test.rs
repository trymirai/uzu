use std::sync::Arc;

use shoji::types::{
    basic::{ReasoningEffort, Value},
    session::chat::{ChatContentBlock, ChatMessage},
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

use super::{HanashiEncodingImpl, config::HanashiConfig};
use crate::Encoding as _;

fn tokenizer() -> Arc<Tokenizer> {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: Vec<(String, u32)> =
        alphabet.into_iter().enumerate().map(|(index, value)| (value.to_string(), index as u32)).collect();
    let vocabulary: [(String, u32); 256] = vocabulary.try_into().unwrap();
    let model = BPE::builder().vocab_and_merges(vocabulary, Vec::new()).build().unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_tokens(&["system", "user", "assistant", "tool"].map(|value| AddedToken::from(value, false)));
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

fn decode(
    encoding: &mut HanashiEncodingImpl,
    text: &str,
) {
    for token in encoding.tokenize(text).unwrap() {
        encoding.decode(vec![token]).unwrap();
    }
}

#[test]
fn compact_tool_call_continuation_preserves_sampled_prefix() {
    let mut encoding = HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer()).unwrap();
    let mut history = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default),
        ChatMessage::user().with_text("Inspect the repository.".to_string()),
    ];
    encoding.encode(history.clone()).unwrap();
    decode(
        &mut encoding,
        "Ready.\n</think>\n\n<tool_call>\n<function=inspect>\n<parameter=options>\n{\"a\":1,\"b\":2}\n</parameter>\n</function>\n</tool_call><|im_end|>",
    );
    let sampled_text = encoding.state().text();
    history.push(encoding.state().messages.last().unwrap().clone());
    let generated_identifier = Some("generated-tool-call-id".to_string());
    let ChatContentBlock::ToolCall {
        value: tool_call,
    } = history.last_mut().unwrap().content.last_mut().unwrap()
    else {
        panic!("expected parsed tool call");
    };
    tool_call.identifier.clone_from(&generated_identifier);
    let canonical = encoding.renderer.render(&history, false, None, None, None).unwrap();
    assert!(!canonical.starts_with(&sampled_text), "fixture must exercise a noncanonical sampled tool call");
    let sampled = encoding.state().tokens.clone();
    history.push(ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier: generated_identifier,
        name: Some("inspect".to_string()),
        value: Value {
            json: "\"done\"".to_string(),
        },
    }));

    let suffix = encoding.try_append(&history).unwrap().expect("tool result should extend the sampled prefix");

    assert!(!suffix.is_empty());
    assert_eq!(&encoding.state().tokens[..sampled.len()], sampled);
    assert_eq!(encoding.state().tokens[sampled.len()..].iter().map(|token| token.id).collect::<Vec<_>>(), suffix);
}
