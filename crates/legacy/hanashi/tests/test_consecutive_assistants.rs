use std::sync::Arc;

use hanashi::{
    Encoding,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::session::chat::{ChatMessage, ChatRole};
use tokenizers::{
    AddedToken, Tokenizer,
    models::bpe::{BPE, Vocab},
    pre_tokenizers::byte_level::ByteLevel,
};

fn encoding(config: HanashiConfig) -> HanashiEncodingImpl {
    // A byte tokenizer exercises rendering and streaming without downloaded model assets.
    let vocab: Vocab =
        ByteLevel::alphabet().into_iter().enumerate().map(|(id, c)| (c.to_string(), id as u32)).collect();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_special_tokens(
        &["<|im_start|>", "<|im_end|>", "<think>", "</think>"].map(|token| AddedToken::from(token, true)),
    );
    tokenizer.add_tokens(&["system", "user", "assistant"].map(|role| AddedToken::from(role, false).single_word(true)));
    HanashiEncodingImpl::new(config, Arc::new(tokenizer)).unwrap()
}

fn compaction_history() -> Vec<ChatMessage> {
    vec![
        ChatMessage::user().with_text("Original task".into()),
        ChatMessage::assistant().with_text("Compacted session summary".into()),
        ChatMessage::assistant()
            .with_reasoning("Retained reasoning".into())
            .with_text("Retained recent assistant turn".into()),
    ]
}

fn assert_reply_preserves_history(
    encoding: &mut HanashiEncodingImpl,
    history: &[ChatMessage],
) {
    assert_eq!(&encoding.state().messages[..history.len()], history);
    assert_eq!(encoding.state().messages.len(), history.len() + 1);
    assert_eq!(encoding.state().messages.last().unwrap().role, ChatRole::Assistant {});
    assert!(encoding.state().messages.last().unwrap().text().unwrap_or_default().is_empty());

    let completion = if encoding.state().text().ends_with("<think>\n") {
        "Done thinking.</think>\n\nResumed successfully.<|im_end|>"
    } else {
        "Resumed successfully.<|im_end|>"
    };
    for token in encoding.tokenize(completion).unwrap() {
        encoding.decode(vec![token]).unwrap();
    }
    assert_eq!(&encoding.state().messages[..history.len()], history);
    assert_eq!(encoding.state().messages.len(), history.len() + 1);
    assert_eq!(encoding.state().messages.last().unwrap().text().as_deref(), Some("Resumed successfully."));
}

#[test]
fn compaction_history_accepts_consecutive_assistants() {
    for extra_assistant in [false, true] {
        let mut encoding = encoding(HanashiConfig::Qwen36);
        let mut history = compaction_history();
        if extra_assistant {
            history.push(ChatMessage::assistant().with_text("Another retained turn".into()));
        }
        history.push(ChatMessage::user().with_text("Continue".into()));
        encoding.encode(history.clone()).unwrap();

        let prompt = encoding.state().text();
        assert!(prompt.contains(concat!(
            "<|im_start|>assistant\nCompacted session summary<|im_end|>\n",
            "<|im_start|>assistant\nRetained recent assistant turn<|im_end|>"
        )));
        assert_reply_preserves_history(&mut encoding, &history);
    }
}

#[test]
fn history_ending_with_assistant_keeps_a_separate_reply() {
    for config in [HanashiConfig::Qwen36, HanashiConfig::Qwen3Instruct] {
        let mut encoding = encoding(config);
        // Even a single historical assistant must survive the new generation prompt.
        let history = vec![
            ChatMessage::user().with_text("Original task".into()),
            ChatMessage::assistant().with_text("Summary".into()),
        ];
        encoding.encode(history.clone()).unwrap();
        assert_reply_preserves_history(&mut encoding, &history);

        encoding.reset().unwrap();
        let mut history = compaction_history();
        history[2] = ChatMessage::assistant().with_text("Retained recent assistant turn".into());
        encoding.encode(history.clone()).unwrap();
        assert_reply_preserves_history(&mut encoding, &history);
    }
}
