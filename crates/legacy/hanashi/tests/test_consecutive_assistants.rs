use std::sync::Arc;

use hanashi::{
    Encoding,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::{
    basic::{ToolDescription, ToolFunction, ToolNamespace},
    session::chat::{ChatMessage, ChatRole},
};
use tokenizers::{
    AddedToken, Tokenizer,
    models::bpe::{BPE, Vocab},
    pre_tokenizers::byte_level::ByteLevel,
};

fn encoding(config: HanashiConfig) -> HanashiEncodingImpl {
    let mut config = config.resolve().unwrap();
    // A byte tokenizer exercises rendering and streaming without downloaded model assets.
    let vocab: Vocab =
        ByteLevel::alphabet().into_iter().enumerate().map(|(id, c)| (c.to_string(), id as u32)).collect();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    let mut special_tokens = config.parsing.framing_config().tokens;
    special_tokens.push("<bos>".into());
    tokenizer
        .add_special_tokens(&special_tokens.into_iter().map(|token| AddedToken::from(token, true)).collect::<Vec<_>>());
    tokenizer.add_tokens(&["system", "user", "assistant"].map(|role| AddedToken::from(role, false).single_word(true)));
    config.tokens.bos_token_id = config.tokens.bos_token_id.map(|_| tokenizer.token_to_id("<bos>").unwrap());
    HanashiEncodingImpl::new(
        HanashiConfig::Custom {
            config,
        },
        Arc::new(tokenizer),
    )
    .unwrap()
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
    completion: &str,
) {
    let prompt = encoding.state().text();
    assert_eq!(&encoding.state().messages[..history.len()], history);
    assert_eq!(encoding.state().messages.len(), history.len() + 1);
    assert_eq!(encoding.state().messages.last().unwrap().role, ChatRole::Assistant {});
    assert!(encoding.state().messages.last().unwrap().text().unwrap_or_default().is_empty());
    assert!(encoding.state().messages.last().unwrap().reasoning().unwrap_or_default().is_empty());

    for token in encoding.tokenize(completion).unwrap() {
        encoding.decode(vec![token]).unwrap();
        assert_eq!(&encoding.state().messages[..history.len()], history);
        assert_eq!(encoding.state().messages.len(), history.len() + 1);
        let reply = encoding.state().messages.last().unwrap();
        assert!("Resumed successfully.".starts_with(reply.text().unwrap_or_default().trim()), "{reply:?}");
        assert!("Done thinking.".starts_with(reply.reasoning().unwrap_or_default().trim()), "{reply:?}");
    }
    assert_eq!(&encoding.state().messages[..history.len()], history);
    assert_eq!(encoding.state().messages.len(), history.len() + 1);
    assert_eq!(encoding.state().messages.last().unwrap().text().as_deref(), Some("Resumed successfully."));
    assert_eq!(encoding.state().text(), format!("{prompt}{completion}"));
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
        assert_reply_preserves_history(
            &mut encoding,
            &history,
            "Done thinking.</think>\n\nResumed successfully.<|im_end|>",
        );
    }
}

#[test]
fn history_ending_with_assistant_keeps_a_separate_reply() {
    for config in [HanashiConfig::Qwen36, HanashiConfig::Qwen3Instruct] {
        let completion = match config {
            HanashiConfig::Qwen36 => "Done thinking.</think>\n\nResumed successfully.<|im_end|>",
            _ => "Resumed successfully.<|im_end|>",
        };
        let mut encoding = encoding(config);
        // Even a single historical assistant must survive the new generation prompt.
        let history = vec![
            ChatMessage::user().with_text("Original task".into()),
            ChatMessage::assistant().with_text("Summary".into()),
        ];
        encoding.encode(history.clone()).unwrap();
        assert_reply_preserves_history(&mut encoding, &history, completion);

        encoding.reset().unwrap();
        let mut history = compaction_history();
        history[2] = ChatMessage::assistant().with_text("Retained recent assistant turn".into());
        encoding.encode(history.clone()).unwrap();
        assert_reply_preserves_history(&mut encoding, &history, completion);
    }
}

fn assert_merging_parser_keeps_a_separate_reply(
    config: HanashiConfig,
    completion: &str,
) {
    let mut encoding = encoding(config);
    for history in [
        vec![
            ChatMessage::user().with_text("Original task".into()),
            ChatMessage::assistant().with_text("Summary".into()),
        ],
        compaction_history(),
    ] {
        encoding.reset().unwrap();
        encoding.encode(history.clone()).unwrap();
        assert_reply_preserves_history(&mut encoding, &history, completion);
        assert_eq!(encoding.state().messages.last().unwrap().reasoning().as_deref(), Some("Done thinking."));

        // Start another reply on the same parser, preserving the reply just generated too.
        let next_messages = vec![
            ChatMessage::user().with_text("Continue".into()),
            ChatMessage::assistant().with_text("Another retained turn".into()),
        ];
        let mut history = encoding.state().messages.clone();
        history.extend(next_messages.clone());
        encoding.encode(next_messages).unwrap();
        assert_reply_preserves_history(&mut encoding, &history, completion);

        let next_message = ChatMessage::user().with_text("Continue again".into());
        let mut history = encoding.state().messages.clone();
        history.push(next_message.clone());
        encoding.encode(vec![next_message]).unwrap();
        assert_reply_preserves_history(&mut encoding, &history, completion);
    }
}

#[test]
fn gpt_oss_history_ending_with_assistant_keeps_a_separate_reply() {
    assert_merging_parser_keeps_a_separate_reply(
        HanashiConfig::GptOss,
        concat!(
            "<|channel|>analysis<|message|>Done thinking.<|end|>",
            "<|start|>assistant<|channel|>final<|message|>Resumed successfully.<|return|>"
        ),
    );
}

#[test]
fn muse_glimmer_history_ending_with_assistant_keeps_a_separate_reply() {
    assert_merging_parser_keeps_a_separate_reply(
        HanashiConfig::MuseGlimmer,
        concat!(
            " to=self<|message|>Done thinking.<|eom|>",
            "<|start|>assistant to=user<|message|>Resumed successfully.<|eot|>"
        ),
    );
}

#[test]
fn history_ending_with_assistant_preserves_tool_context() {
    let mut encoding = encoding(HanashiConfig::Llama32);
    // Llama's bare-JSON tool-call extraction depends on the parser's `tools` variable.
    for tools_declared in [true, false] {
        encoding.reset().unwrap();
        let mut history = vec![ChatMessage::system().with_text("You are a helpful assistant".into())];
        if tools_declared {
            history.push(ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
                name: "functions".into(),
                description: None,
                tools: vec![ToolDescription::Function {
                    tool_function: ToolFunction {
                        name: "get_current_time".into(),
                        description: "Returns the current time".into(),
                        parameters: None,
                        return_definition: None,
                    },
                }],
            }]));
        }
        history.push(ChatMessage::user().with_text("What time is it?".into()));
        history.push(ChatMessage::assistant().with_text("Retained answer".into()));
        encoding.encode(history.clone()).unwrap();

        let completion = r#"{"name": "get_current_time", "parameters": {"timezone": "UTC"}}<|eom_id|>"#;
        encoding.decode(encoding.tokenize(completion).unwrap()).unwrap();
        assert_eq!(&encoding.state().messages[..history.len()], history);
        assert_eq!(encoding.state().messages.len(), history.len() + 1);
        let reply = encoding.state().messages.last().unwrap();
        let calls = reply.tool_calls();
        if tools_declared {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_current_time");
        } else {
            assert!(calls.is_empty());
            assert!(reply.text().unwrap().contains("\"parameters\""));
        }
    }
}
