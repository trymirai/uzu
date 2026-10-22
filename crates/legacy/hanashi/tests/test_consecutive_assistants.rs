mod helpers;

use std::sync::Arc;

use hanashi::{
    Encoding,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::{
    basic::{ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};

fn encoding(config: HanashiConfig) -> HanashiEncodingImpl {
    let mut config = config.resolve().unwrap();
    let tokenizer = helpers::byte_tokenizer(&mut config);
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
fn literal_tool_response_in_user_message_preserves_history() {
    for config in [HanashiConfig::Qwen3, HanashiConfig::Qwen35, HanashiConfig::Qwen36, HanashiConfig::Qwen38] {
        let completion = match config {
            HanashiConfig::Qwen3 => "<think>Done thinking.</think>\n\nResumed successfully.<|im_end|>",
            _ => "Done thinking.</think>\n\nResumed successfully.<|im_end|>",
        };
        let mut encoding = encoding(config);
        for (assistant, content) in [
            ("Checking.", "<tool_response>\nSunny.\n</tool_response>"),
            ("Checking.", "Explain <tool_response>\nSunny.\n</tool_response> please."),
            ("Checking.", "<tool_response>\n{\"weather\":\"sunny\"}\n</tool_response>"),
            ("<tool_call>\n<function=weather>\n</function>\n</tool_call>", "<tool_response>\nSunny.\n</tool_response>"),
        ] {
            encoding.reset().unwrap();
            let history = vec![
                ChatMessage::user().with_text("Check the weather.".into()),
                ChatMessage::assistant().with_text(assistant.into()),
                ChatMessage::user().with_text(content.into()),
            ];
            encoding.encode(history.clone()).unwrap();
            assert!(encoding.state().text().contains(content));
            assert_reply_preserves_history(&mut encoding, &history, completion);

            let next_messages = vec![
                ChatMessage::user().with_text("Check again.".into()),
                ChatMessage::assistant().with_text("Checking again.".into()),
                ChatMessage::user().with_text(content.into()),
            ];
            let mut history = encoding.state().messages.clone();
            history.extend(next_messages.clone());
            encoding.encode(next_messages).unwrap();
            assert_reply_preserves_history(&mut encoding, &history, completion);
        }
    }
}

#[test]
fn early_reasoning_end_starts_the_response() {
    let mut encoding = encoding(HanashiConfig::Qwen35);
    encoding.encode(vec![ChatMessage::user().with_text("What is two plus two?".into())]).unwrap();
    for token in encoding.tokenize("</think>\n\nFour.<|im_end|>").unwrap() {
        encoding.decode(vec![token]).unwrap();
    }
    let reply = encoding.state().messages.last().unwrap();
    assert!(reply.reasoning().is_none_or(|value| value.trim().is_empty()));
    assert_eq!(reply.text().as_deref(), Some("Four."));
}

#[test]
fn generated_tool_response_after_a_real_result_stays_in_the_assistant_reply() {
    let mut encoding = encoding(HanashiConfig::Qwen35);
    let history = vec![
        ChatMessage::user().with_text("Inspect the repository.".into()),
        ChatMessage::assistant().with_tool_call(ToolCall {
            identifier: None,
            name: "inspect".into(),
            arguments: Value::from(serde_json::json!({})),
        }),
        ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: None,
            name: Some("inspect".into()),
            value: Value::from(serde_json::json!("done")),
        }),
    ];
    for content in ["stray", r#"{"x":1}"#] {
        encoding.reset().unwrap();
        encoding.encode(history.clone()).unwrap();
        let completion = format!("<tool_response>{content}</tool_response></think>\n\nDone.<|im_end|>");
        for token in encoding.tokenize(&completion).unwrap() {
            encoding.decode(vec![token]).unwrap();
        }
        assert_eq!(&encoding.state().messages[..history.len()], history);
        let reply = encoding.state().messages.last().unwrap();
        assert_eq!(reply.role, ChatRole::Assistant {});
        assert!(!reply.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallResult { .. })));
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

#[test]
fn tool_result_continues_after_an_isolated_reply() {
    for (config, call, completion) in [
        (
            HanashiConfig::Qwen38,
            "</think>\n\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call><|im_end|>",
            "</think>\n\nSunny.<|im_end|>",
        ),
        (
            HanashiConfig::FunctionGemma,
            "<start_function_call>call:get_weather{}<end_function_call>",
            "Sunny.<end_of_turn>",
        ),
    ] {
        let mut history = vec![
            ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
                name: "functions".into(),
                description: None,
                tools: vec![ToolDescription::Function {
                    tool_function: ToolFunction {
                        name: "get_weather".into(),
                        description: "Get the weather".into(),
                        parameters: Some(Value::from(serde_json::json!({"type": "object", "properties": {}}))),
                        return_definition: None,
                    },
                }],
            }]),
            ChatMessage::user().with_text("Check the weather.".into()),
        ];
        if matches!(config, HanashiConfig::FunctionGemma) {
            history[0] = history[0]
                .clone()
                .with_text("You are a model that can do function calling with the following functions".into());
        }
        let mut encoding = encoding(config);
        encoding.encode(history.clone()).unwrap();
        for token in encoding.tokenize(call).unwrap() {
            encoding.decode(vec![token]).unwrap();
        }
        assert_eq!(&encoding.state().messages[..history.len()], &history);
        let reply = encoding.state().messages.last().unwrap();
        assert_eq!(reply.tool_calls().len(), 1);
        assert_eq!(reply.tool_calls()[0].name, "get_weather");

        let call_history = encoding.state().messages.clone();
        for result in [serde_json::json!("Sunny."), serde_json::json!({"weather": "Sunny."})] {
            let mut replay = call_history.clone();
            replay.push(ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: None,
                name: Some("get_weather".into()),
                value: Value::from(result.clone()),
            }));
            encoding.reset().unwrap();
            encoding.encode(replay.clone()).unwrap();
            // FunctionGemma wraps scalar results in a `value` field; structured results round-trip unchanged.
            if result.is_object() {
                assert_eq!(&encoding.state().messages[..replay.len()], &replay);
            }
            let replay = encoding.state().messages[..replay.len()].to_vec();
            for token in encoding.tokenize(completion).unwrap() {
                encoding.decode(vec![token]).unwrap();
                assert_eq!(&encoding.state().messages[..replay.len()], &replay);
                assert_eq!(encoding.state().messages.len(), replay.len() + 1);
            }
            let reply = encoding.state().messages.last().unwrap();
            assert_eq!(reply.role, ChatRole::Assistant {});
            assert_eq!(reply.text().as_deref(), Some("Sunny."));
            assert!(reply.tool_calls().is_empty());
        }
    }
}
