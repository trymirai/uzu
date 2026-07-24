mod helpers;

use std::collections::HashMap;

use hanashi::{
    Encoding as EncodingTrait,
    chat::{
        Context, Encoding, EncodingConfig, TokenizerLocation, hanashi::config::HanashiConfig, harmony::HarmonyConfig,
    },
};
use helpers::{
    build_messages, load_registry, load_response_test_data, load_tokenizer, normalize_pattern, response_path,
    tokenizer_directory,
};
use shoji::types::{
    basic::{ReasoningEffort, Token, ToolCall, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatMessageMetadata, ChatRole},
};

fn print_warning(message: &str) {
    println!("\x1b[33m{}\x1b[0m", message);
}

fn hanashi(config: HanashiConfig) -> EncodingConfig {
    EncodingConfig::Hanashi {
        config,
    }
}

fn harmony(config: HarmonyConfig) -> EncodingConfig {
    EncodingConfig::Harmony {
        config,
    }
}

fn run_encoding_test(
    expected_config: &EncodingConfig,
    date_pattern: Option<&str>,
    tools_pattern: Option<&str>,
    ignore_prompt: bool,
    ignore_reasoning: bool,
    messages_normalizer: Option<fn(String, Option<&str>, Vec<ChatMessage>) -> Vec<ChatMessage>>,
) {
    let registry = load_registry();

    for model in &registry {
        let has_encoding = model.encodings.iter().any(|encoding_value| {
            serde_json::from_value::<EncodingConfig>(encoding_value.clone())
                .map(|config| &config == expected_config)
                .unwrap_or(false)
        });
        if !has_encoding {
            continue;
        }
        let tokenizer_directory = tokenizer_directory(&model.name());
        if !tokenizer_directory.exists() {
            continue;
        }
        let response_path = response_path(&model.name());
        if !response_path.exists() {
            continue;
        }
        println!("Validation of {}", model.repo_id);

        let tokenizer = load_tokenizer(&model.name());
        let test_data = load_response_test_data(&model.name());
        let context = Context {
            tokenizer_location: TokenizerLocation::Directory {
                path: tokenizer_directory.to_str().unwrap().to_string(),
                name: None,
            },
        };
        let mut encoding = Encoding::new(expected_config.clone(), context).unwrap();

        for data in &test_data {
            let messages = match messages_normalizer {
                Some(normalize) => normalize(data.result.prompt.clone(), date_pattern, build_messages(data)),
                None => build_messages(data),
            };
            let completion_token_ids =
                tokenizer.encode(data.result.completion.as_str(), false).unwrap().get_ids().to_vec();

            encoding.reset().unwrap();
            encoding.encode(messages).unwrap();
            let prompt = encoding.state().text();
            let prompt = match date_pattern {
                Some(pattern) => normalize_pattern(&prompt, &data.result.prompt, pattern),
                None => prompt,
            };
            let prompt = match tools_pattern {
                Some(pattern) => normalize_pattern(&prompt, &data.result.prompt, pattern),
                None => prompt,
            };
            if ignore_prompt {
                if prompt != data.result.prompt {
                    print_warning(&format!(
                        "Prompt mismatch\nProvided: {:?}\nExpected: {:?}\n",
                        prompt, data.result.prompt
                    ));
                }
            } else {
                assert_eq!(prompt, data.result.prompt);
            }

            for token_id in completion_token_ids {
                encoding.decode(vec![token_id]).unwrap();
            }
            let assistant_message = encoding.state().messages.last().unwrap();
            let has_reasoning =
                assistant_message.content.iter().any(|block| matches!(block, ChatContentBlock::Reasoning { .. }));
            let used_tool_calls = assistant_message
                .content
                .iter()
                .flat_map(|block| match block {
                    ChatContentBlock::ToolCall {
                        value,
                    } => Some(value.name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(assistant_message.role, ChatRole::Assistant {});
            if ignore_reasoning {
                if has_reasoning != data.expectations.reasoning {
                    print_warning("Reasoning mismatch");
                }
            } else {
                assert_eq!(has_reasoning, data.expectations.reasoning);
            }
            if used_tool_calls != data.expectations.tool_call_names {
                print_warning(&format!(
                    "Tool calls mismatch\nProvided: {:?}\nExpected: {:?}\nUsed: {:?}\nCompletion: {}\nReqiest: {:#?}\nMessage: {:#?}\n",
                    data.parameters.tool_names(),
                    data.expectations.tool_call_names,
                    used_tool_calls,
                    data.result.completion,
                    data.parameters.messages,
                    assistant_message,
                ));
            }
        }
    }
}

#[test]
fn test_encoding_gpt_oss() {
    let date_pattern = Some(r"Current date: (\d{4}-\d{2}-\d{2})");
    let tools_pattern = Some(r"(?s)# Tools\n\n.*?} // namespace functions");

    run_encoding_test(&hanashi(HanashiConfig::GptOss), date_pattern, None, false, false, None);

    run_encoding_test(
        &harmony(HarmonyConfig::GptOss),
        date_pattern,
        tools_pattern,
        false,
        false,
        Some(|prompt, date_pattern, messages| {
            let date_regex = regex::Regex::new(date_pattern.unwrap()).unwrap();
            let conversation_date = date_regex
                .captures(&prompt)
                .and_then(|captures| captures.get(1))
                .map(|matched| matched.as_str().to_string())
                .unwrap();

            let original_system_message = messages.iter().find(|message| message.role == ChatRole::System {}).cloned();
            let original_developer_message =
                messages.iter().find(|message| message.role == ChatRole::Developer {}).cloned();
            let other_messages = messages
                .into_iter()
                .filter(|message| message.role != ChatRole::System {} && message.role != ChatRole::Developer {})
                .collect::<Vec<_>>();

            let system_message = ChatMessage {
                role: ChatRole::System {},
                content: vec![
                    ChatContentBlock::ReasoningEffort {
                        value: ReasoningEffort::Medium,
                    },
                    ChatContentBlock::ConversationStartDate {
                        value: conversation_date,
                    },
                ],
                metadata: ChatMessageMetadata {
                    values: HashMap::new(),
                },
            };
            let mut developer_message = ChatMessage {
                role: ChatRole::Developer {},
                content: vec![],
                metadata: ChatMessageMetadata {
                    values: HashMap::new(),
                },
            };
            if let Some(original_developer_message) = original_developer_message {
                for block in original_developer_message.content {
                    let new_block = block.clone();
                    if let ChatContentBlock::Tools {
                        namespaces,
                    } = block
                        && namespaces.iter().any(|namespace| !namespace.tools.is_empty())
                    {
                        developer_message.content.push(new_block);
                    }
                }
            }
            if let Some(original_system_message) = original_system_message {
                for block in original_system_message.content {
                    if let ChatContentBlock::Text {
                        value,
                    } = block
                    {
                        developer_message.content.push(ChatContentBlock::Text {
                            value: format!(
                                "{}{}",
                                value,
                                if developer_message.content.is_empty() {
                                    "\n\n"
                                } else {
                                    ""
                                }
                            )
                            .to_string(),
                        });
                        break;
                    }
                }
            }

            let mut normalized_messages = vec![system_message];
            if !developer_message.content.is_empty() {
                normalized_messages.push(developer_message);
            }
            normalized_messages.extend(other_messages);
            normalized_messages
        }),
    );
}

/// Multi-turn tool calling: prior tool calls re-rendered into the prompt must match the reference
/// gpt-oss chat template byte for byte (`to=` before the channel, plain `json` content type, the
/// message closed by `<|call|>`) — the form the model expects to see in its context. Deviations
/// (e.g. `<|constrain|>json` rendered as a special token) push the follow-up turn out of
/// distribution and make the model more likely to emit malformed tool-call headers.
#[test]
fn test_encoding_gpt_oss_tool_call_rerender() {
    let tokenizer_directory = tokenizer_directory("openai_gpt-oss-20b");
    if !tokenizer_directory.exists() {
        return;
    }

    let context = Context {
        tokenizer_location: TokenizerLocation::Directory {
            path: tokenizer_directory.to_str().unwrap().to_string(),
            name: None,
        },
    };
    let mut encoding = Encoding::new(harmony(HarmonyConfig::GptOss), context).unwrap();

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()).with_block(
            ChatContentBlock::ConversationStartDate {
                value: "2026-07-23".to_string(),
            },
        ),
        ChatMessage::user().with_text("What time is it now?".to_string()),
        ChatMessage::assistant().with_reasoning("Need the current time.".to_string()).with_tool_call(ToolCall {
            identifier: None,
            name: "get_current_time".to_string(),
            arguments: Value::from(serde_json::json!({})),
        }),
        ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: None,
            name: Some("get_current_time".to_string()),
            value: Value::from(serde_json::json!({"time": "17:03"})),
        }),
    ];
    encoding.encode(messages).unwrap();

    let expected = concat!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n",
        "Knowledge cutoff: 2024-06\n",
        "Current date: 2026-07-23\n",
        "\n",
        "Reasoning: medium\n",
        "\n",
        "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
        "<|start|>developer<|message|># Instructions\n",
        "\n",
        "You are a helpful assistant<|end|>",
        "<|start|>user<|message|>What time is it now?<|end|>",
        "<|start|>assistant<|channel|>analysis<|message|>Need the current time.<|end|>",
        "<|start|>assistant to=functions.get_current_time<|channel|>commentary json<|message|>{}<|call|>",
        "<|start|>functions.get_current_time to=assistant<|channel|>commentary<|message|>{\"time\":\"17:03\"}<|end|>",
        "<|start|>assistant",
    );
    assert_eq!(encoding.state().text(), expected);
}

/// gpt-oss sometimes drops or misplaces the `<|channel|>` token when it opens a tool-call header
/// right after a tool response (e.g. `commentary to=functions.x<|channel|>commentary
/// <|constrain|>json<|message|>` straight after `<|start|>assistant`). The decoder must repair
/// such headers instead of failing the whole turn, while keeping the parser strict for anything
/// it cannot attribute.
#[test]
fn test_decoding_gpt_oss_malformed_tool_call_header() {
    let tokenizer_directory = tokenizer_directory("openai_gpt-oss-20b");
    if !tokenizer_directory.exists() {
        return;
    }
    let tokenizer = load_tokenizer("openai_gpt-oss-20b");

    let context = Context {
        tokenizer_location: TokenizerLocation::Directory {
            path: tokenizer_directory.to_str().unwrap().to_string(),
            name: None,
        },
    };
    let mut encoding = Encoding::new(harmony(HarmonyConfig::GptOss), context).unwrap();

    let cases: &[(&str, Option<&str>)] = &[
        // the exact quirk observed from gpt-oss-20b: channel name first, marker misplaced
        (
            "commentary to=functions.get_current_location<|channel|>commentary <|constrain|>json<|message|>{}<|call|>",
            Some("get_current_location"),
        ),
        // channel marker missing entirely
        ("commentary to=functions.get_current_time <|constrain|>json<|message|>{}<|call|>", Some("get_current_time")),
        // plain-text content type, no marker
        ("commentary to=functions.get_current_time json<|message|>{}<|call|>", Some("get_current_time")),
        // well-formed headers must pass through untouched
        (
            "<|channel|>commentary to=functions.get_current_time <|constrain|>json<|message|>{}<|call|>",
            Some("get_current_time"),
        ),
        (
            "<|channel|>analysis<|message|>Need the time.<|end|>\
             <|start|>assistant to=functions.get_current_time<|channel|>commentary <|constrain|>json<|message|>{}<|call|>",
            Some("get_current_time"),
        ),
        ("<|channel|>final<|message|>It is 17:03.<|return|>", None),
    ];

    for (completion, expected_tool_call) in cases {
        encoding.reset().unwrap();
        encoding.encode(vec![ChatMessage::user().with_text("What time is it now?".to_string())]).unwrap();

        let token_ids = tokenizer.encode(*completion, false).unwrap().get_ids().to_vec();
        for token_id in token_ids {
            encoding.decode(vec![token_id]).unwrap_or_else(|error| panic!("Failed to decode {completion:?}: {error}"));
        }

        let assistant_message = encoding.state().messages.last().unwrap();
        assert_eq!(assistant_message.role, ChatRole::Assistant {}, "for completion {completion:?}");
        let tool_call_names = assistant_message
            .content
            .iter()
            .filter_map(|block| match block {
                ChatContentBlock::ToolCall {
                    value,
                } => Some(value.name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        match expected_tool_call {
            Some(name) => assert_eq!(tool_call_names, vec![*name], "for completion {completion:?}"),
            None => assert!(tool_call_names.is_empty(), "for completion {completion:?}"),
        }
    }

    // a header the repair cannot attribute must still fail strict parsing
    encoding.reset().unwrap();
    encoding.encode(vec![ChatMessage::user().with_text("What time is it now?".to_string())]).unwrap();
    let garbled = "commentary nonsense to=functions.get_current_time<|message|>{}<|call|>";
    let token_ids = tokenizer.encode(garbled, false).unwrap().get_ids().to_vec();
    let result = token_ids.into_iter().try_for_each(|token_id| encoding.decode(vec![token_id]));
    assert!(result.is_err(), "Expected garbled header {garbled:?} to fail decoding");
}

#[test]
fn test_encoding_llama_32() {
    run_encoding_test(&hanashi(HanashiConfig::Llama32), Some(r"Today Date: (.+?)\n"), None, false, false, None);
}

/// After a tool response, Llama 3.2 1B sometimes replies by echoing the result JSON as a new
/// pseudo tool call (`<|python_tag|>{"time": ...}<|eom_id|>`). Once such a section finishes it
/// parses into an object that is not a tool call; it must surface as plain text so the turn ends
/// with a normal reply instead of dead-ending on a tool-call candidate.
#[test]
fn test_decoding_llama_32_malformed_tool_call() {
    let tokenizer_directory = tokenizer_directory("meta-llama_Llama-3.2-1B-Instruct");
    if !tokenizer_directory.exists() {
        return;
    }
    let tokenizer = load_tokenizer("meta-llama_Llama-3.2-1B-Instruct");

    let context = Context {
        tokenizer_location: TokenizerLocation::Directory {
            path: tokenizer_directory.to_str().unwrap().to_string(),
            name: None,
        },
    };
    let mut encoding = Encoding::new(hanashi(HanashiConfig::Llama32), context).unwrap();

    // (completion, expected tool call, expected text fragment)
    let cases: &[(&str, Option<&str>, Option<&str>)] = &[
        // proper tool call: unchanged behavior
        (
            "<|python_tag|>{\"name\": \"get_current_time\", \"parameters\": {}}<|eom_id|>",
            Some("get_current_time"),
            None,
        ),
        // tool-result echo: must become text, not a tool-call candidate
        ("<|python_tag|>{\"time\": \"17:03\", \"return\": \"time\"}<|eom_id|>", None, Some("17:03")),
    ];

    for (completion, expected_tool_call, expected_fragment) in cases {
        encoding.reset().unwrap();
        encoding
            .encode(vec![
                ChatMessage::system().with_text("You are a helpful assistant".to_string()),
                ChatMessage::user().with_text("What is the time now?".to_string()),
            ])
            .unwrap();

        let token_ids = tokenizer.encode(*completion, false).unwrap().get_ids().to_vec();
        for token_id in token_ids {
            encoding.decode(vec![token_id]).unwrap_or_else(|error| panic!("Failed to decode {completion:?}: {error}"));
        }

        let assistant_message = encoding.state().messages.last().unwrap();
        assert_eq!(assistant_message.role, ChatRole::Assistant {}, "for completion {completion:?}");
        let tool_call_names = assistant_message.tool_calls().iter().map(|call| call.name.clone()).collect::<Vec<_>>();
        let has_candidates =
            assistant_message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
        match expected_tool_call {
            Some(name) => assert_eq!(tool_call_names, vec![name.to_string()], "for completion {completion:?}"),
            None => {
                assert!(tool_call_names.is_empty(), "for completion {completion:?}");
                assert!(!has_candidates, "Expected no tool-call candidates for completion {completion:?}");
            },
        }
        if let Some(fragment) = expected_fragment {
            let text = assistant_message.text().unwrap_or_default();
            assert!(
                text.contains(fragment),
                "Expected text with {fragment:?} for completion {completion:?}, got {text:?}"
            );
        }
    }
}

#[test]
fn test_encoding_qwen3() {
    run_encoding_test(&hanashi(HanashiConfig::Qwen3), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen3_instruct() {
    run_encoding_test(&hanashi(HanashiConfig::Qwen3Instruct), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen3_thinking() {
    run_encoding_test(&hanashi(HanashiConfig::Qwen3Thinking), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen35() {
    run_encoding_test(&hanashi(HanashiConfig::Qwen35), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm2() {
    run_encoding_test(&hanashi(HanashiConfig::Lfm2), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm25_instruct() {
    run_encoding_test(&hanashi(HanashiConfig::Lfm25Instruct), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm25_thinking() {
    run_encoding_test(&hanashi(HanashiConfig::Lfm25Thinking), None, None, false, false, None);
}

#[test]
fn test_encoding_gemma_3() {
    run_encoding_test(&hanashi(HanashiConfig::Gemma3), None, None, false, false, None);
}

#[test]
fn test_encoding_gemma_4() {
    run_encoding_test(&hanashi(HanashiConfig::Gemma4), None, None, false, true, None);
}

#[test]
fn test_encoding_functiongemma() {
    run_encoding_test(
        &hanashi(HanashiConfig::FunctionGemma),
        None,
        None,
        false,
        false,
        Some(|_, _, messages| {
            let mut result: Vec<ChatMessage> = Vec::new();
            for mut message in messages {
                if matches!(message.role, ChatRole::System {}) {
                    message.role = ChatRole::Developer {};
                }
                if let Some(last) = result.last_mut()
                    && matches!(last.role, ChatRole::Developer {})
                    && matches!(message.role, ChatRole::Developer {})
                {
                    last.content.extend(message.content);
                    continue;
                }
                result.push(message);
            }
            result
        }),
    );
}

#[test]
fn test_rendering_functiongemma_non_object_tool_results() {
    let config = HanashiConfig::FunctionGemma.resolve().unwrap();
    let renderer = hanashi::chat::hanashi::renderer::Renderer::new(config.rendering);
    let cases = [
        (serde_json::json!(42), "response:get_value{value:42}"),
        (
            serde_json::json!(["first", "second"]),
            "response:get_value{value:[<escape>first<escape>,<escape>second<escape>]}",
        ),
    ];

    for (value, expected) in cases {
        let message = ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: None,
            name: Some("get_value".to_string()),
            value: Value::from(value),
        });
        let rendered = renderer
            .render(
                &[message],
                true,
                Some(Token {
                    id: 0,
                    value: "<bos>".to_string(),
                    is_special: true,
                }),
                None,
                None,
            )
            .unwrap();

        assert!(rendered.contains(expected), "rendered prompt did not preserve the tool result: {rendered}");
    }
}
