mod helpers;

use std::collections::HashMap;

use hanashi::{
    Encoding as EncodingTrait,
    chat::{
        Config as EncodingConfig, Context, Encoding, TokenizerLocation,
        hanashi::Config as HanashiConfig,
        harmony::{Config as HarmonyConfig, EncodingName as HarmonyEncodingName},
    },
};
use helpers::{
    build_messages, load_registry, load_response_test_data, load_tokenizer, normalize_pattern, response_path,
    tokenizer_directory,
};
use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatReasoningEffort, ChatRole};

fn print_warning(message: &str) {
    println!("\x1b[33m{}\x1b[0m", message);
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

    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::GptOss), date_pattern, None, false, false, None);

    run_encoding_test(
        &EncodingConfig::Harmony(HarmonyConfig {
            encoding_name: HarmonyEncodingName::GptOss,
        }),
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
                        value: ChatReasoningEffort::Medium,
                    },
                    ChatContentBlock::ConversationStartDate {
                        value: conversation_date,
                    },
                ],
                metadata: HashMap::new(),
            };
            let mut developer_message = ChatMessage {
                role: ChatRole::Developer {},
                content: vec![],
                metadata: HashMap::new(),
            };
            if let Some(original_developer_message) = original_developer_message {
                for block in original_developer_message.content {
                    let new_block = block.clone();
                    match block {
                        ChatContentBlock::Tools {
                            namespaces,
                        } => {
                            if namespaces.iter().any(|namespace| !namespace.tools.is_empty()) {
                                developer_message.content.push(new_block);
                            }
                        },
                        _ => {},
                    }
                }
            }
            if let Some(original_system_message) = original_system_message {
                for block in original_system_message.content {
                    match block {
                        ChatContentBlock::Text {
                            value,
                        } => {
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
                        },
                        _ => {},
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

#[test]
fn test_encoding_llama_31() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Llama31), None, None, false, false, None);
}

#[test]
fn test_encoding_llama_32() {
    run_encoding_test(
        &EncodingConfig::Hanashi(HanashiConfig::Llama32),
        Some(r"Today Date: (.+?)\n"),
        None,
        false,
        false,
        None,
    );
}

#[test]
fn test_encoding_qwen25() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen25), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen25_coder() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen25Coder), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen3() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen3), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen3_instruct() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen3Instruct), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen3_thinking() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen3Thinking), None, None, false, false, None);
}

#[test]
fn test_encoding_qwen35() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Qwen35), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm2() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Lfm2), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm25_instruct() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Lfm25Instruct), None, None, false, false, None);
}

#[test]
fn test_encoding_lfm25_thinking() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Lfm25Thinking), None, None, false, false, None);
}

#[test]
fn test_encoding_deepseek_r1_distill_qwen() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::DeepseekR1DistillQwen), None, None, false, false, None);
}

#[test]
fn test_encoding_rnj_1() {
    run_encoding_test(
        &EncodingConfig::Hanashi(HanashiConfig::Rnj1),
        None,
        Some(r"(?s)<tools>\n.*?</tools>"),
        false,
        false,
        None,
    );
}

#[test]
fn test_encoding_gemma_2() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Gemma2), None, None, false, false, None);
}

#[test]
fn test_encoding_gemma_3() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Gemma3), None, None, false, false, None);
}

#[test]
fn test_encoding_functiongemma() {
    run_encoding_test(
        &EncodingConfig::Hanashi(HanashiConfig::FunctionGemma),
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
                if let Some(last) = result.last_mut() {
                    if matches!(last.role, ChatRole::Developer {}) && matches!(message.role, ChatRole::Developer {}) {
                        last.content.extend(message.content);
                        continue;
                    }
                }
                result.push(message);
            }
            result
        }),
    );
}

#[test]
fn test_encoding_translategemma() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::TranslateGemma), None, None, false, false, None);
}

#[test]
fn test_encoding_gemma_4() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Gemma4), None, None, false, true, None);
}

#[test]
fn test_encoding_smollm2() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::SmolLm2), None, None, false, false, None);
}

#[test]
fn test_encoding_smollm3() {
    run_encoding_test(
        &EncodingConfig::Hanashi(HanashiConfig::SmolLm3),
        Some(r"Today Date: (.+?)\n"),
        Some(r"(?s)<tools>\n.*?</tools>"),
        false,
        true,
        None,
    );
}

#[test]
fn test_encoding_codestral() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Codestral), None, None, false, false, None);
}

#[test]
fn test_encoding_polaris() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Polaris), None, None, false, false, None);
}

#[test]
fn test_encoding_nanbeige() {
    run_encoding_test(&EncodingConfig::Hanashi(HanashiConfig::Nanbeige), None, None, true, false, None);
}
