use super::*;

#[test]
fn qwen_tool_calls_preserve_hyphenated_function_names() {
    use token_stream_parser::{Parser, token_stream::TokenStreamParser, types::Token};

    use crate::chat::hanashi::{config::HanashiConfig, messages::streamed::Message};

    for config in [HanashiConfig::Qwen35, HanashiConfig::Qwen36, HanashiConfig::Qwen38] {
        for (name, parameters, arguments) in [
            ("ping-pong", "", serde_json::json!({})),
            ("get_weather-v2", "<parameter=city>\nLondon\n</parameter>\n", serde_json::json!({"city": "London"})),
        ] {
            let mut parser = TokenStreamParser::new(config.resolve().unwrap().parsing).unwrap();
            let body = format!("\n<function={name}>\n{parameters}</function>\n");
            let fragments = ["<|im_start|>", "assistant", "\n", "<tool_call>"]
                .into_iter()
                .map(str::to_string)
                .chain(body.chars().map(|value| value.to_string()))
                .chain(["</tool_call>", "<|im_end|>"].into_iter().map(str::to_string));
            for value in fragments {
                parser
                    .push(&Token {
                        id: 0,
                        value,
                        is_special: false,
                    })
                    .unwrap();
            }
            let messages: Vec<Message> = serde_json::from_value(parser.state().value.clone()).unwrap();
            let message: shoji::types::session::chat::ChatMessage = messages.into_iter().last().unwrap().into();
            let calls = message.tool_calls();
            assert_eq!(calls.len(), 1, "{config:?}: {:?}", message.content);
            assert_eq!(calls[0].name, name);
            assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments.json).unwrap(), arguments);
        }
    }
}

#[test]
fn qwen38_exposes_tool_candidates_before_the_call_closes() {
    use token_stream_parser::{Parser, token_stream::TokenStreamParser, types::Token};

    use crate::chat::hanashi::{config::HanashiConfig, messages::streamed::Message};
    let mut parser = TokenStreamParser::new(HanashiConfig::Qwen38.resolve().unwrap().parsing).unwrap();
    for value in ["<|im_start|>", "assistant", "\n", "<tool_call>"] {
        parser
            .push(&Token {
                id: 0,
                value: value.into(),
                is_special: false,
            })
            .unwrap();
    }
    let text = "\n<function=write>\n<parameter=content>\nhello";
    for value in text.chars() {
        parser
            .push(&Token {
                id: 0,
                value: value.to_string(),
                is_special: false,
            })
            .unwrap();
    }
    let messages: Vec<Message> = serde_json::from_value(parser.state().value.clone()).unwrap();
    let blocks = messages.into_iter().last().unwrap().content.unwrap().blocks(&ChatRole::Assistant {});
    let candidate = blocks
        .iter()
        .find_map(|block| match block {
            ChatContentBlock::ToolCallCandidate {
                value,
                ..
            } => Some(value),
            _ => None,
        })
        .expect("Qwen must expose the unfinished tool call");
    let raw: String = serde_json::from_str(&candidate.json).unwrap();
    assert!(raw.contains("<function=write>"));
    assert!(raw.ends_with("hello"));
    parser
        .push(&Token {
            id: 0,
            value: "\n</parameter>\n</function>\n".into(),
            is_special: false,
        })
        .unwrap();
    parser
        .push(&Token {
            id: 0,
            value: "</tool_call>".into(),
            is_special: false,
        })
        .unwrap();
    let messages: Vec<Message> = serde_json::from_value(parser.state().value.clone()).unwrap();
    let message: shoji::types::session::chat::ChatMessage = messages.into_iter().last().unwrap().into();
    let calls = message.tool_calls();
    assert_eq!(calls[0].name, "write");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].arguments.json).unwrap(),
        serde_json::json!({"content": "hello"})
    );
}

#[test]
fn preserves_reasoning_whitespace_but_normalizes_text() {
    let blocks = Content::Sections(vec![
        Section::Reasoning {
            value: Some(" think\n\n".to_string()),
        },
        Section::Text {
            value: Some(" answer \n".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert_eq!(
        blocks,
        vec![
            ChatContentBlock::Reasoning {
                value: " think\n\n".to_string(),
            },
            ChatContentBlock::Text {
                value: "answer".to_string(),
            },
        ]
    );
}

#[test]
fn removes_qwen_template_separators_from_visible_text() {
    let blocks = Content::Sections(vec![
        Section::Text {
            value: Some("\n".to_string()),
        },
        Section::Reasoning {
            value: Some("\nLet me think.\n".to_string()),
        },
        Section::Text {
            value: Some("\n\nHello!".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert_eq!(
        blocks,
        vec![
            ChatContentBlock::Reasoning {
                value: "\nLet me think.\n".to_string(),
            },
            ChatContentBlock::Text {
                value: "Hello!".to_string(),
            },
        ]
    );
}

#[test]
fn drops_whitespace_only_text_sections() {
    let blocks = Content::Sections(vec![
        Section::Reasoning {
            value: Some(" \n".to_string()),
        },
        Section::Text {
            value: Some("\t".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert!(blocks.is_empty());
}
