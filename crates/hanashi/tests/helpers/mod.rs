#![allow(dead_code)]

use std::{collections::HashMap, path::PathBuf};

use serde::Deserialize;
use shoji::types::{
    basic::{ReasoningEffort, ToolDescription, ToolNamespace, TranslationPayload},
    session::chat::{ChatContentBlock, ChatMessage, ChatMessageMetadata, ChatRole},
};
use tokenizers::Tokenizer;

fn test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("workspace").join("data")
}

fn configs_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs")
}

#[derive(Deserialize, Clone, Debug)]
pub struct ResponseTestParameters {
    pub messages: Vec<serde_json::Value>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub context: HashMap<String, serde_json::Value>,
}

impl ResponseTestParameters {
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.as_ref().unwrap().iter().map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect()
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ResponseTestResult {
    pub prompt: String,
    pub completion: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ResponseTestExpectations {
    pub reasoning: bool,
    pub tool_call_names: Vec<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ResponseTestData {
    pub repo_id: String,
    pub parameters: ResponseTestParameters,
    pub result: ResponseTestResult,
    pub expectations: ResponseTestExpectations,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RegistryModel {
    pub repo_id: String,
    pub encodings: Vec<serde_json::Value>,
}

impl RegistryModel {
    pub fn name(&self) -> String {
        self.repo_id.replace("/", "_")
    }
}

pub fn load_registry() -> Vec<RegistryModel> {
    let path = test_data_path().join("registry.json");
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("Failed to read {}: {error}", path.display()));
    serde_json::from_str(&content).unwrap_or_else(|error| panic!("Failed to parse {}: {error}", path.display()))
}

pub fn response_path(model_name: &str) -> PathBuf {
    test_data_path().join("responses").join(format!("{model_name}.json"))
}

pub fn load_response_test_data(model_name: &str) -> Vec<ResponseTestData> {
    let path = response_path(model_name);
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("Failed to read {}: {error}", path.display()));
    serde_json::from_str(&content).unwrap_or_else(|error| panic!("Failed to parse {}: {error}", path.display()))
}

pub fn tokenizer_directory(model_name: &str) -> PathBuf {
    test_data_path().join("tokenizers").join(model_name)
}

pub fn load_tokenizer(model_name: &str) -> Tokenizer {
    let path = tokenizer_directory(model_name).join("tokenizer.json");
    Tokenizer::from_file(path.to_str().unwrap())
        .unwrap_or_else(|error| panic!("Failed to load tokenizer {}: {error}", path.display()))
}

fn resolve_reasoning_effort(context: &HashMap<String, serde_json::Value>) -> Option<ReasoningEffort> {
    let value = context.get("enable_thinking")?;
    match value.as_bool() {
        Some(true) => Some(ReasoningEffort::Default),
        Some(false) => Some(ReasoningEffort::Disabled),
        _ => None,
    }
}

fn build_system_message(
    raw_messages: &[serde_json::Value],
    reasoning_effort: Option<ReasoningEffort>,
) -> Option<ChatMessage> {
    let system_raw = raw_messages.iter().find(|raw| raw["role"].as_str() == Some("system"));

    let mut content = Vec::new();
    if let Some(raw) = system_raw {
        content.push(ChatContentBlock::Text {
            value: raw["content"].as_str().unwrap().to_string(),
        });
    }
    if let Some(effort) = reasoning_effort {
        content.push(ChatContentBlock::ReasoningEffort {
            value: effort,
        });
    }
    if content.is_empty() {
        return None;
    }

    Some(ChatMessage {
        role: ChatRole::System {},
        content,
        metadata: ChatMessageMetadata {
            values: HashMap::new(),
        },
    })
}

fn build_developer_message(tools: &Option<Vec<serde_json::Value>>) -> Option<ChatMessage> {
    let tools = tools.as_ref()?;
    let tool_descriptions: Vec<ToolDescription> =
        tools.iter().filter_map(|tool| serde_json::from_value(tool.clone()).ok()).collect();

    Some(ChatMessage {
        role: ChatRole::Developer {},
        content: vec![ChatContentBlock::Tools {
            namespaces: vec![ToolNamespace {
                name: "functions".to_string(),
                description: None,
                tools: tool_descriptions,
            }],
        }],
        metadata: ChatMessageMetadata {
            values: HashMap::new(),
        },
    })
}

fn build_user_content(raw: &serde_json::Value) -> Vec<ChatContentBlock> {
    if let Some(text) = raw["content"].as_str() {
        return vec![ChatContentBlock::Text {
            value: text.to_string(),
        }];
    }
    if let Some(items) = raw["content"].as_array() {
        return items
            .iter()
            .map(|item| {
                let content_type = item["type"].as_str().unwrap();
                match content_type {
                    "text" => ChatContentBlock::Translation {
                        payload: TranslationPayload::Text {
                            text: item["text"].as_str().unwrap().to_string(),
                        },
                        source_language_code: item["source_lang_code"].as_str().unwrap().to_string(),
                        target_language_code: item["target_lang_code"].as_str().unwrap().to_string(),
                    },
                    "image" => ChatContentBlock::Translation {
                        payload: TranslationPayload::Image {
                            url: item["url"].as_str().unwrap().to_string(),
                        },
                        source_language_code: item["source_lang_code"].as_str().unwrap().to_string(),
                        target_language_code: item["target_lang_code"].as_str().unwrap().to_string(),
                    },
                    other => {
                        panic!("Unknown translation content type: {other}")
                    },
                }
            })
            .collect();
    }
    panic!("User content must be a string or array");
}

fn build_user_messages(raw_messages: &[serde_json::Value]) -> Vec<ChatMessage> {
    raw_messages
        .iter()
        .filter(|raw| raw["role"].as_str() == Some("user"))
        .map(|raw| ChatMessage {
            role: ChatRole::User {},
            content: build_user_content(raw),
            metadata: ChatMessageMetadata {
                values: HashMap::new(),
            },
        })
        .collect()
}

pub fn build_messages(data: &ResponseTestData) -> Vec<ChatMessage> {
    let raw_messages = &data.parameters.messages;
    let reasoning_effort = resolve_reasoning_effort(&data.parameters.context);

    let mut used_count = 0;
    let mut messages = Vec::new();

    if let Some(system_message) = build_system_message(raw_messages, reasoning_effort) {
        if raw_messages.iter().any(|raw| raw["role"].as_str() == Some("system")) {
            used_count += 1;
        }
        messages.push(system_message);
    }

    if let Some(developer_message) = build_developer_message(&data.parameters.tools) {
        messages.push(developer_message);
    }

    let user_messages = build_user_messages(raw_messages);
    used_count += user_messages.len();
    messages.extend(user_messages);

    assert_eq!(used_count, raw_messages.len(), "Not all raw messages were used");

    messages
}

pub fn normalize_pattern(
    result: &str,
    expected: &str,
    pattern: &str,
) -> String {
    let regex = regex::Regex::new(pattern).unwrap();
    let expected_match = regex.find(expected).map(|matched| matched.as_str());
    let result_match = regex.find(result).map(|matched| matched.as_str());
    match (expected_match, result_match) {
        (Some(expected), Some(actual)) => result.replace(actual, expected),
        _ => result.to_string(),
    }
}
