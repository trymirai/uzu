#![allow(dead_code)]

use std::path::PathBuf;

use serde_json::Value;
use token_stream_parser::{
    Parser, ParserState,
    extraction::ExtractionParserState,
    framing::FramingParserState,
    reduction::ReductionParserState,
    token_stream::{TokenStreamParser, TokenStreamParserConfig},
    types::Token,
};
use tokenizers::Tokenizer;
use tracing_subscriber::{EnvFilter, Registry, prelude::*};
use tracing_tree::HierarchicalLayer;

fn test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("workspace").join("data")
}

fn configs_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("hanashi").join("configs")
}

pub struct TestData {
    pub prompt: String,
    pub completion: String,
}

pub struct TestExpectations {
    pub framing: Option<FramingParserState>,
    pub reduction: Option<ReductionParserState>,
    pub extraction: ExtractionParserState,
}

pub struct TestSuite {
    pub config: TokenStreamParserConfig,
    pub tokenizer: Tokenizer,
}

impl TestSuite {
    pub fn load(
        config_name: &str,
        model_name: &str,
    ) -> Self {
        let parsing_path = configs_path().join("parsing");

        let config_path = parsing_path.join(format!("{config_name}.json"));
        let config_content = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|error| panic!("Failed to read config {}: {error}", config_path.display()));
        let config: TokenStreamParserConfig = serde_json::from_str(&config_content)
            .unwrap_or_else(|error| panic!("Failed to parse config {}: {error}", config_path.display()));

        let tokenizer_path = test_data_path().join("tokenizers").join(model_name).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|error| panic!("Failed to load tokenizer {}: {error}", tokenizer_path.display()));

        Self {
            config,
            tokenizer,
        }
    }

    pub fn create_parser(&self) -> TokenStreamParser {
        TokenStreamParser::new(self.config.clone()).unwrap()
    }

    pub fn token(
        &self,
        value: &str,
    ) -> Token {
        let tokens = self.tokenize(value);
        assert_eq!(tokens.len(), 1, "Expected single token for value: {value}, got {} tokens", tokens.len());
        tokens.into_iter().next().unwrap()
    }

    pub fn tokenize(
        &self,
        text: &str,
    ) -> Vec<Token> {
        let encoding = self.tokenizer.encode(text, false).unwrap_or_else(|error| panic!("Failed to tokenize: {error}"));
        encoding
            .get_ids()
            .iter()
            .map(|token_id| Token {
                id: *token_id,
                value: self
                    .tokenizer
                    .decode(&[*token_id], false)
                    .unwrap_or_else(|error| panic!("Failed to decode token {token_id}: {error}")),
                is_special: self
                    .tokenizer
                    .id_to_token(*token_id)
                    .map(|raw| self.tokenizer.get_added_vocabulary().is_special_token(&raw))
                    .unwrap_or(false),
            })
            .collect()
    }

    pub fn expect_framing(
        &self,
        json: &str,
    ) -> FramingParserState {
        let mut value: Value =
            serde_json::from_str(json).unwrap_or_else(|error| panic!("Failed to parse framing expectation: {error}"));
        self.resolve_tokens_in_value(&mut value);
        serde_json::from_value(value).unwrap_or_else(|error| panic!("Failed to deserialize framing state: {error}"))
    }

    pub fn expect_reduction(
        &self,
        json: &str,
    ) -> ReductionParserState {
        let mut value: Value =
            serde_json::from_str(json).unwrap_or_else(|error| panic!("Failed to parse reduction expectation: {error}"));
        self.resolve_tokens_in_value(&mut value);
        mark_all_groups_finished(&mut value);
        serde_json::from_value(value).unwrap_or_else(|error| panic!("Failed to deserialize reduction state: {error}"))
    }

    pub fn expect_extraction(
        &self,
        value: Value,
    ) -> ExtractionParserState {
        ExtractionParserState {
            value,
        }
    }

    fn resolve_tokens_in_value(
        &self,
        value: &mut Value,
    ) {
        match value {
            Value::Object(map) => {
                for (key, field) in map.iter_mut() {
                    match key.as_str() {
                        "name" => {},
                        "marker" | "open" | "close" => {
                            if let Value::String(token_value) = field {
                                *field = serde_json::to_value(&self.token(token_value)).unwrap();
                            }
                        },
                        "text" => {
                            if let Value::Array(items) = field {
                                for item in items.iter_mut() {
                                    if let Value::String(token_value) = item {
                                        *item = serde_json::to_value(&self.token(token_value)).unwrap();
                                    }
                                }
                            }
                        },
                        _ => self.resolve_tokens_in_value(field),
                    }
                }
            },
            Value::Array(items) => {
                for item in items.iter_mut() {
                    self.resolve_tokens_in_value(item);
                }
            },
            _ => {},
        }
    }
}

fn mark_all_groups_finished(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.contains_key("sections") {
                map.insert("finished".to_string(), Value::Bool(true));
            }
            for (_, field) in map.iter_mut() {
                mark_all_groups_finished(field);
            }
        },
        Value::Array(items) => {
            for item in items.iter_mut() {
                mark_all_groups_finished(item);
            }
        },
        _ => {},
    }
}

pub fn run_parser_test(
    suite: &TestSuite,
    data: &TestData,
    expectations: &TestExpectations,
) {
    let mut parser = suite.create_parser();
    let full_text = format!("{}{}", data.prompt, data.completion);
    let tokens = suite.tokenize(&full_text);

    for (index, token) in tokens.iter().enumerate() {
        parser.push(token).unwrap();

        let expected_tokens: Vec<&Token> = tokens[..=index].iter().collect();
        let framing_tokens = parser.framing().state().tokens();
        let reduction_tokens = parser.reduction().state().tokens();
        assert_eq!(framing_tokens, expected_tokens, "Framing tokens mismatch after token {:?}", token.value,);
        assert_eq!(reduction_tokens, expected_tokens, "Reduction tokens mismatch after token {:?}", token.value,);

        if let Some(framing) = &expectations.framing {
            assert!(
                parser.framing().state().is_substate_of(framing),
                "Framing state after token {:?} is not a substate of expected",
                token.value,
            );
        }
        if let Some(reduction) = &expectations.reduction {
            assert!(
                parser.reduction().state().is_substate_of(reduction),
                "Reduction state after token {:?} is not a substate of expected",
                token.value,
            );
        }
        assert!(
            parser.state().is_substate_of(&expectations.extraction),
            "Extraction state after token {:?} is not a substate of expected.\n  Got: {}",
            token.value,
            serde_json::to_string_pretty(&parser.state().value).unwrap()
        );

        println!("Result: {}", serde_json::to_string_pretty(&parser.state().value).unwrap());
    }

    let expected_tokens: Vec<&Token> = tokens.iter().collect();
    assert_eq!(parser.framing().state().tokens(), expected_tokens, "Final framing tokens mismatch");
    assert_eq!(parser.reduction().state().tokens(), expected_tokens, "Final reduction tokens mismatch");

    if let Some(framing) = &expectations.framing {
        assert_eq!(parser.framing().state(), framing, "Final framing state mismatch");
    }
    if let Some(reduction) = &expectations.reduction {
        assert_eq!(parser.reduction().state(), reduction, "Final reduction state mismatch");
    }
    assert_eq!(parser.state().value, expectations.extraction.value, "Final extraction state mismatch");
}

pub fn init_tracing_for_tests() {
    let filter = EnvFilter::from_default_env()
        .add_directive("json_transform".parse().unwrap())
        .add_directive("token_stream_parser".parse().unwrap());

    let layer = HierarchicalLayer::new(2).with_bracketed_fields(true).with_targets(true);

    let _ = Registry::default().with(filter).with(layer).try_init();
}
