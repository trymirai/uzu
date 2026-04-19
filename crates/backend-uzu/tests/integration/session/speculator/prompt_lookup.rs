use std::collections::HashMap;

use backend_uzu::prelude::{PromptLookupSpeculator, Speculator};
use tokenizers::Tokenizer;

use crate::common::path::get_test_model_path;

fn load_tokenizer() -> Tokenizer {
    let tokenizer_path = get_test_model_path().join("tokenizer.json");
    match Tokenizer::from_file(tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            panic!("Failed to load tokenizer: {}", e);
        },
    }
}

fn tokenize_text(
    tokenizer: &Tokenizer,
    text: &str,
) -> Vec<u64> {
    let encoding = tokenizer.encode(text, false).expect("Failed to encode text");
    encoding.get_ids().iter().map(|&id| id as u64).collect()
}

#[test]
fn test_empty_input() {
    let speculator = PromptLookupSpeculator::new();
    let prefix: Vec<u64> = vec![];

    let proposals = speculator.speculate(&prefix);

    assert_eq!(proposals.len(), 0);
}

#[test]
fn test_no_pattern_found() {
    let speculator = PromptLookupSpeculator::new();
    let prefix = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

    let proposals = speculator.speculate(&prefix);

    assert_eq!(proposals.len(), 0);
}

#[test]
fn test_with_real_text() {
    let tokenizer = load_tokenizer();
    let speculator = PromptLookupSpeculator::new_with_params(3);

    let text = "The quick brown fox jumps over the lazy dog. The quick brown fox";

    let mut tokens = tokenize_text(&tokenizer, text);

    for expected in tokenize_text(&tokenizer, " jumps over the lazy dog") {
        let proposals = speculator.speculate(&tokens);

        assert_eq!(proposals, HashMap::from([(expected, 1.0)]));

        tokens.push(expected);
    }
}
