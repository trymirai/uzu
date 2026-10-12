use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
use uzu_engine_macros::uzu_test;

use super::{Grammar, GrammarConfig, GrammarError};

fn grammar(
    config: &GrammarConfig,
    trigger: Option<Vec<u64>>,
) -> Grammar {
    let vocab = ["a", "b", "ab", "{", "}", "[", "]", "<eos>", "<unk>"]
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect();
    let tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("<unk>".into()).build().unwrap());
    Grammar::new(config, &tokenizer, trigger, Some(&[7])).unwrap()
}

#[uzu_test]
fn regex_masks_reject_invalid_tokens_and_allow_stop_after_completion() {
    let mut grammar = grammar(&GrammarConfig::regex("ab".into(), false), None);
    let mut mask = [u32::MAX; 3];
    assert!(grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [0b101, 0, 0]);
    assert!(matches!(grammar.accept_token(1), Err(GrammarError::GrammarReject)));
    grammar.accept_token(0).unwrap();
    assert!(grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [0b10, 0, 0]);
    grammar.accept_token(1).unwrap();
    assert!(grammar.is_terminated());
    assert!(grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [1 << 7, 0, 0]);
    grammar.accept_token(7).unwrap();
}

#[uzu_test]
fn rollback_restores_masks_and_termination() {
    let mut grammar = grammar(&GrammarConfig::regex("ab".into(), false), None);
    grammar.accept_token(0).unwrap();
    grammar.accept_token(1).unwrap();
    grammar.rollback(1);
    assert!(!grammar.is_terminated());
    let mut mask = [0];
    grammar.next_bitmask(&mut mask);
    assert_eq!(mask, [0b10]);
    grammar.rollback(1);
    grammar.next_bitmask(&mut mask);
    assert_eq!(mask, [0b101]);
}

#[uzu_test]
fn trigger_rollback_restores_unconstrained_sampling() {
    let mut grammar = grammar(&GrammarConfig::regex("ab".into(), false), Some(vec![5, 6]));
    let mut mask = [0; 2];
    assert!(!grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [u32::MAX; 2]);
    grammar.accept_token(5).unwrap();
    grammar.accept_token(6).unwrap();
    grammar.accept_token(2).unwrap();
    assert!(grammar.is_terminated());
    grammar.rollback(2);
    assert!(!grammar.is_terminated());
    assert!(!grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [u32::MAX; 2]);
    grammar.accept_token(6).unwrap();
    assert!(grammar.next_bitmask(&mut mask));
    assert_eq!(mask, [0b101, 0]);
}

#[uzu_test]
fn json_and_json_schema_accept_complete_objects() {
    for config in [
        GrammarConfig::builtin_json(),
        GrammarConfig::json_schema_simple(r#"{"type":"object","additionalProperties":false}"#.into()),
    ] {
        let mut grammar = grammar(&config, None);
        grammar.accept_token(3).unwrap();
        grammar.accept_token(4).unwrap();
        assert!(grammar.is_terminated());
        let mut mask = [u32::MAX; 2];
        assert!(grammar.next_bitmask(&mut mask));
        assert_eq!(mask, [1 << 7, 0]);
    }
}

#[uzu_test]
fn invalid_grammars_preserve_compilation_errors() {
    let tokenizer = Tokenizer::new(WordLevel::default());
    for config in [GrammarConfig::regex("[".into(), false), GrammarConfig::json_schema_simple("{".into())] {
        let error = match Grammar::new(&config, &tokenizer, None, None) {
            Err(error) => error,
            Ok(_) => panic!("invalid grammar compiled"),
        };
        assert!(matches!(error, GrammarError::Compile(_)));
        assert!(std::error::Error::source(&error).is_some());
    }
}
