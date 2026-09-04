use tokenizers::{Tokenizer, decoders::fuse::Fuse, models::wordlevel::WordLevel};

use super::grammar_trigger_token_sequence_for_prompt;

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            [("<unk>".to_string(), 0), ("</think>".to_string(), 1), ("\n\n".to_string(), 2), ("next".to_string(), 3)]
                .into_iter()
                .collect(),
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(Fuse::new()));
    tokenizer
}

#[test]
fn grammar_engages_immediately_when_prompt_closes_reasoning() {
    let tokenizer = tokenizer();

    assert_eq!(grammar_trigger_token_sequence_for_prompt(Some(&[1]), &[1, 2], &tokenizer), None);
}

#[test]
fn grammar_waits_for_trigger_when_only_history_contains_it() {
    let tokenizer = tokenizer();

    assert_eq!(grammar_trigger_token_sequence_for_prompt(Some(&[1]), &[1, 2, 3], &tokenizer), Some(vec![1]));
}
