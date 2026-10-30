use std::pin::Pin;

use futures::{Stream, stream};
use shoji::{
    traits::backend::{Error as BackendError, chat_token::StreamOutput as ChatTokenStreamOutput},
    types::basic::{
        ContextLength, Grammar as ShojiGrammar, SamplingMethod as ShojiSamplingMethod,
        SamplingPolicy as ShojiSamplingPolicy,
    },
};
use tokenizers::Tokenizer;
use uzu_engine::{
    backends::Backend,
    engine::language_model::{
        LanguageModel,
        grammar::{Grammar as UzuGrammar, GrammarConfig, GrammarError},
        stream::SamplingMethod as UzuSamplingMethod,
    },
};

pub fn error_stream<'a>(
    message: String
) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, BackendError>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}

pub fn get_grammar(
    grammar: ShojiGrammar,
    tokenizer: &Tokenizer,
    stop_token_ids: &[i32],
    trigger_token_sequence: Option<Vec<u64>>,
) -> Result<UzuGrammar, GrammarError> {
    let config = match grammar {
        ShojiGrammar::JsonAny {
            ..
        } => GrammarConfig::builtin_json(),
        ShojiGrammar::JsonSchema {
            schema,
        } => GrammarConfig::json_schema_simple(schema),
        ShojiGrammar::Regex {
            pattern,
        } => GrammarConfig::regex(pattern, false),
    };
    UzuGrammar::new(&config, tokenizer, trigger_token_sequence, Some(stop_token_ids))
}

/// The token sequence of the model's end-of-thinking tag. A reasoning model
/// opens every reply with reasoning tokens that no grammar can accept, so the
/// grammar must stay disengaged until this sequence has been generated.
/// Leading whitespace-only tokens are excluded because BPE can merge them with
/// preceding whitespace in both the prompt and generated reasoning. Tokens that
/// fuse whitespace with marker text retain their original encoding.
pub fn grammar_trigger_token_sequence(
    end_of_thinking_tag: Option<&str>,
    tokenizer: &Tokenizer,
) -> Option<Vec<u64>> {
    let tag = end_of_thinking_tag?;
    if tag.trim().is_empty() {
        return None;
    }

    let encoding = tokenizer.encode(tag, false).ok()?;
    let token_ids = encoding.get_ids();
    let leading_whitespace_tokens = token_ids
        .iter()
        .take_while(|&&token_id| {
            tokenizer.decode(&[token_id], false).is_ok_and(|text| !text.is_empty() && text.trim().is_empty())
        })
        .count();
    let token_ids: Vec<u64> =
        token_ids[leading_whitespace_tokens..].iter().map(|token_id| u64::from(*token_id)).collect();
    (!token_ids.is_empty()).then_some(token_ids)
}

/// Returns the end-of-thinking trigger only when it still has to be generated.
/// Toggle-capable models render the trigger into the generation prompt when
/// reasoning is disabled, in which case the grammar must engage immediately.
pub fn grammar_trigger_token_sequence_for_prompt(
    trigger_token_sequence: Option<&[u64]>,
    prompt: &[u64],
    tokenizer: &Tokenizer,
) -> Option<Vec<u64>> {
    let trigger_token_sequence = trigger_token_sequence.filter(|sequence| !sequence.is_empty())?;
    let trailing_tokens = prompt
        .windows(trigger_token_sequence.len())
        .rposition(|window| window == trigger_token_sequence)
        .map(|start| &prompt[start + trigger_token_sequence.len()..]);

    let trigger_is_in_generation_prompt = trailing_tokens.is_some_and(|trailing_tokens| {
        let Ok(trailing_tokens) = trailing_tokens.iter().copied().map(u32::try_from).collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        tokenizer.decode(&trailing_tokens, false).is_ok_and(|suffix| suffix.trim().is_empty())
    });

    (!trigger_is_in_generation_prompt).then(|| trigger_token_sequence.to_vec())
}

pub fn get_max_context_length<B: Backend>(
    model: &LanguageModel<B>,
    context_length: ContextLength,
) -> Option<u32> {
    match context_length {
        ContextLength::Default {
            ..
        } => model.recommended_context_length(),
        ContextLength::Maximal {
            ..
        } => model.max_context_length(),
        ContextLength::Custom {
            length,
        } => Some(length.max(0) as u32),
    }
}

pub fn get_sampling_method<B: Backend>(
    model: &LanguageModel<B>,
    sampling_method: &ShojiSamplingPolicy,
) -> Result<UzuSamplingMethod, String> {
    match sampling_method {
        ShojiSamplingPolicy::Default {
            ..
        } => Ok(model.default_sampling_method()),
        ShojiSamplingPolicy::Custom {
            method,
        } => get_custom_sampling_method(method),
    }
}

fn get_custom_sampling_method(method: &ShojiSamplingMethod) -> Result<UzuSamplingMethod, String> {
    match method {
        ShojiSamplingMethod::Greedy {
            ..
        } => Ok(UzuSamplingMethod::Greedy),
        ShojiSamplingMethod::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            suffix_repetition_length,
        } => {
            if let Some(temperature) = temperature
                && (temperature.is_nan() || *temperature <= 0.0)
            {
                return Err(format!(
                    "stochastic sampling needs temperature > 0, got {temperature}, use greedy sampling instead"
                ));
            }
            if let Some(repetition_penalty) = repetition_penalty
                && (repetition_penalty.is_nan() || *repetition_penalty <= 0.0)
            {
                return Err(format!("repetition_penalty must be > 0, got {repetition_penalty}"));
            }
            if let Some(suffix_repetition_length) = suffix_repetition_length
                && *suffix_repetition_length < 1
            {
                return Err(format!("suffix_repetition_length must be >= 1, got {suffix_repetition_length}"));
            }
            if repetition_penalty.is_some() && suffix_repetition_length.is_none() {
                return Err("repetition_penalty needs suffix_repetition_length".to_string());
            }
            Ok(UzuSamplingMethod::Stochastic {
                temperature: temperature.map(|value| value as f32),
                top_k: top_k.map(|value| value as u32),
                top_p: top_p.map(|value| value as f32),
                min_p: min_p.map(|value| value as f32),
                repetition_penalty: repetition_penalty.map(|value| value as f32),
                suffix_repetition_length: suffix_repetition_length.map(|value| value as u32),
            })
        },
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        AddedToken, Tokenizer,
        decoders::fuse::Fuse,
        models::{bpe::BPE, wordlevel::WordLevel},
        pre_tokenizers::byte_level::ByteLevel,
    };

    use super::{ShojiGrammar, get_grammar, grammar_trigger_token_sequence, grammar_trigger_token_sequence_for_prompt};

    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<unk>".to_string(), 0),
                    ("</think>".to_string(), 1),
                    ("\n\n".to_string(), 2),
                    ("next".to_string(), 3),
                ]
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

    // Like Qwen's tokenizer, merge adjacent newlines while keeping the thinking
    // tags as special tokens. No model download is needed to reproduce the bug.
    fn bpe_tokenizer() -> Tokenizer {
        let vocab = [
            ("<unk>", 0),
            ("Ċ", 1),
            ("ĊĊ", 2),
            ("<think>", 3),
            ("</think>", 4),
            ("Y", 5),
            ("E", 6),
            ("S", 7),
            ("N", 8),
            ("O", 9),
            ("{", 10),
            ("}", 11),
            ("<eos>", 12),
            ("Ġ", 13),
            ("ĠY", 14),
        ]
        .map(|(token, id)| (token.to_string(), id));
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![("Ċ".into(), "Ċ".into()), ("Ġ".into(), "Y".into())])
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<think>", true),
            AddedToken::from("</think>", true),
            AddedToken::from("<eos>", true),
        ]);
        tokenizer
    }

    fn encode(
        tokenizer: &Tokenizer,
        text: &str,
    ) -> Vec<u64> {
        tokenizer.encode(text, false).unwrap().get_ids().iter().map(|id| u64::from(*id)).collect()
    }

    fn grammars() -> [ShojiGrammar; 3] {
        [
            ShojiGrammar::Regex {
                pattern: "(YES|NO)".into(),
            },
            ShojiGrammar::JsonAny {},
            ShojiGrammar::JsonSchema {
                schema: r#"{"type":"object","additionalProperties":false}"#.into(),
            },
        ]
    }

    #[test]
    fn grammar_engages_when_prompt_closes_reasoning_with_merged_newlines() {
        let tokenizer = bpe_tokenizer();
        assert_eq!(encode(&tokenizer, "\n</think>"), [1, 4]);
        let prompt = encode(&tokenizer, "<think>\n\n</think>\n\n");
        assert_eq!(prompt, [3, 2, 4, 2]);
        let trigger = grammar_trigger_token_sequence(Some("\n</think>"), &tokenizer);
        let trigger = grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &prompt, &tokenizer);
        assert_eq!(trigger, None);

        for config in grammars() {
            let mut grammar = get_grammar(config, &tokenizer, &[12], trigger.clone()).unwrap();
            assert!(grammar.next_bitmask(&mut [0]));
        }
    }

    #[test]
    fn grammar_engages_after_generated_trigger_with_merged_newlines() {
        let tokenizer = bpe_tokenizer();
        let trigger = grammar_trigger_token_sequence(Some("\n</think>"), &tokenizer);
        let prompt = encode(&tokenizer, "<think>\n");
        let trigger = grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &prompt, &tokenizer);

        for config in grammars() {
            let mut grammar = get_grammar(config, &tokenizer, &[12], trigger.clone()).unwrap();
            assert!(!grammar.next_bitmask(&mut [0]));
            grammar.accept_token(2).unwrap(); // merged newlines during reasoning
            assert!(!grammar.next_bitmask(&mut [0]));
            grammar.accept_token(4).unwrap(); // </think>
            assert!(grammar.next_bitmask(&mut [0]));
            // Speculative rollback across the trigger must restore reasoning.
            grammar.rollback(1);
            assert!(!grammar.next_bitmask(&mut [0]));
            grammar.accept_token(4).unwrap();
            assert!(grammar.next_bitmask(&mut [0]));
        }
    }

    #[test]
    fn grammar_trigger_preserves_multi_token_tags() {
        let tokenizer = bpe_tokenizer();
        let trigger = grammar_trigger_token_sequence(Some("\n</think><think>"), &tokenizer);
        assert_eq!(trigger, Some(vec![4, 3]));
        let prompt = encode(&tokenizer, "<think>\n\n</think><think>\n\n");
        assert_eq!(grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &prompt, &tokenizer), None);
        let history = encode(&tokenizer, "</think><think>\nYES<think>\n");
        assert_eq!(grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &history, &tokenizer), trigger);
    }

    #[test]
    fn grammar_trigger_preserves_whitespace_fused_with_marker_text() {
        let tokenizer = bpe_tokenizer();
        let expected = encode(&tokenizer, " YES");
        assert_eq!(expected, [14, 6, 7]);
        assert_ne!(expected, encode(&tokenizer, "YES"));

        // Preserve the fused space even when a separate newline precedes it.
        for tag in [" YES", "\n YES"] {
            let trigger = grammar_trigger_token_sequence(Some(tag), &tokenizer);
            assert_eq!(trigger.as_deref(), Some(expected.as_slice()));

            let prompt = encode(&tokenizer, "<think>\n\n YES\n\n");
            assert_eq!(grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &prompt, &tokenizer), None);

            let prompt = encode(&tokenizer, "<think>\n");
            let trigger = grammar_trigger_token_sequence_for_prompt(trigger.as_deref(), &prompt, &tokenizer);
            let mut grammar = get_grammar(
                ShojiGrammar::Regex {
                    pattern: "(YES|NO)".into(),
                },
                &tokenizer,
                &[12],
                trigger,
            )
            .unwrap();
            for token in encode(&tokenizer, "\n\n YES") {
                assert!(!grammar.next_bitmask(&mut [0]));
                grammar.accept_token(token).unwrap();
            }
            assert!(grammar.next_bitmask(&mut [0]));
        }
    }

    #[test]
    fn grammar_trigger_ignores_missing_or_whitespace_only_tags() {
        let tokenizer = bpe_tokenizer();
        for tag in [None, Some(""), Some("\n\n")] {
            assert_eq!(grammar_trigger_token_sequence(tag, &tokenizer), None);
        }
    }
}

#[cfg(test)]
mod sampling_tests {
    use shoji::types::basic::SamplingMethod as ShojiSamplingMethod;
    use uzu_engine::engine::language_model::stream::SamplingMethod as UzuSamplingMethod;

    use super::get_custom_sampling_method;

    fn stochastic(temperature: Option<f64>) -> ShojiSamplingMethod {
        ShojiSamplingMethod::Stochastic {
            temperature,
            top_k: None,
            top_p: Some(0.9),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }
    }

    #[test]
    fn stochastic_rejects_non_positive_temperature() {
        for temperature in [0.0, -0.5, f64::NAN] {
            assert!(get_custom_sampling_method(&stochastic(Some(temperature))).is_err(), "temperature {temperature}");
        }
    }

    #[test]
    fn stochastic_accepts_positive_or_unset_temperature() {
        for temperature in [Some(0.7), None] {
            assert!(matches!(
                get_custom_sampling_method(&stochastic(temperature)),
                Ok(UzuSamplingMethod::Stochastic { .. })
            ));
        }
    }

    fn with_repetition(
        repetition_penalty: Option<f64>,
        suffix_repetition_length: Option<i64>,
    ) -> ShojiSamplingMethod {
        ShojiSamplingMethod::Stochastic {
            temperature: Some(0.7),
            top_k: None,
            top_p: Some(0.9),
            min_p: None,
            repetition_penalty,
            suffix_repetition_length,
        }
    }

    #[test]
    fn stochastic_rejects_invalid_repetition_settings() {
        for (penalty, suffix) in [
            (Some(0.0), Some(64)),
            (Some(f64::NAN), Some(64)),
            (Some(1.1), Some(0)),
            (None, Some(0)),
            (Some(1.1), None),
        ] {
            assert!(get_custom_sampling_method(&with_repetition(penalty, suffix)).is_err(), "{penalty:?} {suffix:?}");
        }
    }

    #[test]
    fn stochastic_accepts_valid_repetition_settings() {
        for (penalty, suffix) in [(Some(1.1), Some(64)), (None, None), (None, Some(64))] {
            assert!(matches!(
                get_custom_sampling_method(&with_repetition(penalty, suffix)),
                Ok(UzuSamplingMethod::Stochastic { .. })
            ));
        }
    }
}
