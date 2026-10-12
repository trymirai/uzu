use tokenizers::Tokenizer;
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo, get_bitmask_size};

use super::{GrammarConfig, GrammarError, engagement::GrammarEngagementState};

pub struct Grammar {
    bitmask_size: usize,
    matcher: GrammarMatcher,
    engagement_state: GrammarEngagementState,
}

impl Grammar {
    pub fn new(
        config: &GrammarConfig,
        tokenizer: &Tokenizer,
        trigger_token_sequence: Option<Vec<u64>>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, GrammarError> {
        let tokenizer_info =
            TokenizerInfo::from_huggingface(tokenizer, None, stop_token_ids).map_err(GrammarError::XGrammar)?;

        let bitmask_size = get_bitmask_size(tokenizer_info.vocab_size()) as usize;
        let compiler = GrammarCompiler::with_defaults(tokenizer_info);

        let compiled = match config {
            GrammarConfig::JsonSchema {
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
            } => {
                let separators_ref = separators.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
                compiler.compile_json_schema(schema, *any_whitespace, *indent, separators_ref, *strict_mode, None)?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => {
                let compiled = compiler.compile_regex(pattern)?;
                if *print_converted_ebnf {
                    println!("{}", compiled.grammar());
                }
                compiled
            },
            GrammarConfig::BuiltinJson => compiler.compile_builtin_json_grammar(),
        };
        let matcher = GrammarMatcher::from_compiled_grammar(&compiled, true);

        let engagement_state = match trigger_token_sequence.filter(|sequence| !sequence.is_empty()) {
            Some(trigger_sequence) => GrammarEngagementState::Triggered {
                trigger_sequence,
                match_history: Vec::new(),
            },
            None => GrammarEngagementState::Always,
        };

        Ok(Self {
            bitmask_size,
            matcher,
            engagement_state,
        })
    }
}

impl Grammar {
    pub fn next_bitmask(
        &mut self,
        bitmask: &mut [u32],
    ) -> bool {
        assert!(bitmask.len() >= self.bitmask_size); // NOTE: tokenizer vocab can be smaller than model vocab

        if self.engagement_state.is_engaged() {
            bitmask[self.bitmask_size..].fill(0);
            self.matcher
                .fill_next_token_bitmask(
                    bytemuck::cast_slice_mut(&mut bitmask[..self.bitmask_size]),
                    &[self.bitmask_size as i64],
                    0,
                )
                .expect("bitmask shape matches the tokenizer vocabulary")
        } else {
            bitmask.fill(u32::MAX);

            false
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), GrammarError> {
        // A terminated matcher cannot advance and its bitmask only allows stop
        // tokens, which close generation without being part of the grammar, so
        // tokens sampled after termination must not be rejected.
        if self.engagement_state.is_engaged()
            && !self.matcher.is_terminated()
            && !self.matcher.accept_token(token_id as i32)
        {
            return Err(GrammarError::GrammarReject);
        }

        self.engagement_state.accept_token(token_id);
        Ok(())
    }

    pub fn rollback(
        &mut self,
        num_tokens: usize,
    ) {
        let num_grammar_tokens = self.engagement_state.rollback(num_tokens);

        if num_grammar_tokens > 0 {
            self.matcher
                .rollback(num_grammar_tokens as i32)
                .expect("cannot roll back more tokens than the grammar accepted");
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }
}

#[cfg(test)]
#[path = "../../../../unit/engine/language_model/grammar/grammar_test.rs"]
mod tests;
