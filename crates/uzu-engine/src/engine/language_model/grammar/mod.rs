use thiserror::Error;
use tokenizers::Tokenizer;
use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar as XGrammarGrammar, GrammarCompiler, GrammarMatcher,
    TokenizerInfo, c_void,
};

use crate::{data_type::DataType, engine::language_model::grammar::engagement::GrammarEngagementState};

mod config;
mod data_type;
mod engagement;

pub use config::GrammarConfig;

// TODO: jumpforward?

pub struct Grammar {
    vocab_size: usize,
    matcher: GrammarMatcher,
    engagement_state: GrammarEngagementState,
}

#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("XGrammar error: {0}")]
    XGrammar(String),
}

impl Grammar {
    pub fn new(
        config: &GrammarConfig,
        tokenizer: &Tokenizer,
        trigger_token_id: Option<u64>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, GrammarError> {
        let tokenizer_info =
            TokenizerInfo::from_huggingface(tokenizer, None, stop_token_ids).map_err(GrammarError::XGrammar)?;

        let vocab_size = tokenizer_info.vocab_size();

        let grammar = match config {
            GrammarConfig::JsonSchema {
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
            } => {
                let separators_ref = separators.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
                XGrammarGrammar::from_json_schema(
                    schema,
                    *any_whitespace,
                    *indent,
                    separators_ref,
                    *strict_mode,
                    None,
                    false,
                    false,
                )
                .map_err(GrammarError::XGrammar)?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => XGrammarGrammar::from_regex(pattern, *print_converted_ebnf).map_err(GrammarError::XGrammar)?,
            GrammarConfig::BuiltinJson => XGrammarGrammar::builtin_json_grammar(),
        };
        let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1).map_err(GrammarError::XGrammar)?;
        let compiled = compiler.compile_grammar(&grammar).map_err(GrammarError::XGrammar)?;
        let matcher = GrammarMatcher::new(&compiled, None, true, -1).map_err(GrammarError::XGrammar)?;

        let engagement_state = if let Some(trigger_token_id) = trigger_token_id {
            GrammarEngagementState::Triggered {
                trigger_token_id,
                trigger_distance: None,
            }
        } else {
            GrammarEngagementState::Always
        };

        Ok(Self {
            vocab_size,
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
        let vocab_size_in_u32s = self.vocab_size.div_ceil(DataType::U32.size_in_bits());
        assert!(bitmask.len() >= vocab_size_in_u32s); // NOTE: tokenizer vocab can be smaller than model vocab

        if self.engagement_state.is_engaged() {
            let mut shape_i64 = [vocab_size_in_u32s as i64];
            let mut bitmask_tensor = unsafe {
                DLTensor::new(
                    bitmask.as_mut_ptr() as *mut c_void,
                    DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    1,
                    DLDataType {
                        code: 0,
                        bits: 32,
                        lanes: 1,
                    },
                    shape_i64.as_mut_ptr(),
                    core::ptr::null_mut(),
                    0,
                )
            };

            bitmask[vocab_size_in_u32s..].fill(0);
            self.matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false)
        } else {
            bitmask.fill(u32::MAX);

            false
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), GrammarError> {
        if self.engagement_state.is_engaged() && !self.matcher.accept_token(token_id as i32) {
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
            self.matcher.rollback(num_grammar_tokens as i32);
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }
}
