use std::iter::repeat_n;

use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler,
    GrammarMatcher, TokenizerInfo,
};

use crate::session::{config::GrammarConfig, types::Error};

enum CompiledGrammarEngagementState {
    Always,
    Triggered {
        trigger_token_id: u64,
        trigger_distance: Option<usize>,
    },
}

impl CompiledGrammarEngagementState {
    fn is_engaged(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Triggered {
                trigger_token_id: _,
                trigger_distance,
            } => trigger_distance.is_some(),
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) {
        match self {
            Self::Always => (),
            Self::Triggered {
                trigger_token_id,
                trigger_distance,
            } => {
                if let Some(trigger_distance) = trigger_distance {
                    *trigger_distance += 1;
                } else if token_id == *trigger_token_id {
                    *trigger_distance = Some(0);
                }
            },
        }
    }

    pub fn rollback(
        &mut self,
        num_tokens: usize,
    ) -> usize {
        match self {
            Self::Always => num_tokens,
            Self::Triggered {
                trigger_token_id: _,
                trigger_distance,
            } => {
                let num_grammar_tokens =
                    usize::min(trigger_distance.unwrap_or(0), num_tokens);
                *trigger_distance =
                    trigger_distance.and_then(|x| x.checked_sub(num_tokens));
                num_grammar_tokens
            },
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Always => (),
            Self::Triggered {
                trigger_token_id: _,
                trigger_distance,
            } => *trigger_distance = None,
        }
    }
}

pub struct CompiledGrammar {
    vocab_size: usize,
    matcher: GrammarMatcher,
    engagement_state: CompiledGrammarEngagementState,
}

impl CompiledGrammar {
    pub fn from_config(
        config: &GrammarConfig,
        trigger_token_id: Option<u64>,
        tokenizer_info: &TokenizerInfo,
    ) -> Result<Self, Error> {
        let vocab_size = tokenizer_info.vocab_size() as usize;

        let grammar = match config {
            GrammarConfig::JsonSchema {
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
            } => {
                let separators_ref =
                    separators.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
                Grammar::from_json_schema(
                    schema,
                    *any_whitespace,
                    *indent,
                    separators_ref,
                    *strict_mode,
                    None,
                    false,
                )
                .map_err(|error_message| Error::GrammarError(error_message))?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => Grammar::from_regex(pattern, *print_converted_ebnf)
                .map_err(|error_message| Error::GrammarError(error_message))?,
            GrammarConfig::BuiltinJson => Grammar::builtin_json_grammar(),
        };
        let mut compiler = GrammarCompiler::new(tokenizer_info, 8, true, -1)
            .map_err(|error_message| Error::GrammarError(error_message))?;
        let compiled = compiler
            .compile_grammar(&grammar)
            .map_err(|error_message| Error::GrammarError(error_message))?;
        let matcher = GrammarMatcher::new(&compiled, None, true, -1)
            .map_err(|error_message| Error::GrammarError(error_message))?;

        let engagement_state = if let Some(trigger_token_id) = trigger_token_id
        {
            CompiledGrammarEngagementState::Triggered {
                trigger_token_id,
                trigger_distance: None,
            }
        } else {
            CompiledGrammarEngagementState::Always
        };

        Ok(Self {
            vocab_size,
            matcher,
            engagement_state,
        })
    }

    pub fn next_bitmask(&mut self) -> Result<Option<Box<[u32]>>, Error> {
        if self.engagement_state.is_engaged() {
            let mut cpu_mask = repeat_n(0, self.vocab_size.div_ceil(32))
                .collect::<Box<[u32]>>();
            let batch_mask_slice = &mut cpu_mask;
            let mut shape_i64 = [self.vocab_size.div_ceil(32) as i64];
            let mut bitmask_tensor = DLTensor {
                data: batch_mask_slice.as_mut_ptr() as *mut core::ffi::c_void,
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: 0,
                    bits: 32,
                    lanes: 1,
                },
                shape: shape_i64.as_mut_ptr(),
                strides: core::ptr::null_mut(),
                byte_offset: 0,
            };

            let success = self.matcher.fill_next_token_bitmask(
                &mut bitmask_tensor,
                0,
                false,
            );
            if !success {
                return Err(Error::GrammarError(
                    "Failed to fill next token bitmask".to_string(),
                ));
            }

            Ok(Some(cpu_mask))
        } else {
            Ok(None)
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), Error> {
        if self.engagement_state.is_engaged() {
            if (token_id as usize) >= self.vocab_size {
                return Err(Error::TokenOutOfGrammarRange(
                    token_id,
                    self.vocab_size,
                ));
            }

            if !self.matcher.accept_token(token_id as i32) {
                return Err(Error::GrammarReject);
            }
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

    pub fn reset(&mut self) {
        self.matcher.reset();
        self.engagement_state.reset();
    }

    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }
}
