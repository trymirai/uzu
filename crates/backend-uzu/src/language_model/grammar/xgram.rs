#![cfg(grammar_xgrammar)]

use std::{collections::VecDeque, iter::repeat_n};

use xgrammar::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler, GrammarMatcher,
    TokenizerInfo, c_void,
};

use crate::{
    data_type::DataType,
    language_model::grammar::CompiledGrammar,
    session::{config::GrammarConfig, types::Error},
};

enum CompiledGrammarEngagementState {
    Always,
    Triggered {
        trigger_tokens: Vec<u64>,
        pre_engagement: VecDeque<u64>,
        pre_engagement_retained_tokens: usize,
        pre_engagement_trim_slack_tokens: usize,
        trigger_distance: Option<usize>,
    },
}

impl CompiledGrammarEngagementState {
    fn from_config(
        trigger_token_ids: &[u64],
        pre_engagement_retained_tokens: usize,
        pre_engagement_trim_slack_tokens: usize,
    ) -> Self {
        if trigger_token_ids.is_empty() {
            Self::Always
        } else {
            let retained = pre_engagement_retained_tokens.max(trigger_token_ids.len());
            Self::Triggered {
                trigger_tokens: trigger_token_ids.to_vec(),
                pre_engagement: VecDeque::new(),
                pre_engagement_retained_tokens: retained,
                pre_engagement_trim_slack_tokens,
                trigger_distance: None,
            }
        }
    }

    fn ends_with_tokens(
        pre_engagement: &VecDeque<u64>,
        trigger_tokens: &[u64],
    ) -> bool {
        if trigger_tokens.len() > pre_engagement.len() {
            return false;
        }

        pre_engagement.iter().rev().zip(trigger_tokens.iter().rev()).all(|(observed, expected)| observed == expected)
    }

    fn is_engaged(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Triggered {
                trigger_distance,
                ..
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
                trigger_tokens,
                pre_engagement,
                pre_engagement_retained_tokens,
                pre_engagement_trim_slack_tokens,
                trigger_distance,
            } => {
                if let Some(trigger_distance) = trigger_distance {
                    *trigger_distance += 1;
                } else {
                    pre_engagement.push_back(token_id);
                    let trim_threshold =
                        pre_engagement_retained_tokens.saturating_add(*pre_engagement_trim_slack_tokens);
                    if pre_engagement.len() > trim_threshold {
                        let to_drop = pre_engagement.len() - *pre_engagement_retained_tokens;
                        for _ in 0..to_drop {
                            let _ = pre_engagement.pop_front();
                        }
                    }
                    if Self::ends_with_tokens(pre_engagement, trigger_tokens) {
                        *trigger_distance = Some(0);
                    }
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
                pre_engagement,
                trigger_distance,
                ..
            } => match *trigger_distance {
                Some(distance) => {
                    let num_grammar_tokens = usize::min(distance, num_tokens);
                    if num_tokens <= distance {
                        *trigger_distance = Some(distance - num_tokens);
                    } else {
                        *trigger_distance = None;
                        let pre_rollback = num_tokens - distance;
                        pre_engagement.truncate(pre_engagement.len().saturating_sub(pre_rollback));
                    }
                    num_grammar_tokens
                },
                None => {
                    pre_engagement.truncate(pre_engagement.len().saturating_sub(num_tokens));
                    0
                },
            },
        }
    }
}

pub struct CompiledXGrammar {
    vocab_size: usize,
    matcher: GrammarMatcher,
    engagement_state: CompiledGrammarEngagementState,
}

impl CompiledXGrammar {
    pub fn from_config(
        config: &GrammarConfig,
        trigger_token_ids: &[u64],
        pre_engagement_retained_tokens: usize,
        pre_engagement_trim_slack_tokens: usize,
        tokenizer_info: &TokenizerInfo,
    ) -> Result<Self, Error> {
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
                Grammar::from_json_schema(schema, *any_whitespace, *indent, separators_ref, *strict_mode, None, false)
                    .map_err(Error::GrammarError)?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => Grammar::from_regex(pattern, *print_converted_ebnf).map_err(Error::GrammarError)?,
            GrammarConfig::BuiltinJson => Grammar::builtin_json_grammar(),
        };
        let mut compiler = GrammarCompiler::new(tokenizer_info, 8, true, -1).map_err(Error::GrammarError)?;
        let compiled = compiler.compile_grammar(&grammar).map_err(Error::GrammarError)?;
        let matcher = GrammarMatcher::new(&compiled, None, true, -1).map_err(Error::GrammarError)?;

        let engagement_state = CompiledGrammarEngagementState::from_config(
            trigger_token_ids,
            pre_engagement_retained_tokens,
            pre_engagement_trim_slack_tokens,
        );

        Ok(Self {
            vocab_size,
            matcher,
            engagement_state,
        })
    }
}

impl CompiledGrammar for CompiledXGrammar {
    fn next_bitmask(&mut self) -> Result<Option<Box<[u32]>>, Error> {
        if self.engagement_state.is_engaged() {
            let mut cpu_mask = repeat_n(0, self.vocab_size.div_ceil(32)).collect::<Box<[u32]>>();
            let batch_mask_slice = &mut cpu_mask;
            let mut shape_i64 = [self.vocab_size.div_ceil(32) as i64];
            let mut bitmask_tensor = unsafe {
                DLTensor::new(
                    batch_mask_slice.as_mut_ptr() as *mut c_void,
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

            let success = self.matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false);
            if !success {
                return Err(Error::GrammarError("Failed to fill next token bitmask".to_string()));
            }

            Ok(Some(cpu_mask))
        } else {
            Ok(None)
        }
    }

    fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), Error> {
        if self.engagement_state.is_engaged() {
            if (token_id as usize) >= self.vocab_size {
                return Err(Error::TokenOutOfGrammarRange(token_id, self.vocab_size));
            }

            if !self.matcher.accept_token(token_id as i32) {
                return Err(Error::GrammarReject);
            }
        }

        self.engagement_state.accept_token(token_id);
        Ok(())
    }

    fn rollback(
        &mut self,
        num_tokens: usize,
    ) {
        let num_grammar_tokens = self.engagement_state.rollback(num_tokens);

        if num_grammar_tokens > 0 {
            self.matcher.rollback(num_grammar_tokens as i32);
        }
    }

    fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }
}

trait DLDataTypeCodeProvider {
    fn dl_data_type_code(self) -> DLDataTypeCode;
}

impl DLDataTypeCodeProvider for DataType {
    fn dl_data_type_code(self) -> DLDataTypeCode {
        match self {
            DataType::BF16 => DLDataTypeCode::kDLBfloat,
            DataType::F16 | DataType::F32 | DataType::F64 => DLDataTypeCode::kDLFloat,
            DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => DLDataTypeCode::kDLInt,
            DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => DLDataTypeCode::kDLUInt,
        }
    }
}

impl From<DataType> for DLDataType {
    fn from(data_type: DataType) -> Self {
        Self {
            code: data_type.dl_data_type_code() as u8,
            bits: data_type.size_in_bits() as u8,
            lanes: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::CompiledGrammarEngagementState;

    #[uzu_test]
    fn test_sequence_trigger_engages_only_after_full_sequence() {
        let trigger = [10u64, 11, 12];
        let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

        // Reasoning tokens, including the trailing sub-token of the tag appearing
        // mid-reasoning, must not engage the grammar.
        for token in [99, 12, 10, 11, 7] {
            state.accept_token(token);
            assert!(!state.is_engaged());
        }

        // The full sequence only engages once all sub-tokens arrive in order.
        state.accept_token(10);
        assert!(!state.is_engaged());
        state.accept_token(11);
        assert!(!state.is_engaged());
        state.accept_token(12);
        assert!(state.is_engaged());

        // Tokens accepted after engagement count toward the grammar distance.
        state.accept_token(42);
        state.accept_token(43);
        assert_eq!(state.rollback(0), 0);
        assert!(state.is_engaged());
    }

    #[uzu_test]
    fn test_rollback_across_engagement_boundary() {
        let trigger = [10u64, 11, 12];
        let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

        for token in [10, 11, 12] {
            state.accept_token(token);
        }
        assert!(state.is_engaged());

        // Two JSON tokens accepted while engaged (distance == 2).
        state.accept_token(42);
        state.accept_token(43);

        // Rolling back 4 tokens crosses the boundary: only the 2 grammar tokens
        // are reported, and the state disengages while restoring the partial tag.
        let grammar_rollback = state.rollback(4);
        assert_eq!(grammar_rollback, 2);
        assert!(!state.is_engaged());

        // After disengaging, the partial-match prefix is preserved, so completing
        // the remaining sub-tokens of the tag re-engages exactly.
        state.accept_token(11);
        assert!(!state.is_engaged());
        state.accept_token(12);
        assert!(state.is_engaged());
    }

    #[uzu_test]
    fn test_sequence_trigger_still_matches_after_long_reasoning_prefix() {
        let trigger = [10u64, 11, 12];
        let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

        for _ in 0..384 {
            state.accept_token(99);
            assert!(!state.is_engaged());
        }

        state.accept_token(10);
        state.accept_token(11);
        state.accept_token(12);
        assert!(state.is_engaged());
    }
}
