use thiserror::Error;
use tokenizers::Tokenizer;
use xgrammar::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler, GrammarMatcher,
    TokenizerInfo, c_void,
};

use crate::{
    data_type::DataType,
    engine::language_model::grammar::{CompiledGrammar, GrammarConfig, GrammarError},
};

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
                let num_grammar_tokens = usize::min(trigger_distance.unwrap_or(0), num_tokens);
                *trigger_distance = trigger_distance.and_then(|x| x.checked_sub(num_tokens));
                num_grammar_tokens
            },
        }
    }
}

pub struct CompiledXGrammar {
    vocab_size: usize,
    matcher: GrammarMatcher,
    engagement_state: CompiledGrammarEngagementState,
}

#[derive(Debug, Error)]
pub enum CompiledXGrammarError {
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("XGrammar error: {0}")]
    XGrammar(String),
}

impl Into<GrammarError> for CompiledXGrammarError {
    fn into(self) -> GrammarError {
        GrammarError(Box::new(self))
    }
}

impl CompiledXGrammar {
    pub fn new(
        config: &GrammarConfig,
        tokenizer: &Tokenizer,
        trigger_token_id: Option<u64>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, GrammarError> {
        let tokenizer_info = TokenizerInfo::from_huggingface(tokenizer, None, stop_token_ids)
            .map_err(|e| CompiledXGrammarError::XGrammar(e).into())?;

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
                    .map_err(|e| CompiledXGrammarError::XGrammar(e).into())?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => Grammar::from_regex(pattern, *print_converted_ebnf)
                .map_err(|e| CompiledXGrammarError::XGrammar(e).into())?,
            GrammarConfig::BuiltinJson => Grammar::builtin_json_grammar(),
        };
        let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1)
            .map_err(|e| CompiledXGrammarError::XGrammar(e).into())?;
        let compiled = compiler.compile_grammar(&grammar).map_err(|e| CompiledXGrammarError::XGrammar(e).into())?;
        let matcher =
            GrammarMatcher::new(&compiled, None, true, -1).map_err(|e| CompiledXGrammarError::XGrammar(e).into())?;

        let engagement_state = if let Some(trigger_token_id) = trigger_token_id {
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
}

impl CompiledGrammar for CompiledXGrammar {
    fn next_bitmask(
        &mut self,
        bitmask: &mut [u32],
    ) -> bool {
        let vocab_size_in_u32s = self.vocab_size.div_ceil(32);
        assert!(bitmask.len() == vocab_size_in_u32s);

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

            self.matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false)
        } else {
            bitmask.fill(u32::MAX);

            false
        }
    }

    fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), GrammarError> {
        if self.engagement_state.is_engaged() && !self.matcher.accept_token(token_id as i32) {
            return Err(CompiledXGrammarError::GrammarReject.into());
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
