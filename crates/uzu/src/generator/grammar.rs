use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler,
    GrammarMatcher, TokenizerInfo,
};

use crate::session::{config::GrammarConfig, types::Error};

pub struct CompiledGrammar {
    pub matcher: GrammarMatcher,
    pub vocab_size: usize,
}

impl CompiledGrammar {
    pub fn from_config(
        config: &GrammarConfig,
        tokenizer_info: &TokenizerInfo,
    ) -> Result<Self, Error> {
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
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => Grammar::from_regex(pattern, *print_converted_ebnf),
            GrammarConfig::BuiltinJson => Grammar::builtin_json_grammar(),
        };

        let mut compiler = GrammarCompiler::new(tokenizer_info, 8, true, -1);
        let compiled = compiler.compile_grammar(&grammar);
        let matcher = GrammarMatcher::new(&compiled, None, true, -1);
        let vocab_size = tokenizer_info.vocab_size() as usize;

        Ok(Self {
            matcher,
            vocab_size,
        })
    }

    pub fn next_bitmask(&mut self) -> Result<Vec<u32>, Error> {
        let mut cpu_mask = vec![0; self.vocab_size.div_ceil(32)];
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

        let success =
            self.matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false);
        if !success {
            return Err(Error::GrammarError);
        }

        Ok(cpu_mask)
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) {
        self.matcher.accept_token(token_id as i32);
    }

    pub fn rollback(
        &mut self,
        num_tokens: i32,
    ) {
        self.matcher.rollback(num_tokens);
    }

    pub fn reset(&mut self) {
        self.matcher.reset();
    }
}
