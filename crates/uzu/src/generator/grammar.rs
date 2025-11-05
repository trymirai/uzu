use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler,
    GrammarMatcher, TokenizerInfo,
};

use crate::{
    Array,
    backends::metal::forward_pass::ForwardPassState,
    device::device_context::DeviceContext,
    session::{config::GrammarConfig, types::Error},
};

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
                let separators_ref = separators.as_ref().map(|(a, b)| {
                    (a.as_str(), b.as_str())
                });
                Grammar::from_json_schema(
                    schema,
                    *any_whitespace,
                    *indent,
                    separators_ref,
                    *strict_mode,
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

    pub fn fill_bitmask_for_batch(
        &mut self,
        state: &mut ForwardPassState,
        batch_size: usize,
    ) -> Result<(), Error> {
        if let Some(bitmask_cell) = &state.token_bitmask {
            let mut bitmask_array = bitmask_cell.borrow_mut();
            let bitmask_shape = bitmask_array.shape();

            assert_eq!(
                bitmask_shape.len(),
                2,
                "Bitmask must be 2D [batch_size, vocab_size]"
            );
            assert_eq!(
                bitmask_shape[0], batch_size,
                "Bitmask batch size mismatch"
            );
            assert_eq!(
                bitmask_shape[1], self.vocab_size,
                "Bitmask vocab size mismatch"
            );

            let mut cpu_mask: Vec<u8> =
                vec![0u8; batch_size * self.vocab_size];

            for batch_idx in 0..batch_size {
                let batch_offset = batch_idx * self.vocab_size;
                let batch_mask_slice = &mut cpu_mask
                    [batch_offset..batch_offset + self.vocab_size];

                let mut shape_i64 = [self.vocab_size as i64];
                let mut bitmask_tensor = DLTensor {
                    data: batch_mask_slice.as_mut_ptr()
                        as *mut core::ffi::c_void,
                    device: DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DLDataType {
                        code: 1,
                        bits: 8,
                        lanes: 1,
                    },
                    shape: shape_i64.as_mut_ptr(),
                    strides: core::ptr::null_mut(),
                    byte_offset: 0,
                };

                let success = self.matcher.fill_next_token_bitmask(
                    &mut bitmask_tensor,
                    batch_idx as i32,
                    false,
                );
                if !success {
                    return Err(Error::GrammarError);
                }
            }

            let context = state.mtl_context();
            context.copy_from_view(&mut bitmask_array, cpu_mask.as_slice().into());

            Ok(())
        } else {
            Err(Error::GrammarError)
        }
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) {
        self.matcher.accept_token(token_id as i32);
    }

    pub fn reset(&mut self) {
        self.matcher.reset();
    }
}

