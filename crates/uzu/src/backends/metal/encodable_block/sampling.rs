//! Sampling kernel encodable.

use std::rc::Rc;

use metal::{CommandBufferRef, ComputeCommandEncoderRef};

use super::{EncodableBlock, EncodingParameters};
use crate::backends::metal::{
    KernelDataType, MTLContext,
    forward_pass::{ArrayId, ForwardPassState},
    kernel::sampling::{ArgmaxStrategy, SamplingError, SamplingKernel},
};

pub struct Sampling {
    pub kernel: SamplingKernel,
}

impl Sampling {
    pub fn new(
        context: &Rc<MTLContext>,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
    ) -> Result<Self, SamplingError> {
        let kernel = SamplingKernel::new(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
        )?;
        Ok(Self {
            kernel,
        })
    }

    pub fn new_with_strategy(
        context: &Rc<MTLContext>,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
    ) -> Result<Self, SamplingError> {
        let kernel = SamplingKernel::new_with_strategy(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            argmax_strategy,
        )?;
        Ok(Self {
            kernel,
        })
    }
}

impl EncodableBlock for Sampling {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        assert!(
            state.sampling_output().is_some(),
            "Sampling output buffer must be pre-allocated"
        );

        let logits_binding = state.arrays(&[ArrayId::Logits]);
        let mut logits = logits_binding[0].borrow_mut();

        let logits_shape = {
            use crate::Array;
            logits.shape()
        };
        let batch_size = state.sampling_length();
        if batch_size == 0 {
            return;
        }
        let sampling_start = state.sampling_start();
        let vocab_size = logits_shape[1];

        let seeds_binding = state.arrays(&[ArrayId::TokenSeeds]);
        let mut seeds = seeds_binding[0].borrow_mut();

        let mut output_buffer_ref =
            state.sampling_output().unwrap().borrow_mut();

        let sampling_method = state.sampling_method().unwrap();
        let seeds_offset =
            seeds.buffer_offset() + sampling_start * std::mem::size_of::<u64>();

        let (bitmask_buffer, bitmask_offset) =
            state.token_bitmask().map_or((None, 0usize), |cell| {
                let mut bitmask = cell.borrow_mut();
                let bitmask_row_len = {
                    use crate::Array;
                    bitmask.shape()[1]
                };
                let bitmask_offset = bitmask.buffer_offset()
                    + sampling_start
                        * bitmask_row_len
                        * std::mem::size_of::<u32>();
                (Some(unsafe { bitmask.mtl_buffer().clone() }), bitmask_offset)
            });
        if let Err(e) = self.kernel.encode_with_encoder(
            unsafe { &logits.mtl_buffer() },
            unsafe { Some(&seeds.mtl_buffer()) },
            seeds_offset,
            bitmask_buffer.as_ref(),
            bitmask_offset,
            unsafe { &output_buffer_ref.mtl_buffer() },
            sampling_method,
            batch_size,
            vocab_size,
            encoder,
        ) {
            panic!("Sampling encoding failed: {:?}", e);
        }
    }
}
