//! Sampling kernel encodable.

use std::rc::Rc;

use metal::CommandBufferRef;

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
        let batch_size = state.active_suffix_length();
        let vocab_size = logits_shape[1];

        let seeds_binding = state.arrays(&[ArrayId::TokenSeeds]);
        let mut seeds = seeds_binding[0].borrow_mut();

        let mut output_buffer_ref =
            state.sampling_output().unwrap().borrow_mut();

        let sampling_method = state.sampling_method().unwrap();
        let seeds_offset = seeds.buffer_offset();

        let root_command_buffer = command_buffer.to_owned();
        let bitmask_buffer = state
            .token_bitmask()
            .map(|cell| unsafe { cell.borrow_mut().mtl_buffer().clone() });
        if let Err(e) = self.kernel.encode(
            unsafe { &logits.mtl_buffer() },
            unsafe { Some(&seeds.mtl_buffer()) },
            seeds_offset,
            bitmask_buffer.as_ref(),
            unsafe { &output_buffer_ref.mtl_buffer() },
            sampling_method,
            batch_size,
            vocab_size,
            &root_command_buffer,
        ) {
            panic!("Sampling encoding failed: {:?}", e);
        }

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }
}
