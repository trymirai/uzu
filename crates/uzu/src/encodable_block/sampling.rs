//! Sampling kernel encodable.

use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::sampling::{ArgmaxStrategy, SamplingError, SamplingKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

use super::{EncodableBlock, EncodingParameters};

pub struct Sampling<B: Backend> {
    kernel: SamplingKernel<B>,
}

impl<B: Backend> Sampling<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        max_batch_size: usize,
        max_vocab_size: usize,
    ) -> Result<Self, SamplingError<B>> {
        let kernel = SamplingKernel::new(context, data_type, max_batch_size, max_vocab_size)?;
        Ok(Self {
            kernel,
        })
    }

    pub fn new_with_strategy(
        context: &B::Context,
        data_type: DataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
    ) -> Result<Self, SamplingError<B>> {
        let kernel =
            SamplingKernel::new_with_strategy(context, data_type, max_batch_size, max_vocab_size, argmax_strategy)?;
        Ok(Self {
            kernel,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Sampling<B> {
    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        _parameters: &EncodingParameters<B>,
    ) {
        assert!(state.sampling_output().is_some(), "Sampling output buffer must be pre-allocated");

        let logits_binding = state.arrays(&[ArrayId::Logits]);
        let logits = logits_binding[0].borrow();

        let logits_shape = logits.shape();
        let batch_size = state.sampling_length();
        if batch_size == 0 {
            return;
        }
        let sampling_start = state.sampling_start();
        let vocab_size = logits_shape[1];

        let seeds_binding = state.arrays(&[ArrayId::TokenSeeds]);
        let seeds = seeds_binding[0].borrow();

        let output_buffer_ref = state.sampling_output().unwrap().borrow();

        let sampling_method = state.sampling_method().unwrap();
        let seeds_offset = seeds.offset() + sampling_start * std::mem::size_of::<u64>();

        let (bitmask_buffer, bitmask_offset) = state.token_bitmask().map_or((None, 0usize), |cell| {
            let bitmask = cell.borrow();
            let bitmask_row_len = bitmask.shape()[1];
            let bitmask_offset = bitmask.offset() + sampling_start * bitmask_row_len * std::mem::size_of::<u32>();
            (Some(bitmask.buffer().clone()), bitmask_offset)
        });
        if let Err(e) = self.kernel.encode_with_encoder(
            logits.buffer(),
            Some(seeds.buffer()),
            seeds_offset,
            bitmask_buffer.as_ref(),
            bitmask_offset,
            output_buffer_ref.buffer(),
            sampling_method,
            batch_size,
            vocab_size,
            encoder,
        ) {
            panic!("Sampling encoding failed: {:?}", e);
        }
    }
}
