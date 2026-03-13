//! Sampling kernel encodable.

use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::sampling::{SamplingError, SamplingKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        assert!(state.sampling_output().is_some(), "Sampling output buffer must be pre-allocated");

        let logits_binding = state.arrays(&[ArrayId::Logits]);
        let logits = logits_binding[0].borrow();

        let logits_shape = logits.shape();
        let batch_size = state.sampling_length();
        if batch_size == 0 {
            return Ok(());
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
            (Some(bitmask.buffer()), bitmask_offset)
        });
        let bitmask_borrow = bitmask_buffer.as_ref().map(|b| b.borrow());
        let logits_buf_rc = logits.buffer();
        let mut logits_buf_borrow = logits_buf_rc.borrow_mut();
        let seeds_buf_rc = seeds.buffer();
        let seeds_buf_borrow = seeds_buf_rc.borrow();
        let output_buf_rc = output_buffer_ref.buffer();
        let mut output_buf_borrow = output_buf_rc.borrow_mut();
        if let Err(e) = self.kernel.encode(
            logits_buf_borrow.deref_mut(),
            Some(seeds_buf_borrow.deref()),
            seeds_offset,
            bitmask_borrow.as_deref(),
            bitmask_offset,
            output_buf_borrow.deref_mut(),
            sampling_method,
            batch_size,
            vocab_size,
            encoder,
        ) {
            panic!("Sampling encoding failed: {:?}", e);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
