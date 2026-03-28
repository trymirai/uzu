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
        let batch_size = state.sampling_length();
        let sampling_start = state.sampling_start();
        let sampling_method = state.sampling_method().unwrap();

        let logits = state.array(ArrayId::Logits);
        let seeds = state.array(ArrayId::TokenSeeds);
        let output = state.sampling_output().unwrap();

        let vocab_size = logits.shape()[1];
        let seeds_offset = seeds.offset() + sampling_start * std::mem::size_of::<u64>();

        let (bitmask_buffer, bitmask_offset) = state.token_bitmask().map_or((None, 0usize), |bitmask| {
            let bitmask_row_len = bitmask.shape()[1];
            let bitmask_offset = bitmask.offset() + sampling_start * bitmask_row_len * std::mem::size_of::<u32>();
            (Some(bitmask.buffer()), bitmask_offset)
        });
        let bitmask_borrow = bitmask_buffer.as_ref().map(|b| b.borrow());

        self.kernel
            .encode(
                logits.buffer().borrow_mut().deref_mut(),
                seeds.buffer().borrow().deref(),
                seeds_offset,
                bitmask_borrow.as_deref(),
                bitmask_offset,
                output.buffer().borrow_mut().deref_mut(),
                sampling_method,
                batch_size,
                vocab_size,
                encoder,
            )
            .expect("Sampling encoding failed");

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
