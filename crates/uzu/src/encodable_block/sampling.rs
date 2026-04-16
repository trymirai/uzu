//! Sampling kernel encodable.

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::sampling::{SamplingError, SamplingKernel},
    },
    session::parameter::SamplingMethod,
};

pub struct Sampling<B: Backend> {
    kernel: SamplingKernel<B>,
}

pub struct SamplingArguments<'a, B: Backend> {
    pub logits: &'a mut Allocation<B>,
    pub seeds: &'a Allocation<B>,
    pub seeds_offset: usize,
    pub bitmask: Option<&'a Allocation<B>>,
    pub bitmask_offset: usize,
    pub output: &'a mut Allocation<B>,
    pub sampling_method: SamplingMethod,
    pub batch_size: usize,
    pub vocab_size: usize,
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
        args: SamplingArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.kernel
            .encode(
                args.logits,
                args.seeds,
                args.seeds_offset,
                args.bitmask,
                args.bitmask_offset,
                args.output,
                args.sampling_method,
                args.batch_size,
                args.vocab_size,
                encoder,
            )
            .expect("Sampling encoding failed");

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
