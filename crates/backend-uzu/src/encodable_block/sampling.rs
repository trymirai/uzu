//! Sampling kernel encodable.

use ndarray::ArrayView2;

use crate::{
    DataType,
    array::{Array, ArrayContextExt},
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
    pub bitmask: Option<&'a Allocation<B>>,
    pub output: &'a mut Allocation<B>,
    pub sampling_method: SamplingMethod,
    pub batch_size: usize,
    pub vocab_size: usize,
}

#[derive(Clone)]
pub(crate) struct SamplingInputs<B: Backend> {
    pub seeds: Array<B>,
    pub bitmask: Option<Array<B>>,
}

impl<B: Backend> SamplingInputs<B> {
    pub(crate) fn bitmask_array_from_slice(
        context: &B::Context,
        row_count: usize,
        token_bitmask: &[u32],
        bitmask_row_len: usize,
    ) -> Array<B> {
        let mut array = context.create_array_zeros(&[row_count, bitmask_row_len], DataType::U32, "sampling_bitmask");
        let source =
            ArrayView2::from_shape((row_count, bitmask_row_len), token_bitmask).expect("Invalid token bitmask shape");
        array.copy_from_view(source);
        array
    }

    pub(crate) fn from_slices(
        context: &B::Context,
        token_seeds: &[u64],
        token_bitmask: Option<&[u32]>,
        bitmask_row_len: Option<usize>,
    ) -> Self {
        let seeds = context.create_array_from(&[token_seeds.len()], token_seeds, "sampling_seeds");

        let bitmask = match (token_bitmask, bitmask_row_len) {
            (Some(token_bitmask), Some(bitmask_row_len)) => {
                Some(Self::bitmask_array_from_slice(context, token_seeds.len(), token_bitmask, bitmask_row_len))
            },
            (None, None) => None,
            _ => panic!("bitmask data and row length must either both exist or both be absent"),
        };

        Self {
            seeds,
            bitmask,
        }
    }
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
        args: SamplingArguments<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), SamplingError<B>> {
        self.kernel.encode(
            args.logits,
            args.seeds,
            args.bitmask,
            args.output,
            args.sampling_method,
            args.batch_size,
            args.vocab_size,
            encoder,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
