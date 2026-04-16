//! Sampling kernel encodable.

use ndarray::ArrayView2;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, allocation_helpers,
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

pub(crate) struct SamplingInputs<B: Backend> {
    pub seeds: Allocation<B>,
    pub bitmask: Option<Allocation<B>>,
    pub bitmask_row_len: Option<usize>,
}

impl<B: Backend> SamplingInputs<B> {
    pub(crate) fn bitmask_allocation_from_slice(
        context: &B::Context,
        row_count: usize,
        token_bitmask: &[u32],
        bitmask_row_len: usize,
    ) -> Allocation<B> {
        let mut allocation =
            allocation_helpers::create_zeroed_allocation(context, &[row_count, bitmask_row_len], DataType::U32);
        let source =
            ArrayView2::from_shape((row_count, bitmask_row_len), token_bitmask).expect("Invalid token bitmask shape");
        allocation_helpers::copy_view_to_allocation(&mut allocation, source);
        allocation
    }

    pub(crate) fn from_slices(
        context: &B::Context,
        token_seeds: &[u64],
        token_bitmask: Option<&[u32]>,
        bitmask_row_len: Option<usize>,
    ) -> Self {
        let mut seeds = allocation_helpers::create_allocation(context, &[token_seeds.len()], DataType::U64);
        allocation_helpers::copy_slice_to_allocation(&mut seeds, token_seeds);

        let bitmask = match (token_bitmask, bitmask_row_len) {
            (Some(token_bitmask), Some(bitmask_row_len)) => {
                Some(Self::bitmask_allocation_from_slice(context, token_seeds.len(), token_bitmask, bitmask_row_len))
            },
            (None, None) => None,
            _ => panic!("bitmask data and row length must either both exist or both be absent"),
        };

        Self {
            seeds,
            bitmask,
            bitmask_row_len,
        }
    }

    pub(crate) fn bitmask_offset(
        &self,
        sampling_start: usize,
    ) -> usize {
        self.bitmask_row_len.map_or(0, |row_len| sampling_start * row_len * std::mem::size_of::<u32>())
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
