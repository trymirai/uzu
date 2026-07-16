use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    mem::size_of,
    ops::Range,
};

use crate::{
    backends::common::{
        Allocation, AllocationType, AsBufferRangeRef, Backend, Context, Encoder, Kernels,
        kernel::{RepetitionPenaltyKernel, TensorCopyKernel, UnifiedSamplingKernel},
    },
    data_type::DataType,
    encodable_block::batch_topology::BatchTopology,
};

mod gumbel;
mod prng;

pub use gumbel::{gumbel_float, revidx, speculator_sample};
pub use prng::PRng;

pub struct Sampling<B: Backend> {
    vocab_size: usize,
    data_type: DataType,
    unified_kernels: RefCell<HashMap<UnifiedSamplingKey, <B::Kernels as Kernels>::UnifiedSamplingKernel>>,
}

impl<B: Backend> Sampling<B> {
    pub fn new(
        data_type: DataType,
        vocab_size: usize,
    ) -> Self {
        Self {
            vocab_size,
            data_type,
            unified_kernels: RefCell::new(HashMap::new()),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum SamplingMethod {
    Greedy,
    Stochastic {
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        suffix_repetition_length: Option<usize>,
    },
}

impl SamplingMethod {
    pub fn suffix_repetition_length(&self) -> Option<usize> {
        match self {
            SamplingMethod::Greedy => None,
            SamplingMethod::Stochastic {
                suffix_repetition_length,
                ..
            } => *suffix_repetition_length,
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
struct UnifiedSamplingKey {
    is_stochastic: bool,
    has_bitmask: bool,
    has_temperature: bool,
    has_top_k: bool,
    has_top_p: bool,
    has_min_p: bool,
}

// TODO: repetition penalties shouldn't be stochastic only

impl<B: Backend> Sampling<B> {
    pub fn encode(
        &self,
        logits: &Allocation<B>,
        seeds: Option<&Allocation<B>>,
        bitmask: Option<&Allocation<B>>,
        context_ring: Option<&Allocation<B>>,
        token_ids: Option<&Allocation<B>>,
        sampling_method: &SamplingMethod,
        batch_dim: &BatchTopology,
        sampling_range: Range<usize>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let (is_stochastic, temperature, top_k, top_p, min_p, repetition_penalty, suffix_repetition_length) =
            match sampling_method {
                SamplingMethod::Greedy => (false, None, None, None, None, None, None),
                SamplingMethod::Stochastic {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    suffix_repetition_length,
                } => (true, *temperature, *top_k, *top_p, *min_p, *repetition_penalty, *suffix_repetition_length),
            };

        assert_eq!(is_stochastic, seeds.is_some(), "mismatch between sampling method type and seeds presence");

        let key = UnifiedSamplingKey {
            is_stochastic,
            has_bitmask: bitmask.is_some(),
            has_temperature: temperature.is_some(),
            has_top_k: top_k.is_some(),
            has_top_p: top_p.is_some(),
            has_min_p: min_p.is_some(),
        };

        // TODO: repetition penalty is vibe coded garbage, remove or rewrite properly
        let penalized_logits = if let Some(repetition_penalty) = repetition_penalty {
            assert!(batch_dim.is_flat(), "repetition_penalty currently only supports flat batches");
            let suffix_repetition_length =
                suffix_repetition_length.expect("suffix_repetition_length is required for repetition_penalty");

            let mut logits_copy = encoder.allocate_scratch(logits.as_buffer_range_ref().range().len())?;
            let tensor_copy = <B::Kernels as Kernels>::TensorCopyKernel::new(encoder.context(), self.data_type)?;
            tensor_copy.encode(logits, &mut logits_copy, (self.vocab_size * sampling_range.len()) as u32, encoder);

            let repetition_penalty_kernel =
                <B::Kernels as Kernels>::RepetitionPenaltyKernel::new(encoder.context(), self.data_type)?;
            repetition_penalty_kernel.encode(
                logits,
                &mut logits_copy,
                context_ring.expect("context_ring is required for repetition_penalty"),
                token_ids.expect("token_ids is required for repetition_penalty"),
                repetition_penalty,
                suffix_repetition_length as u32,
                self.vocab_size as u32,
                sampling_range.start as u32,
                sampling_range.len() as u32,
                encoder,
            );
            Some(logits_copy)
        } else {
            None
        };
        let logits = penalized_logits.as_ref().unwrap_or(logits);

        let mut unified_kernels = self.unified_kernels.borrow_mut();
        let entry = unified_kernels.entry(key);
        let kernel = match entry {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => {
                let key = vacant.key();

                let kernel = <B::Kernels as Kernels>::UnifiedSamplingKernel::new(
                    encoder.context(),
                    self.data_type,
                    key.is_stochastic,
                    key.has_bitmask,
                    key.has_temperature,
                    key.has_top_k,
                    key.has_top_p,
                    key.has_min_p,
                )?;

                vacant.insert(kernel)
            },
        };

        let mut output =
            encoder.context().create_allocation(sampling_range.len() * size_of::<u32>(), AllocationType::Global)?;

        kernel.encode(
            logits,
            &mut output,
            seeds,
            bitmask,
            temperature,
            top_k,
            top_p,
            min_p,
            self.vocab_size as u32,
            sampling_range.len() as u32,
            encoder,
        );

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
