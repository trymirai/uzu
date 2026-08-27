use std::{
    collections::{HashMap, hash_map::Entry},
    mem::size_of,
    range::Range,
};

use parking_lot::Mutex;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        kernel::{RepetitionPenaltyKernel, UnifiedSamplingKernel},
    },
    data_type::DataType,
    encodable_block::batch_topology::BatchTopology,
};

#[cfg(backend = "cpu")]
mod gumbel;
mod prng;

#[cfg(backend = "cpu")]
pub use gumbel::{gumbel_float, revidx};
pub use prng::PRng;

pub struct Sampling<B: Backend> {
    vocab_size: u32,
    data_type: DataType,
    unified_kernels: Mutex<HashMap<UnifiedSamplingKey, <B::Kernels as Kernels>::UnifiedSamplingKernel>>,
}

impl<B: Backend> Sampling<B> {
    pub fn new(
        data_type: DataType,
        vocab_size: u32,
    ) -> Self {
        Self {
            vocab_size,
            data_type,
            unified_kernels: Mutex::new(HashMap::new()),
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
        suffix_repetition_length: Option<u32>,
    },
}

impl SamplingMethod {
    pub fn suffix_repetition_length(&self) -> Option<u32> {
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
        logits: impl BufferRef<Backend = B, Buffer: Sized>,
        seeds: Option<impl BufferRef<Backend = B>>,
        bitmask: Option<impl BufferRef<Backend = B>>,
        context_ring: Option<impl BufferRef<Backend = B>>,
        token_ids: Option<impl BufferRef<Backend = B>>,
        sampling_method: &SamplingMethod,
        batch_dim: &BatchTopology,
        sampling_range: Range<u32>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::GlobalBuffer, B::Error> {
        command_buffer.push_debug_group("sampling");

        let sampling_length = sampling_range.end - sampling_range.start;

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

            let mut logits_copy = command_buffer.allocate_scratch(logits.size())?;
            let copy_size = self.vocab_size as usize * sampling_length as usize * self.data_type.size_in_bytes();
            command_buffer.encode_copy(logits.subrange(..copy_size), logits_copy.subrange_mut(..copy_size));

            let repetition_penalty_kernel =
                <B::Kernels as Kernels>::RepetitionPenaltyKernel::new(command_buffer.context(), self.data_type)?;
            repetition_penalty_kernel.encode(
                logits,
                &mut logits_copy,
                context_ring.expect("context_ring is required for repetition_penalty"),
                token_ids.expect("token_ids is required for repetition_penalty"),
                repetition_penalty,
                suffix_repetition_length,
                self.vocab_size,
                sampling_range.start,
                sampling_length,
                command_buffer,
            );
            Some(logits_copy)
        } else {
            None
        };
        let logits = match &penalized_logits {
            Some(logits) => (logits as &dyn Buffer<Backend = B>).subrange(..),
            None => {
                let (buffer, range) = logits.parts();
                (buffer as &dyn Buffer<Backend = B>).subrange(range)
            },
        };

        let mut unified_kernels = self.unified_kernels.lock();
        let entry = unified_kernels.entry(key);
        let kernel = match entry {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => {
                let key = vacant.key();

                let kernel = <B::Kernels as Kernels>::UnifiedSamplingKernel::new(
                    command_buffer.context(),
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

        let mut output = command_buffer.context().create_buffer(sampling_length as usize * size_of::<u32>())?;

        kernel.encode(
            logits,
            &mut output,
            seeds,
            bitmask,
            temperature,
            top_k,
            top_p,
            min_p,
            self.vocab_size,
            sampling_length,
            command_buffer,
        );

        command_buffer.pop_debug_group();

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/sampling_test.rs"]
mod tests;
