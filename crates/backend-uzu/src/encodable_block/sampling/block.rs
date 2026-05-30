use std::mem::size_of;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{
            ArgmaxFinalKernel, ArgmaxMainKernel, BitmaskKernel, GumbelKernel, MinPKernel, TemperatureKernel,
            TopKKernel, TopPKernel,
        },
    },
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
};

pub struct SamplingBlock<B: Backend> {
    bitmask: <B::Kernels as Kernels>::BitmaskKernel,
    temperature: <B::Kernels as Kernels>::TemperatureKernel,
    topk: <B::Kernels as Kernels>::TopKKernel,
    topp: <B::Kernels as Kernels>::TopPKernel,
    minp: <B::Kernels as Kernels>::MinPKernel,
    gumbel: <B::Kernels as Kernels>::GumbelKernel,
    argmax_main: <B::Kernels as Kernels>::ArgmaxMainKernel,
    argmax_final: <B::Kernels as Kernels>::ArgmaxFinalKernel,
}

#[derive(Debug, Error)]
pub enum SamplingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("batch size must be greater than 0")]
    EmptyBatch,
    #[error("vocab size must be greater than 0")]
    EmptyVocab,
}

impl<B: Backend> SamplingBlock<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, SamplingError<B>> {
        let bitmask = <B::Kernels as Kernels>::BitmaskKernel::new(context, data_type, true)
            .map_err(SamplingError::BackendError)?;
        let temperature = <B::Kernels as Kernels>::TemperatureKernel::new(context, data_type, true)
            .map_err(SamplingError::BackendError)?;
        let topk =
            <B::Kernels as Kernels>::TopKKernel::new(context, data_type, true).map_err(SamplingError::BackendError)?;
        let topp =
            <B::Kernels as Kernels>::TopPKernel::new(context, data_type, true).map_err(SamplingError::BackendError)?;
        let minp =
            <B::Kernels as Kernels>::MinPKernel::new(context, data_type, true).map_err(SamplingError::BackendError)?;
        let gumbel = <B::Kernels as Kernels>::GumbelKernel::new(context, data_type, true)
            .map_err(SamplingError::BackendError)?;
        let argmax_main =
            <B::Kernels as Kernels>::ArgmaxMainKernel::new(context, data_type).map_err(SamplingError::BackendError)?;
        let argmax_final =
            <B::Kernels as Kernels>::ArgmaxFinalKernel::new(context).map_err(SamplingError::BackendError)?;

        Ok(Self {
            bitmask,
            temperature,
            topk,
            topp,
            minp,
            gumbel,
            argmax_main,
            argmax_final,
        })
    }

    pub fn encode(
        &self,
        logits: &mut Allocation<B>,
        seeds: &Allocation<B>,
        bitmask: Option<&Allocation<B>>,
        sampled_tokens: &mut Allocation<B>,
        sampling_method: SamplingMethod,
        batch_size: usize,
        vocab_size: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), SamplingError<B>> {
        if batch_size == 0 {
            return Err(SamplingError::EmptyBatch);
        }
        if vocab_size == 0 {
            return Err(SamplingError::EmptyVocab);
        }

        if let Some(bitmask) = bitmask {
            self.bitmask.encode(
                None::<&Allocation<B>>,
                bitmask,
                &mut *logits,
                batch_size as u32,
                vocab_size as u32,
                encoder,
            );
        }

        if let SamplingMethod::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            processing_order,
        } = sampling_method
        {
            if let Some(temperature) = temperature
                && processing_order == SamplingProcessingOrder::TemperatureThenFilters
            {
                self.temperature.encode(
                    None::<&Allocation<B>>,
                    &mut *logits,
                    batch_size as u32,
                    vocab_size as u32,
                    temperature,
                    encoder,
                );
            }

            if let Some(top_k) = top_k {
                self.topk.encode(
                    None::<&Allocation<B>>,
                    &mut *logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_k,
                    encoder,
                );
            }
            if let Some(top_p) = top_p {
                self.topp.encode(
                    None::<&Allocation<B>>,
                    &mut *logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_p,
                    encoder,
                );
            }
            if let Some(min_p) = min_p {
                self.minp.encode(
                    None::<&Allocation<B>>,
                    &mut *logits,
                    batch_size as u32,
                    vocab_size as u32,
                    min_p,
                    encoder,
                );
            }

            if let Some(temperature) = temperature
                && processing_order == SamplingProcessingOrder::FiltersThenTemperature
            {
                self.temperature.encode(
                    None::<&Allocation<B>>,
                    &mut *logits,
                    batch_size as u32,
                    vocab_size as u32,
                    temperature,
                    encoder,
                );
            }

            self.gumbel.encode(
                None::<&Allocation<B>>,
                seeds,
                0,
                &mut *logits,
                batch_size as u32,
                vocab_size as u32,
                encoder,
            );
        }

        let block_size = 1024;
        let grain_size = 4;
        let elements_per_group = block_size * grain_size;
        let vocab_groups_per_batch = vocab_size.div_ceil(elements_per_group);
        let partial_results_count = batch_size * vocab_groups_per_batch;
        let mut partial_results = encoder
            .allocate_scratch(partial_results_count * size_of::<ArgmaxPair>())
            .map_err(SamplingError::BackendError)?;
        self.argmax_main.encode(&*logits, &mut partial_results, batch_size as u32, vocab_size as u32, encoder);
        self.argmax_final.encode(&partial_results, sampled_tokens, batch_size as u32, vocab_size as u32, encoder);

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/sampling_top_p_test.rs"]
mod top_p_tests;
