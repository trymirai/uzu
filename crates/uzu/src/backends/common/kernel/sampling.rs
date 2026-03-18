use std::{
    cell::RefCell,
    mem::size_of,
    ops::{Deref, DerefMut},
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Context, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{
            ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel, BitmaskKernel, GumbelKernel, MinPKernel,
            TemperatureKernel, TopKKernel, TopPKernel,
        },
    },
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArgmaxStrategy {
    SinglePass, // One threadgroup per batch item
    TwoPass,    // Two-stage reduction
}

enum ArgmaxImplementation<B: Backend> {
    SinglePass {
        kernel: <B::Kernels as Kernels>::ArgmaxSingleKernel,
    },
    TwoPass {
        main_kernel: <B::Kernels as Kernels>::ArgmaxMainKernel,
        final_kernel: <B::Kernels as Kernels>::ArgmaxFinalKernel,
        partial_results_buffer: RefCell<B::Buffer>,
    },
}

pub struct SamplingKernel<B: Backend> {
    bitmask: <B::Kernels as Kernels>::BitmaskKernel,
    temperature: <B::Kernels as Kernels>::TemperatureKernel,
    topk: <B::Kernels as Kernels>::TopKKernel,
    topp: <B::Kernels as Kernels>::TopPKernel,
    minp: <B::Kernels as Kernels>::MinPKernel,
    gumbel: <B::Kernels as Kernels>::GumbelKernel,
    argmax_implementation: ArgmaxImplementation<B>,
    max_batch_size: usize,
    max_vocab_size: usize,
}

#[derive(Debug, Error)]
pub enum SamplingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    #[error("Batch size {0} exceeds maximum {1}")]
    BatchSizeExceeded(usize, usize),
    #[error("Vocab size {0} exceeds maximum {1}")]
    VocabSizeExceeded(usize, usize),
    #[error("Stochastic sampling encode must have a seed")]
    StochasticWithoutSeed,
}

impl<B: Backend> SamplingKernel<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        max_batch_size: usize,
        max_vocab_size: usize,
    ) -> Result<Self, SamplingError<B>> {
        Self::new_with_strategy(context, data_type, max_batch_size, max_vocab_size, ArgmaxStrategy::TwoPass)
    }

    pub fn new_with_strategy(
        context: &B::Context,
        data_type: DataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
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

        let argmax_implementation = match argmax_strategy {
            ArgmaxStrategy::SinglePass => {
                let kernel = <B::Kernels as Kernels>::ArgmaxSingleKernel::new(context, data_type)
                    .map_err(SamplingError::BackendError)?;

                ArgmaxImplementation::SinglePass {
                    kernel,
                }
            },
            ArgmaxStrategy::TwoPass => {
                let main_kernel = <B::Kernels as Kernels>::ArgmaxMainKernel::new(context, data_type)
                    .map_err(SamplingError::BackendError)?;
                let final_kernel =
                    <B::Kernels as Kernels>::ArgmaxFinalKernel::new(context).map_err(SamplingError::BackendError)?;

                let block_size = 1024;
                let grain_size = 4;

                let elements_per_group = block_size * grain_size;
                let max_vocab_groups_per_batch = (max_vocab_size + elements_per_group - 1) / elements_per_group;
                let max_partial_results = max_batch_size * max_vocab_groups_per_batch;

                let partial_results_buffer = RefCell::new(
                    context
                        .create_buffer(max_partial_results * size_of::<ArgmaxPair>())
                        .expect("Failed to create partial results buffer"),
                );

                ArgmaxImplementation::TwoPass {
                    main_kernel,
                    final_kernel,
                    partial_results_buffer,
                }
            },
        };

        Ok(Self {
            bitmask,
            temperature,
            topk,
            topp,
            minp,
            gumbel,
            argmax_implementation,
            max_batch_size,
            max_vocab_size,
        })
    }

    pub fn encode(
        &self,
        mut logits_buffer: &mut B::Buffer,
        seeds_buffer: Option<&B::Buffer>,
        seeds_offset: usize,
        bitmask_buffer: Option<&B::Buffer>,
        bitmask_offset: usize,
        sampled_tokens_buffer: &mut B::Buffer,
        sampling_method: SamplingMethod,
        batch_size: usize,
        vocab_size: usize,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), SamplingError<B>> {
        if batch_size > self.max_batch_size {
            return Err(SamplingError::BatchSizeExceeded(batch_size, self.max_batch_size));
        }
        if vocab_size > self.max_vocab_size {
            return Err(SamplingError::VocabSizeExceeded(vocab_size, self.max_vocab_size));
        }

        if let Some(bitmask_buffer) = bitmask_buffer {
            self.bitmask.encode(
                None::<&B::Buffer>,
                (bitmask_buffer, bitmask_offset),
                logits_buffer.deref_mut(),
                batch_size as u32,
                vocab_size as u32,
                command_buffer,
            );
        }

        let stochastic_config = match sampling_method {
            SamplingMethod::Greedy => None,
            SamplingMethod::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
            } => Some((temperature, top_k, top_p, min_p, SamplingProcessingOrder::TemperatureThenFilters)),
            SamplingMethod::AdvancedStochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                processing_order,
            } => Some((temperature, top_k, top_p, min_p, processing_order)),
        };

        if let Some((temperature, top_k, top_p, min_p, processing_order)) = stochastic_config {
            if processing_order == SamplingProcessingOrder::TemperatureThenFilters {
                if let Some(temperature) = temperature {
                    self.temperature.encode(
                        None::<&B::Buffer>,
                        logits_buffer.deref_mut(),
                        batch_size as u32,
                        vocab_size as u32,
                        temperature,
                        command_buffer,
                    );
                }
            }

            if let Some(top_k) = top_k {
                self.topk.encode(
                    None::<&B::Buffer>,
                    logits_buffer.deref_mut(),
                    batch_size as u32,
                    vocab_size as u32,
                    top_k,
                    command_buffer,
                );
            }
            if let Some(top_p) = top_p {
                self.topp.encode(
                    None::<&B::Buffer>,
                    logits_buffer.deref_mut(),
                    batch_size as u32,
                    vocab_size as u32,
                    top_p,
                    command_buffer,
                );
            }
            if let Some(min_p) = min_p {
                self.minp.encode(
                    None::<&B::Buffer>,
                    logits_buffer.deref_mut(),
                    batch_size as u32,
                    vocab_size as u32,
                    min_p,
                    command_buffer,
                );
            }

            if processing_order == SamplingProcessingOrder::FiltersThenTemperature {
                if let Some(temperature) = temperature {
                    self.temperature.encode(
                        None::<&B::Buffer>,
                        logits_buffer.deref_mut(),
                        batch_size as u32,
                        vocab_size as u32,
                        temperature,
                        command_buffer,
                    );
                }
            }

            self.gumbel.encode(
                None::<&B::Buffer>,
                (seeds_buffer.ok_or(SamplingError::StochasticWithoutSeed)?, seeds_offset),
                logits_buffer.deref_mut(),
                batch_size as u32,
                vocab_size as u32,
                command_buffer,
            );
        }

        match &self.argmax_implementation {
            ArgmaxImplementation::SinglePass {
                kernel,
            } => {
                kernel.encode(
                    logits_buffer.deref(),
                    sampled_tokens_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    command_buffer,
                );
            },
            ArgmaxImplementation::TwoPass {
                main_kernel,
                final_kernel,
                partial_results_buffer,
            } => {
                main_kernel.encode(
                    logits_buffer.deref(),
                    partial_results_buffer.borrow_mut().deref_mut(),
                    batch_size as u32,
                    vocab_size as u32,
                    command_buffer,
                );
                final_kernel.encode(
                    partial_results_buffer.borrow().deref(),
                    sampled_tokens_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    command_buffer,
                );
            },
        }

        Ok(())
    }
}
