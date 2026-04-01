use std::{
    cell::RefCell,
    mem::size_of,
    ops::{Deref, DerefMut},
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Context, Encoder, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel, BitmaskKernel, StochasticKernel},
    },
    session::parameter::SamplingMethod,
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

// StochasticKernel retains N_CANDIDATES=64 candidates; top_k must not exceed this.
const MAX_TOP_K: u32 = 64;

pub struct SamplingKernel<B: Backend> {
    bitmask: <B::Kernels as Kernels>::BitmaskKernel,
    argmax_implementation: ArgmaxImplementation<B>,
    stochastic: <B::Kernels as Kernels>::StochasticKernel,
    stochastic_masked: <B::Kernels as Kernels>::StochasticKernel,
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
    #[error("Stochastic: top_k={0} exceeds N_CANDIDATES={MAX_TOP_K}")]
    TopKTooLarge(u32),
    #[error("Stochastic: top_p={0} is not in (0, 1]")]
    TopPOutOfRange(f32),
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

        let stochastic = <B::Kernels as Kernels>::StochasticKernel::new(context, data_type, false)
            .map_err(SamplingError::BackendError)?;
        let stochastic_masked = <B::Kernels as Kernels>::StochasticKernel::new(context, data_type, true)
            .map_err(SamplingError::BackendError)?;

        Ok(Self {
            bitmask,
            argmax_implementation,
            stochastic,
            stochastic_masked,
            max_batch_size,
            max_vocab_size,
        })
    }

    pub fn encode(
        &self,
        mut logits_buffer: &mut B::Buffer,
        seeds_buffer: &B::Buffer,
        seeds_offset: usize,
        bitmask_buffer: Option<&B::Buffer>,
        bitmask_offset: usize,
        sampled_tokens_buffer: &mut B::Buffer,
        sampling_method: SamplingMethod,
        batch_size: usize,
        vocab_size: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), SamplingError<B>> {
        if batch_size > self.max_batch_size {
            return Err(SamplingError::BatchSizeExceeded(batch_size, self.max_batch_size));
        }
        if vocab_size > self.max_vocab_size {
            return Err(SamplingError::VocabSizeExceeded(vocab_size, self.max_vocab_size));
        }

        if let SamplingMethod::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            ..
        } = sampling_method
        {
            let top_k_val = top_k.unwrap_or(0);
            if top_k_val > MAX_TOP_K {
                return Err(SamplingError::TopKTooLarge(top_k_val));
            }
            if let Some(p) = top_p {
                if p <= 0.0 || p > 1.0 {
                    return Err(SamplingError::TopPOutOfRange(p));
                }
            }

            let kernel = if bitmask_buffer.is_some() {
                &self.stochastic_masked
            } else {
                &self.stochastic
            };
            kernel.encode(
                logits_buffer.deref(),
                (seeds_buffer, seeds_offset),
                sampled_tokens_buffer,
                bitmask_buffer.map(|b| (b, bitmask_offset)),
                batch_size as u32,
                vocab_size as u32,
                temperature.unwrap_or(1.0),
                top_k_val,
                top_p.unwrap_or(1.0),
                min_p.unwrap_or(0.0),
                encoder,
            );
            return Ok(());
        }

        // SamplingMethod::Greedy
        if let Some(bitmask_buffer) = bitmask_buffer {
            self.bitmask.encode(
                None::<&B::Buffer>,
                (bitmask_buffer, bitmask_offset),
                logits_buffer.deref_mut(),
                batch_size as u32,
                vocab_size as u32,
                encoder,
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
                    encoder,
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
                    encoder,
                );
                final_kernel.encode(
                    partial_results_buffer.borrow().deref(),
                    sampled_tokens_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    encoder,
                );
            },
        }

        Ok(())
    }
}
