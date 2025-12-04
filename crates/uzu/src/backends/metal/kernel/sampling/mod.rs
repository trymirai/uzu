use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoderRef, MTLResourceOptions,
};
use thiserror::Error;

use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MTLError,
        kernel::dsl::{
            ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel,
            BitmaskKernel, GumbelKernel, MinPKernel, TemperatureKernel,
            TopKKernel, TopPKernel,
        },
    },
    session::parameter::SamplingMethod,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArgmaxStrategy {
    SinglePass, // One threadgroup per batch item
    TwoPass,    // Two-stage reduction
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ArgmaxPair {
    value: f32,
    index: u32,
}

impl Default for ArgmaxPair {
    fn default() -> Self {
        Self {
            value: f32::NEG_INFINITY,
            index: u32::MAX,
        }
    }
}

enum ArgmaxImplementation {
    SinglePass {
        kernel: ArgmaxSingleKernel,
    },
    TwoPass {
        main_kernel: ArgmaxMainKernel,
        final_kernel: ArgmaxFinalKernel,
        partial_results_buffer: MTLBuffer,
    },
}

pub struct SamplingKernel {
    bitmask: BitmaskKernel,
    bitmask_applied_logits: MTLBuffer,
    temperature: TemperatureKernel,
    temperature_applied_logits: MTLBuffer,
    topk: TopKKernel,
    topk_applied_logits: MTLBuffer,
    topp: TopPKernel,
    topp_applied_logits: MTLBuffer,
    minp: MinPKernel,
    minp_applied_logits: MTLBuffer,
    gumbel: GumbelKernel,
    gumbel_applied_logits: MTLBuffer,
    argmax_implementation: ArgmaxImplementation,
    max_batch_size: usize,
    max_vocab_size: usize,
}

#[derive(Debug, Error)]
pub enum SamplingError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    #[error("Batch size {0} exceeds maximum {1}")]
    BatchSizeExceeded(usize, usize),
    #[error("Vocab size {0} exceeds maximum {1}")]
    VocabSizeExceeded(usize, usize),
    #[error("Stochastic sampling encode must have a seed")]
    StochasticWithoutSeed,
}

impl SamplingKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
    ) -> Result<Self, SamplingError> {
        Self::new_with_strategy(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            ArgmaxStrategy::TwoPass,
        )
    }

    pub fn new_with_strategy(
        context: &MTLContext,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
    ) -> Result<Self, SamplingError> {
        let max_elements = max_batch_size * max_vocab_size;

        let bitmask = BitmaskKernel::new(context, data_type)?;

        let bitmask_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let temperature = TemperatureKernel::new(context, data_type)?;

        let temperature_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let topk = TopKKernel::new(context, data_type)?;

        let topk_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let topp = TopPKernel::new(context, data_type)?;

        let topp_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let minp = MinPKernel::new(context, data_type)?;

        let minp_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let gumbel = GumbelKernel::new(context, data_type)?;

        let gumbel_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let argmax_implementation = match argmax_strategy {
            ArgmaxStrategy::SinglePass => {
                let kernel = ArgmaxSingleKernel::new(context, data_type)?;

                ArgmaxImplementation::SinglePass {
                    kernel,
                }
            },
            ArgmaxStrategy::TwoPass => {
                let main_kernel = ArgmaxMainKernel::new(context, data_type)?;
                let final_kernel = ArgmaxFinalKernel::new(context)?;

                let block_size = 1024;
                let grain_size = 4;

                let elements_per_group = block_size * grain_size;
                let max_vocab_groups_per_batch =
                    (max_vocab_size + elements_per_group - 1)
                        / elements_per_group;
                let max_partial_results =
                    max_batch_size * max_vocab_groups_per_batch;

                let partial_results_buffer = context.device.new_buffer(
                    (max_partial_results * size_of::<ArgmaxPair>()) as u64,
                    MTLResourceOptions::StorageModeShared,
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
            bitmask_applied_logits,
            temperature,
            temperature_applied_logits,
            topk,
            topk_applied_logits,
            topp,
            topp_applied_logits,
            minp,
            minp_applied_logits,
            gumbel,
            gumbel_applied_logits,
            argmax_implementation,
            max_batch_size,
            max_vocab_size,
        })
    }

    pub fn encode(
        &self,
        logits_buffer: &MTLBuffer,
        seeds_buffer: Option<&MTLBuffer>,
        seeds_offset: usize,
        bitmask_buffer: Option<&MTLBuffer>,
        bitmask_offset: usize,
        sampled_tokens_buffer: &MTLBuffer,
        sampling_method: SamplingMethod,
        batch_size: usize,
        vocab_size: usize,
        command_buffer: &MTLCommandBuffer,
    ) -> Result<(), SamplingError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_encoder(
            logits_buffer,
            seeds_buffer,
            seeds_offset,
            bitmask_buffer,
            bitmask_offset,
            sampled_tokens_buffer,
            sampling_method,
            batch_size,
            vocab_size,
            &compute_encoder,
        )?;
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_with_encoder(
        &self,
        logits_buffer: &MTLBuffer,
        seeds_buffer: Option<&MTLBuffer>,
        seeds_offset: usize,
        bitmask_buffer: Option<&MTLBuffer>,
        bitmask_offset: usize,
        sampled_tokens_buffer: &MTLBuffer,
        sampling_method: SamplingMethod,
        batch_size: usize,
        vocab_size: usize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        if batch_size > self.max_batch_size {
            return Err(SamplingError::BatchSizeExceeded(
                batch_size,
                self.max_batch_size,
            ));
        }
        if vocab_size > self.max_vocab_size {
            return Err(SamplingError::VocabSizeExceeded(
                vocab_size,
                self.max_vocab_size,
            ));
        }

        let mut last_logits_buffer = logits_buffer;

        if let Some(bitmask_buffer) = bitmask_buffer {
            self.bitmask.encode(
                last_logits_buffer,
                bitmask_buffer,
                bitmask_offset as u32,
                &self.bitmask_applied_logits,
                batch_size as u32,
                vocab_size as u32,
                compute_encoder,
            );
            last_logits_buffer = &self.bitmask_applied_logits;
        }

        if let SamplingMethod::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
        } = sampling_method
        {
            if let Some(temperature) = temperature {
                self.temperature.encode(
                    last_logits_buffer,
                    &self.temperature_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    temperature,
                    compute_encoder,
                );
                last_logits_buffer = &self.temperature_applied_logits;
            }

            if let Some(top_k) = top_k {
                self.topk.encode(
                    last_logits_buffer,
                    &self.topk_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_k,
                    compute_encoder,
                );
                last_logits_buffer = &self.topk_applied_logits;
            }

            if let Some(top_p) = top_p {
                self.topp.encode(
                    last_logits_buffer,
                    &self.topp_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_p,
                    compute_encoder,
                );
                last_logits_buffer = &self.topp_applied_logits;
            }

            if let Some(min_p) = min_p {
                self.minp.encode(
                    last_logits_buffer,
                    &self.minp_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    min_p,
                    compute_encoder,
                );
                last_logits_buffer = &self.minp_applied_logits;
            }

            self.gumbel.encode(
                last_logits_buffer,
                seeds_buffer.ok_or(SamplingError::StochasticWithoutSeed)?,
                &self.gumbel_applied_logits,
                batch_size as u32,
                vocab_size as u32,
                (seeds_offset / size_of::<u64>()) as u32,
                compute_encoder,
            );
            last_logits_buffer = &self.gumbel_applied_logits;
        }

        match &self.argmax_implementation {
            ArgmaxImplementation::SinglePass {
                kernel,
            } => {
                kernel.encode(
                    last_logits_buffer,
                    sampled_tokens_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    compute_encoder,
                );
            },
            ArgmaxImplementation::TwoPass {
                main_kernel,
                final_kernel,
                partial_results_buffer,
            } => {
                main_kernel.encode(
                    last_logits_buffer,
                    partial_results_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    compute_encoder,
                );
                final_kernel.encode(
                    partial_results_buffer,
                    sampled_tokens_buffer,
                    batch_size as u32,
                    vocab_size as u32,
                    compute_encoder,
                );
            },
        }

        Ok(())
    }
}
