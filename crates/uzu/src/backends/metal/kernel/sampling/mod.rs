use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use crate::{
    DataType,
    backends::metal::{KernelDataType, MTLContext, MTLError},
    session::parameter::SamplingMethod,
};

const BLOCK_SIZE: usize = 1024;
const ELEMENTWISE_GRAIN_SIZE: usize = 64;
const TWOPASS_ARGMAX_GRAIN_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArgmaxStrategy {
    SinglePass, // One threadgroup per batch item
    TwoPass,    // Multi-stage reduction
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
        pipeline: MTLComputePipelineState,
    },
    TwoPass {
        main_pipeline: MTLComputePipelineState,
        final_pipeline: MTLComputePipelineState,
        partial_results_buffer: MTLBuffer,
    },
}

pub struct SamplingKernel {
    bitmask_pipeline: MTLComputePipelineState,
    bitmask_applied_logits: MTLBuffer,
    temperature_pipeline: MTLComputePipelineState,
    temperature_applied_logits: MTLBuffer,
    topk_pipeline: MTLComputePipelineState,
    topk_applied_logits: MTLBuffer,
    topp_pipeline: MTLComputePipelineState,
    topp_applied_logits: MTLBuffer,
    minp_pipeline: MTLComputePipelineState,
    minp_applied_logits: MTLBuffer,
    gumbel_pipeline: MTLComputePipelineState,
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
        let data_suffix = data_type.function_name_suffix();
        let max_elements = max_batch_size * max_vocab_size;

        let bitmask_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_bitmask_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let bitmask_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let temperature_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_temperature_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let temperature_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let topk_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_topk_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let topk_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let topp_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_topp_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let topp_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let minp_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_minp_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let minp_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let gumbel_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_gumbel_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;

        let gumbel_applied_logits = context.device.new_buffer(
            (max_elements * Into::<DataType>::into(data_type).size_in_bytes())
                as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let argmax_implementation = match argmax_strategy {
            ArgmaxStrategy::SinglePass => {
                let pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!("batched_argmax_single_{}", data_suffix),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;

                ArgmaxImplementation::SinglePass {
                    pipeline,
                }
            },
            ArgmaxStrategy::TwoPass => {
                let main_pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!("batched_argmax_main_{}", data_suffix),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;

                let final_pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!("batched_argmax_final_{}", data_suffix),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;

                let elements_per_group = BLOCK_SIZE * TWOPASS_ARGMAX_GRAIN_SIZE;
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
                    main_pipeline,
                    final_pipeline,
                    partial_results_buffer,
                }
            },
        };

        Ok(Self {
            bitmask_pipeline,
            bitmask_applied_logits,
            temperature_pipeline,
            temperature_applied_logits,
            topk_pipeline,
            topk_applied_logits,
            topp_pipeline,
            topp_applied_logits,
            minp_pipeline,
            minp_applied_logits,
            gumbel_pipeline,
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
            self.encode_bitmask(
                last_logits_buffer,
                bitmask_buffer,
                bitmask_offset,
                &self.bitmask_applied_logits,
                batch_size as u32,
                vocab_size as u32,
                compute_encoder,
            )?;
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
                self.encode_temperature(
                    last_logits_buffer,
                    &self.temperature_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    temperature,
                    compute_encoder,
                )?;
                last_logits_buffer = &self.temperature_applied_logits;
            }

            if let Some(top_k) = top_k {
                self.encode_topk(
                    last_logits_buffer,
                    &self.topk_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_k,
                    compute_encoder,
                )?;
                last_logits_buffer = &self.topk_applied_logits;
            }

            if let Some(top_p) = top_p {
                self.encode_topp(
                    last_logits_buffer,
                    &self.topp_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    top_p,
                    compute_encoder,
                )?;
                last_logits_buffer = &self.topp_applied_logits;
            }

            if let Some(min_p) = min_p {
                self.encode_minp(
                    last_logits_buffer,
                    &self.minp_applied_logits,
                    batch_size as u32,
                    vocab_size as u32,
                    min_p,
                    compute_encoder,
                )?;
                last_logits_buffer = &self.minp_applied_logits;
            }

            self.encode_gumbel(
                last_logits_buffer,
                seeds_buffer.ok_or(SamplingError::StochasticWithoutSeed)?,
                seeds_offset,
                &self.gumbel_applied_logits,
                batch_size as u32,
                vocab_size as u32,
                compute_encoder,
            )?;
            last_logits_buffer = &self.gumbel_applied_logits;
        }

        match &self.argmax_implementation {
            ArgmaxImplementation::SinglePass {
                pipeline,
            } => self.encode_argmax_single_pass(
                pipeline,
                last_logits_buffer,
                sampled_tokens_buffer,
                batch_size,
                vocab_size,
                compute_encoder,
            ),
            ArgmaxImplementation::TwoPass {
                main_pipeline,
                final_pipeline,
                partial_results_buffer,
            } => self.encode_argmax_two_pass(
                main_pipeline,
                final_pipeline,
                last_logits_buffer,
                partial_results_buffer,
                sampled_tokens_buffer,
                batch_size,
                vocab_size,
                compute_encoder,
            ),
        }
    }

    pub fn encode_bitmask(
        &self,
        logits_buffer: &MTLBuffer,
        bitmask_buffer: &MTLBuffer,
        bitmask_offset: usize,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.bitmask_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(
            1,
            Some(bitmask_buffer),
            bitmask_offset as u64,
        );
        compute_encoder.set_buffer(2, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );

        let elements_in_group = BLOCK_SIZE * ELEMENTWISE_GRAIN_SIZE;
        let groups = (vocab_size + (elements_in_group as u32 - 1))
            / elements_in_group as u32;

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(groups as u64, batch_size as u64, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_temperature(
        &self,
        logits_buffer: &MTLBuffer,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        temperature: f32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.temperature_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &temperature as *const f32 as *const std::ffi::c_void,
        );

        let elements_in_group = BLOCK_SIZE * ELEMENTWISE_GRAIN_SIZE;
        let groups = (vocab_size + (elements_in_group as u32 - 1))
            / elements_in_group as u32;

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(groups as u64, batch_size as u64, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_topk(
        &self,
        logits_buffer: &MTLBuffer,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        top_k: u32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.topk_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &top_k as *const u32 as *const std::ffi::c_void,
        );

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_topp(
        &self,
        logits_buffer: &MTLBuffer,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        top_p: f32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.topp_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &top_p as *const f32 as *const std::ffi::c_void,
        );

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_minp(
        &self,
        logits_buffer: &MTLBuffer,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        min_p: f32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.minp_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<f32>() as u64,
            &min_p as *const f32 as *const std::ffi::c_void,
        );

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_gumbel(
        &self,
        logits_buffer: &MTLBuffer,
        seeds_buffer: &MTLBuffer,
        seeds_offset: usize,
        processed_logits_buffer: &MTLBuffer,
        batch_size: u32,
        vocab_size: u32,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(&self.gumbel_pipeline);

        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(seeds_buffer), seeds_offset as u64);
        compute_encoder.set_buffer(2, Some(processed_logits_buffer), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size as *const u32 as *const std::ffi::c_void,
        );

        let elements_in_group = BLOCK_SIZE * ELEMENTWISE_GRAIN_SIZE;
        let groups = (vocab_size + (elements_in_group as u32 - 1))
            / elements_in_group as u32;

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(groups as u64, batch_size as u64, 1),
            MTLSize::new(BLOCK_SIZE as u64, 1, 1),
        );

        Ok(())
    }

    pub fn encode_argmax_single_pass(
        &self,
        pipeline: &MTLComputePipelineState,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size: usize,
        vocab_size: usize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &(vocab_size as u32) as *const u32 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(BLOCK_SIZE as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }

    pub fn encode_argmax_two_pass(
        &self,
        main_pipeline: &MTLComputePipelineState,
        final_pipeline: &MTLComputePipelineState,
        logits_buffer: &MTLBuffer,
        partial_results_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size: usize,
        vocab_size: usize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        let elements_per_group = BLOCK_SIZE * TWOPASS_ARGMAX_GRAIN_SIZE;
        let vocab_groups_per_batch =
            (vocab_size + elements_per_group - 1) / elements_per_group;

        // Main pass
        compute_encoder.set_compute_pipeline_state(main_pipeline);
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(partial_results_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &(vocab_size as u32) as *const u32 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid =
            MTLSize::new(batch_size as u64, vocab_groups_per_batch as u64, 1);
        let threads_per_threadgroup = MTLSize::new(BLOCK_SIZE as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        // Final pass - two-pass argmax always has exactly 2 stages
        compute_encoder.set_compute_pipeline_state(final_pipeline);
        compute_encoder.set_buffer(0, Some(partial_results_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &(vocab_size as u32) as *const u32 as *const std::ffi::c_void,
        );

        let final_threadgroups = MTLSize::new(batch_size as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            final_threadgroups,
            threads_per_threadgroup,
        );

        Ok(())
    }
}
