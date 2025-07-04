use std::{collections::HashMap, mem::size_of, rc::Rc};

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use thiserror::Error;

use crate::{
    backends::metal::{
        KernelDataType, MTLContext, MTLError,
        forward_pass::{
            ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    session::sampling_config::SamplingConfig,
};

const BLOCK_SIZE: usize = 1024;
const GRAIN_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArgmaxStrategy {
    SinglePass, // One threadgroup per batch item
    TwoPass,    // Multi-stage reduction
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CategoricalStrategy {
    SinglePass, // One threadgroup per batch item
    TwoPass,    // Multi-stage reduction for large vocabularies
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SamplingAlgorithm {
    ArgmaxSinglePass,
    ArgmaxTwoPass,
    TopP,
    CategoricalSinglePass,
    CategoricalTwoPass,
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

pub struct SamplingKernel {
    // Pipeline lookup by algorithm -> array of pipeline stages
    pipelines: HashMap<SamplingAlgorithm, Vec<MTLComputePipelineState>>,
    argmax_strategy: ArgmaxStrategy,
    categorical_strategy: CategoricalStrategy,
    partial_results_buffer: MTLBuffer,
    categorical_partial_buffer: MTLBuffer,
    max_batch_size: usize,
    max_vocab_size: usize,
    sampling_seed: u64,
    invocation_count: std::cell::Cell<u64>,
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
}

impl SamplingKernel {
    fn hash_seed(
        base_seed: u64,
        invocation: u64,
    ) -> u64 {
        let mut hash = base_seed.wrapping_add(invocation);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;
        hash
    }

    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        Self::new_with_strategy(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            ArgmaxStrategy::TwoPass,
            sampling_seed,
        )
    }

    pub fn new_with_strategy(
        context: &MTLContext,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        Self::new_with_strategies(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            argmax_strategy,
            CategoricalStrategy::SinglePass,
            sampling_seed,
        )
    }

    pub fn new_with_strategies(
        context: &MTLContext,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
        categorical_strategy: CategoricalStrategy,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        let data_suffix = data_type.function_name_suffix();
        let mut pipelines = HashMap::new();

        // Build argmax pipelines based on strategy
        match argmax_strategy {
            ArgmaxStrategy::SinglePass => {
                let pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!("batched_argmax_single_{}", data_suffix),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;
                pipelines.insert(
                    SamplingAlgorithm::ArgmaxSinglePass,
                    vec![pipeline],
                );
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

                pipelines.insert(
                    SamplingAlgorithm::ArgmaxTwoPass,
                    vec![main_pipeline, final_pipeline],
                );
            },
        }

        // Build top-p pipeline (single pass)
        let top_p_pipeline = context
            .compute_pipeline_state_with_reflection(
                &format!("batched_topp_main_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .map_err(SamplingError::MetalError)?;
        pipelines.insert(SamplingAlgorithm::TopP, vec![top_p_pipeline]);

        // Build categorical pipelines based on strategy
        match categorical_strategy {
            CategoricalStrategy::SinglePass => {
                let categorical_pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!("batched_categorical_main_{}", data_suffix),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;
                pipelines.insert(
                    SamplingAlgorithm::CategoricalSinglePass,
                    vec![categorical_pipeline],
                );
            },
            CategoricalStrategy::TwoPass => {
                let main_pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!(
                            "batched_categorical_main_2pass_{}",
                            data_suffix
                        ),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;

                let final_pipeline = context
                    .compute_pipeline_state_with_reflection(
                        &format!(
                            "batched_categorical_final_2pass_{}",
                            data_suffix
                        ),
                        None,
                    )
                    .map(|(pipeline, _)| pipeline)
                    .map_err(SamplingError::MetalError)?;

                pipelines.insert(
                    SamplingAlgorithm::CategoricalTwoPass,
                    vec![main_pipeline, final_pipeline],
                );
            },
        }

        let elements_per_group = BLOCK_SIZE * GRAIN_SIZE;
        let max_vocab_groups_per_batch =
            (max_vocab_size + elements_per_group - 1) / elements_per_group;
        let max_partial_results = max_batch_size * max_vocab_groups_per_batch;

        let partial_results_buffer = context.device.new_buffer(
            (max_partial_results * size_of::<ArgmaxPair>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Buffer for categorical 2-pass: stores CategoricalChunkResult per chunk
        // CategoricalChunkResult has 3 floats: max_logit, sum_exp, cumulative_prob
        let categorical_partial_buffer = context.device.new_buffer(
            (max_partial_results * size_of::<f32>() * 3) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipelines,
            argmax_strategy,
            categorical_strategy,
            partial_results_buffer,
            categorical_partial_buffer,
            max_batch_size,
            max_vocab_size,
            sampling_seed,
            invocation_count: std::cell::Cell::new(0),
        })
    }

    pub fn encode(
        &self,
        sampling_config: &SamplingConfig,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size: usize,
        vocab_size: usize,
        command_buffer: &MTLCommandBuffer,
    ) -> Result<(), SamplingError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_encoder(
            sampling_config,
            logits_buffer,
            sampled_tokens_buffer,
            batch_size,
            vocab_size,
            &compute_encoder,
        )?;
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_with_encoder(
        &self,
        sampling_config: &SamplingConfig,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
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

        let batch_size_u32 = batch_size as u32;
        let vocab_size_u32 = vocab_size as u32;

        let elements_per_group = BLOCK_SIZE * GRAIN_SIZE;
        let vocab_groups_per_batch =
            (vocab_size + elements_per_group - 1) / elements_per_group;

        let threads_per_threadgroup = MTLSize::new(BLOCK_SIZE as u64, 1, 1);

        match sampling_config {
            SamplingConfig::Argmax => self.encode_argmax(
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                vocab_groups_per_batch,
                threads_per_threadgroup,
                compute_encoder,
            ),
            SamplingConfig::TopP {
                top_p,
            } => self.encode_top_p(
                *top_p,
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                vocab_groups_per_batch,
                threads_per_threadgroup,
                compute_encoder,
            ),
            SamplingConfig::Categorical {
                temperature,
            } => self.encode_categorical(
                *temperature,
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                vocab_groups_per_batch,
                threads_per_threadgroup,
                compute_encoder,
            ),
        }
    }

    fn encode_argmax(
        &self,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        vocab_groups_per_batch: usize,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        match self.argmax_strategy {
            ArgmaxStrategy::SinglePass => self.encode_argmax_single_pass(
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                threads_per_threadgroup,
                compute_encoder,
            ),
            ArgmaxStrategy::TwoPass => self.encode_argmax_two_pass(
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                vocab_groups_per_batch,
                threads_per_threadgroup,
                compute_encoder,
            ),
        }
    }

    fn encode_argmax_single_pass(
        &self,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        compute_encoder.set_compute_pipeline_state(
            &self.pipelines[&SamplingAlgorithm::ArgmaxSinglePass][0],
        );
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(batch_size_u32 as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }

    fn encode_argmax_two_pass(
        &self,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        vocab_groups_per_batch: usize,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        // Main pass
        compute_encoder.set_compute_pipeline_state(
            &self.pipelines[&SamplingAlgorithm::ArgmaxTwoPass][0],
        );
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.partial_results_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(
            batch_size_u32 as u64,
            vocab_groups_per_batch as u64,
            1,
        );
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        // Final pass - two-pass argmax always has exactly 2 stages
        let final_pipeline =
            &self.pipelines[&SamplingAlgorithm::ArgmaxTwoPass][1];
        compute_encoder.set_compute_pipeline_state(final_pipeline);
        compute_encoder.set_buffer(0, Some(&self.partial_results_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let final_threadgroups = MTLSize::new(batch_size_u32 as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            final_threadgroups,
            threads_per_threadgroup,
        );

        Ok(())
    }

    fn encode_top_p(
        &self,
        top_p: f32,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        _vocab_groups_per_batch: usize,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        let current_invocation = self.invocation_count.get();
        self.invocation_count.set(current_invocation + 1);
        let seed = Self::hash_seed(self.sampling_seed, current_invocation);

        // Single pass - directly write to sampled_tokens_buffer
        compute_encoder.set_compute_pipeline_state(
            &self.pipelines[&SamplingAlgorithm::TopP][0],
        );
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<f32>() as u64,
            &top_p as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u64>() as u64,
            &seed as *const u64 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(batch_size_u32 as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }

    fn encode_categorical(
        &self,
        temperature: f32,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        vocab_groups_per_batch: usize,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        match self.categorical_strategy {
            CategoricalStrategy::SinglePass => self
                .encode_categorical_single_pass(
                    temperature,
                    logits_buffer,
                    sampled_tokens_buffer,
                    batch_size_u32,
                    vocab_size_u32,
                    threads_per_threadgroup,
                    compute_encoder,
                ),
            CategoricalStrategy::TwoPass => self.encode_categorical_two_pass(
                temperature,
                logits_buffer,
                sampled_tokens_buffer,
                batch_size_u32,
                vocab_size_u32,
                vocab_groups_per_batch,
                threads_per_threadgroup,
                compute_encoder,
            ),
        }
    }

    fn encode_categorical_single_pass(
        &self,
        temperature: f32,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        let current_invocation = self.invocation_count.get();
        self.invocation_count.set(current_invocation + 1);
        let seed = Self::hash_seed(self.sampling_seed, current_invocation);

        compute_encoder.set_compute_pipeline_state(
            &self.pipelines[&SamplingAlgorithm::CategoricalSinglePass][0],
        );
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<f32>() as u64,
            &temperature as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u64>() as u64,
            &seed as *const u64 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(batch_size_u32 as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }

    fn encode_categorical_two_pass(
        &self,
        temperature: f32,
        logits_buffer: &MTLBuffer,
        sampled_tokens_buffer: &MTLBuffer,
        batch_size_u32: u32,
        vocab_size_u32: u32,
        vocab_groups_per_batch: usize,
        threads_per_threadgroup: MTLSize,
        compute_encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), SamplingError> {
        let current_invocation = self.invocation_count.get();
        self.invocation_count.set(current_invocation + 1);
        let seed = Self::hash_seed(self.sampling_seed, current_invocation);

        // Main pass - parallel chunk processing
        compute_encoder.set_compute_pipeline_state(
            &self.pipelines[&SamplingAlgorithm::CategoricalTwoPass][0],
        );
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(
            1,
            Some(&self.categorical_partial_buffer),
            0,
        );
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<f32>() as u64,
            &temperature as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u64>() as u64,
            &seed as *const u64 as *const std::ffi::c_void,
        );

        let threadgroups_per_grid = MTLSize::new(
            batch_size_u32 as u64,
            vocab_groups_per_batch as u64,
            1,
        );
        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        // Final pass - combine results and sample
        let final_pipeline =
            &self.pipelines[&SamplingAlgorithm::CategoricalTwoPass][1];
        compute_encoder.set_compute_pipeline_state(final_pipeline);
        compute_encoder.set_buffer(0, Some(logits_buffer), 0);
        compute_encoder.set_buffer(
            1,
            Some(&self.categorical_partial_buffer),
            0,
        );
        compute_encoder.set_buffer(2, Some(sampled_tokens_buffer), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &batch_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &vocab_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &(vocab_groups_per_batch as u32) as *const u32
                as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<f32>() as u64,
            &temperature as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u64>() as u64,
            &seed as *const u64 as *const std::ffi::c_void,
        );

        let final_threadgroups = MTLSize::new(batch_size_u32 as u64, 1, 1);
        compute_encoder.dispatch_thread_groups(
            final_threadgroups,
            threads_per_threadgroup,
        );

        Ok(())
    }
}

pub struct SamplingKernelEncodable {
    pub kernel: SamplingKernel,
}

impl SamplingKernelEncodable {
    pub fn new(
        context: &Rc<MTLContext>,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        let kernel = SamplingKernel::new(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            sampling_seed,
        )?;
        Ok(Self {
            kernel,
        })
    }

    pub fn new_with_strategy(
        context: &Rc<MTLContext>,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        let kernel = SamplingKernel::new_with_strategy(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            argmax_strategy,
            sampling_seed,
        )?;
        Ok(Self {
            kernel,
        })
    }

    pub fn new_with_strategies(
        context: &Rc<MTLContext>,
        data_type: KernelDataType,
        max_batch_size: usize,
        max_vocab_size: usize,
        argmax_strategy: ArgmaxStrategy,
        categorical_strategy: CategoricalStrategy,
        sampling_seed: u64,
    ) -> Result<Self, SamplingError> {
        let kernel = SamplingKernel::new_with_strategies(
            context,
            data_type,
            max_batch_size,
            max_vocab_size,
            argmax_strategy,
            categorical_strategy,
            sampling_seed,
        )?;
        Ok(Self {
            kernel,
        })
    }
}

impl EncodableWithState for SamplingKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let sampling_config = state.sampling_config.unwrap_or_default();

        let logits_binding = state
            .arrays(&[crate::backends::metal::forward_pass::ArrayId::Logits]);
        let logits = logits_binding[0].borrow();
        let logits_shape = {
            use crate::Array;
            logits.shape()
        };
        let batch_size = logits_shape[0];
        let vocab_size = logits_shape[1];
        drop(logits);

        assert!(
            state.sampling_output.is_some(),
            "Sampling output buffer must be pre-allocated"
        );

        let logits_binding = state
            .arrays(&[crate::backends::metal::forward_pass::ArrayId::Logits]);
        let mut logits = logits_binding[0].borrow_mut();

        let mut output_buffer_ref =
            state.sampling_output.as_ref().unwrap().borrow_mut();

        let root_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        if let Err(e) = self.kernel.encode(
            &sampling_config,
            unsafe { &logits.mtl_buffer() },
            unsafe { &output_buffer_ref.mtl_buffer() },
            batch_size,
            vocab_size,
            &root_command_buffer,
        ) {
            panic!("Sampling encoding failed: {:?}", e);
        }

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            root_command_buffer.wait_until_completed();
        }
    }
}
