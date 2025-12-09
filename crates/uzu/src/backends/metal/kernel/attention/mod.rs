use std::{collections::HashMap, mem::size_of};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};
use thiserror::Error;

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

#[derive(Debug, Clone, Copy)]
pub enum AttentionKernelVariant {
    SinglePass,
    TwoPass,
}

type PipelineKey = (usize, bool, bool); // (head_dim, has_sinks, is_causal)

pub struct AttentionKernelPipelines {
    single_pass: HashMap<PipelineKey, MTLComputePipelineState>,
    two_pass_1: HashMap<PipelineKey, MTLComputePipelineState>,
    two_pass_2: HashMap<usize, MTLComputePipelineState>,
    kv_cache_update: Option<MTLComputePipelineState>,
}

pub struct AttentionKernel {
    pipelines: AttentionKernelPipelines,
}

#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    #[error("Unsupported head dimension: {0}")]
    UnsupportedHeadDim(usize),
}

pub struct AttentionSinglePassArguments<'a> {
    pub queries_buffer: &'a MTLBuffer, // buffer(0)
    pub keys_buffer: &'a MTLBuffer,    // buffer(1)
    pub values_buffer: &'a MTLBuffer,  // buffer(2)
    pub output_buffer: &'a MTLBuffer,  // buffer(3)
    pub gqa_factor: i32,               // buffer(4)
    pub sequence_length: i32,          // buffer(5) - sequence_length
    pub k_head_stride: i32,            // buffer(6)
    pub k_seq_stride: i32,             // buffer(7)
    pub v_head_stride: i32,            // buffer(8)
    pub v_seq_stride: i32,             // buffer(9)
    pub scale: f32,                    // buffer(10)
    pub mask_buffer: Option<&'a MTLBuffer>, // buffer(11/12)
    pub mask_kv_seq_stride: i32,       // buffer(13)
    pub mask_q_seq_stride: i32,        // buffer(14)
    pub mask_head_stride: i32,         // buffer(15)
    pub sinks_buffer: Option<&'a MTLBuffer>, // buffer(16)
    pub num_heads: usize,
    pub suffix_length: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

pub struct AttentionTwoPassArguments<'a> {
    pub queries_buffer: &'a MTLBuffer,  // buffer(0)
    pub keys_buffer: &'a MTLBuffer,     // buffer(1)
    pub values_buffer: &'a MTLBuffer,   // buffer(2)
    pub partials_buffer: &'a MTLBuffer, // buffer(3) - pass 1 output
    pub sums_buffer: &'a MTLBuffer,     // buffer(4) - pass 1 output
    pub maxs_buffer: &'a MTLBuffer,     // buffer(5) - pass 1 output
    pub output_buffer: &'a MTLBuffer,   // buffer(3) - pass 2 output
    pub gqa_factor: i32,                // buffer(6)
    pub sequence_length: i32,           // buffer(7) - sequence_length
    pub k_head_stride: i32,             // buffer(8)
    pub k_seq_stride: i32,              // buffer(9)
    pub v_head_stride: i32,             // buffer(10)
    pub v_seq_stride: i32,              // buffer(11)
    pub scale: f32,                     // buffer(12)
    pub mask_buffer: Option<&'a MTLBuffer>, // buffer(13/14)
    pub mask_kv_seq_stride: i32,        // buffer(15)
    pub mask_q_seq_stride: i32,         // buffer(16)
    pub mask_head_stride: i32,          // buffer(17)
    pub sinks_buffer: Option<&'a MTLBuffer>, // buffer(18)
    pub num_heads: usize,
    pub suffix_length: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

pub struct KVCacheUpdateArguments<'a> {
    pub rotated_keys_buffer: &'a MTLBuffer, // buffer(0)
    pub qkv_buffer: &'a MTLBuffer,          // buffer(1)
    pub key_cache_buffer: &'a MTLBuffer,    // buffer(2)
    pub value_cache_buffer: &'a MTLBuffer,  // buffer(3)
    pub num_groups: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub suffix_length: usize,
    pub segment_prefix_length: usize,
    pub max_sequence_length: usize,
}

fn make_function_constants(
    has_sinks_value: bool,
    is_causal_value: bool,
) -> FunctionConstantValues {
    let function_constants = FunctionConstantValues::new();

    let has_mask_value = true;
    let query_transposed_value = false;
    let bool_mask_value = false;
    let float_mask_value = true;

    function_constants.set_constant_value_at_index(
        &has_mask_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        20,
    ); // has_mask
    function_constants.set_constant_value_at_index(
        &query_transposed_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        21,
    ); // query_transposed
    function_constants.set_constant_value_at_index(
        &bool_mask_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        23,
    ); // bool_mask
    function_constants.set_constant_value_at_index(
        &float_mask_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        24,
    ); // float_mask
    function_constants.set_constant_value_at_index(
        &has_sinks_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        25,
    ); // has_sinks
    function_constants.set_constant_value_at_index(
        &is_causal_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        26,
    ); // is_causal

    function_constants
}

impl AttentionKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, AttentionError> {
        let data_suffix = data_type.function_name_suffix();

        let supported_head_dims = [64, 128, 256];
        let mut single_pass = HashMap::new();
        let mut two_pass_1 = HashMap::new();
        let mut two_pass_2 = HashMap::new();

        // Pre-generate all supported variants for sinks and causal configurations
        for &has_sinks_value in &[false, true] {
            for &is_causal_value in &[false, true] {
                let function_constants =
                    make_function_constants(has_sinks_value, is_causal_value);

                for &head_dim in &supported_head_dims {
                    if let Ok((pipeline, _)) = context
                        .compute_pipeline_state_with_reflection(
                            &format!(
                                "attention_single_pass_{}_{}",
                                data_suffix, head_dim
                            ),
                            Some(&function_constants),
                        )
                    {
                        single_pass.insert(
                            (head_dim, has_sinks_value, is_causal_value),
                            pipeline,
                        );
                    }

                    if let Ok((pipeline, _)) = context
                        .compute_pipeline_state_with_reflection(
                            &format!(
                                "attention_2pass_1_{}_{}",
                                data_suffix, head_dim
                            ),
                            Some(&function_constants),
                        )
                    {
                        two_pass_1.insert(
                            (head_dim, has_sinks_value, is_causal_value),
                            pipeline,
                        );
                    }

                    if !two_pass_2.contains_key(&head_dim) {
                        if let Ok((pipeline, _)) = context
                            .compute_pipeline_state_with_reflection(
                                &format!(
                                    "attention_2pass_2_{}_{}",
                                    data_suffix, head_dim
                                ),
                                Some(&function_constants),
                            )
                        {
                            two_pass_2.insert(head_dim, pipeline);
                        }
                    }
                }
            }
        }

        let kv_cache_update = context
            .compute_pipeline_state_with_reflection(
                &format!("update_kv_cache_{}", data_suffix),
                None,
            )
            .map(|(pipeline, _)| pipeline)
            .ok();

        Ok(Self {
            pipelines: AttentionKernelPipelines {
                single_pass,
                two_pass_1,
                two_pass_2,
                kv_cache_update,
            },
        })
    }

    pub fn supports_single_pass(
        &self,
        head_dim: usize,
        is_causal: bool,
    ) -> bool {
        self.pipelines.single_pass.contains_key(&(head_dim, false, is_causal))
    }

    pub fn supports_two_pass(
        &self,
        head_dim: usize,
        is_causal: bool,
    ) -> bool {
        self.pipelines.two_pass_1.contains_key(&(head_dim, false, is_causal))
            && self.pipelines.two_pass_2.contains_key(&head_dim)
    }

    pub fn choose_variant(
        &self,
        sequence_length: usize,
        head_dim: usize,
        is_causal: bool,
    ) -> AttentionKernelVariant {
        if self.supports_two_pass(head_dim, is_causal) && sequence_length > 1024
        {
            AttentionKernelVariant::TwoPass
        } else {
            AttentionKernelVariant::SinglePass
        }
    }

    pub fn encode_single_pass(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: AttentionSinglePassArguments,
    ) -> Result<(), AttentionError> {
        let has_sinks = args.sinks_buffer.is_some();
        let pipeline = self
            .pipelines
            .single_pass
            .get(&(args.head_dim, has_sinks, args.is_causal))
            .ok_or_else(|| AttentionError::UnsupportedHeadDim(args.head_dim))?;

        compute_encoder.set_compute_pipeline_state(pipeline);

        compute_encoder.set_buffer(0, Some(args.queries_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.keys_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.values_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.output_buffer), 0);

        compute_encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &args.gqa_factor as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.sequence_length as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.k_head_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.k_seq_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &args.v_head_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &args.v_seq_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<f32>() as u64,
            &args.scale as *const f32 as *const _,
        );

        if let Some(mask_buffer) = args.mask_buffer {
            compute_encoder.set_buffer(12, Some(mask_buffer), 0); // float_mask
            compute_encoder.set_bytes(
                13,
                size_of::<i32>() as u64,
                &args.mask_kv_seq_stride as *const i32 as *const _,
            );
            compute_encoder.set_bytes(
                14,
                size_of::<i32>() as u64,
                &args.mask_q_seq_stride as *const i32 as *const _,
            );
            compute_encoder.set_bytes(
                15,
                size_of::<i32>() as u64,
                &args.mask_head_stride as *const i32 as *const _,
            );
        }

        if let Some(sinks_buffer) = args.sinks_buffer {
            compute_encoder.set_buffer(16, Some(sinks_buffer), 0);
        }

        let threads_per_threadgroup = MTLSize {
            width: 32 * 32, // sequence_block_size * head_block_size
            height: 1,
            depth: 1,
        };

        let threadgroups_per_grid = MTLSize {
            width: args.num_heads as u64,
            height: args.suffix_length as u64,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );
        Ok(())
    }

    pub fn encode_two_pass(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: AttentionTwoPassArguments,
    ) -> Result<(), AttentionError> {
        let has_sinks = args.sinks_buffer.is_some();
        let pass1_pipeline = self
            .pipelines
            .two_pass_1
            .get(&(args.head_dim, has_sinks, args.is_causal))
            .ok_or_else(|| AttentionError::UnsupportedHeadDim(args.head_dim))?;

        let pass2_pipeline =
            self.pipelines.two_pass_2.get(&args.head_dim).ok_or_else(|| {
                AttentionError::UnsupportedHeadDim(args.head_dim)
            })?;

        compute_encoder.set_compute_pipeline_state(pass1_pipeline);

        compute_encoder.set_buffer(0, Some(args.queries_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.keys_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.values_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.partials_buffer), 0);
        compute_encoder.set_buffer(4, Some(args.sums_buffer), 0);
        compute_encoder.set_buffer(5, Some(args.maxs_buffer), 0);

        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.gqa_factor as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.sequence_length as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &args.k_head_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &args.k_seq_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &args.v_head_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<i32>() as u64,
            &args.v_seq_stride as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.scale as *const f32 as *const _,
        );

        // Set mask buffer if present
        if let Some(mask_buffer) = args.mask_buffer {
            compute_encoder.set_buffer(14, Some(mask_buffer), 0); // float_mask
            compute_encoder.set_bytes(
                15,
                size_of::<i32>() as u64,
                &args.mask_kv_seq_stride as *const i32 as *const _,
            );
            compute_encoder.set_bytes(
                16,
                size_of::<i32>() as u64,
                &args.mask_q_seq_stride as *const i32 as *const _,
            );
            compute_encoder.set_bytes(
                17,
                size_of::<i32>() as u64,
                &args.mask_head_stride as *const i32 as *const _,
            );
        }

        if let Some(sinks_buffer) = args.sinks_buffer {
            compute_encoder.set_buffer(18, Some(sinks_buffer), 0);
        }

        let total_blocks_count = 32u64;
        let pass1_threads_per_threadgroup = MTLSize {
            width: 8 * 32, // sequence_block_size * head_block_size
            height: 1,
            depth: 1,
        };
        let pass1_threadgroups_per_grid = MTLSize {
            width: args.num_heads as u64,
            height: args.suffix_length as u64,
            depth: total_blocks_count,
        };

        compute_encoder.dispatch_thread_groups(
            pass1_threadgroups_per_grid,
            pass1_threads_per_threadgroup,
        );

        compute_encoder.set_compute_pipeline_state(pass2_pipeline);

        compute_encoder.set_buffer(0, Some(args.partials_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.sums_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.maxs_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.output_buffer), 0);

        let pass2_threads_per_threadgroup = MTLSize {
            width: 32 * 32,
            height: 1,
            depth: 1,
        };
        let pass2_threadgroups_per_grid = MTLSize {
            width: args.num_heads as u64,
            height: args.suffix_length as u64,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(
            pass2_threadgroups_per_grid,
            pass2_threads_per_threadgroup,
        );

        Ok(())
    }

    pub fn encode_kv_cache_update(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: KVCacheUpdateArguments,
    ) -> Result<(), AttentionError> {
        let pipeline =
            self.pipelines.kv_cache_update.as_ref().ok_or_else(|| {
                AttentionError::FunctionNotFound(
                    "KV cache update kernel".to_string(),
                )
            })?;

        compute_encoder.set_compute_pipeline_state(pipeline);

        // Set buffers
        compute_encoder.set_buffer(0, Some(args.rotated_keys_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.qkv_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.key_cache_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.value_cache_buffer), 0);

        // Set constants
        compute_encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &(args.num_groups as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &(args.num_heads as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &(args.head_dim as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &(args.suffix_length as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<i32>() as u64,
            &(args.segment_prefix_length as i32) as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<i32>() as u64,
            &(args.max_sequence_length as i32) as *const i32 as *const _,
        );

        let threads_per_grid = MTLSize {
            width: args.num_groups as u64,
            height: args.suffix_length as u64,
            depth: args.head_dim as u64,
        };

        compute_encoder.dispatch_threads(
            threads_per_grid,
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}
