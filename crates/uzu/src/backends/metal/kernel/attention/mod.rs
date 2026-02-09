use std::{collections::HashMap, mem::size_of, ptr::NonNull};

use objc2::rc::Retained;
use thiserror::Error;

use crate::backends::metal::{
    KernelDataType, MTLBuffer, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLContext, MTLDataType, MTLError,
    MTLFunctionConstantValues, MTLSize, ProtocolObject,
};

use crate::backends::common::gpu_types::{AttnMaskParams, AttnParams};

#[derive(Debug, Clone, Copy)]
pub enum AttentionKernelVariant {
    SinglePass,
    TwoPass,
}

type PipelineKey = (usize, bool, bool, bool); // (head_dim, has_sinks, is_causal, has_mask)

pub struct AttentionKernelPipelines {
    single_pass: HashMap<
        PipelineKey,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    >,
    two_pass_1: HashMap<
        PipelineKey,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    >,
    two_pass_2:
        HashMap<usize, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    kv_cache_update:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

pub struct AttentionKernel {
    data_type: KernelDataType,
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
    pub queries_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)
    pub keys_buffer: &'a ProtocolObject<dyn MTLBuffer>,    // buffer(1)
    pub values_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(2)
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(3)
    pub gqa_factor: i32,                                   // buffer(4)
    pub sequence_length: i32, // buffer(5) - sequence_length
    pub k_head_stride: i32,   // buffer(6)
    pub k_seq_stride: i32,    // buffer(7)
    pub v_head_stride: i32,   // buffer(8)
    pub v_seq_stride: i32,    // buffer(9)
    pub scale: f32,           // buffer(10)
    pub mask_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(11/12)
    pub mask_kv_seq_stride: i32, // buffer(13)
    pub mask_q_seq_stride: i32, // buffer(14)
    pub mask_head_stride: i32, // buffer(15)
    pub sinks_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(16)
    pub num_heads: usize,
    pub suffix_length: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

pub struct AttentionTwoPassArguments<'a> {
    pub queries_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)
    pub keys_buffer: &'a ProtocolObject<dyn MTLBuffer>,    // buffer(1)
    pub values_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(2)
    pub partials_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(3) - pass 1 output
    pub sums_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(4) - pass 1 output
    pub maxs_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(5) - pass 1 output
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(3) - pass 2 output
    pub gqa_factor: i32,                                  // buffer(6)
    pub sequence_length: i32, // buffer(7) - sequence_length
    pub k_head_stride: i32,   // buffer(8)
    pub k_seq_stride: i32,    // buffer(9)
    pub v_head_stride: i32,   // buffer(10)
    pub v_seq_stride: i32,    // buffer(11)
    pub scale: f32,           // buffer(12)
    pub mask_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(13/14)
    pub mask_kv_seq_stride: i32, // buffer(15)
    pub mask_q_seq_stride: i32, // buffer(16)
    pub mask_head_stride: i32, // buffer(17)
    pub sinks_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(18)
    pub num_heads: usize,
    pub suffix_length: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

pub struct KVCacheUpdateArguments<'a> {
    pub rotated_keys_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)
    pub qkv_buffer: &'a ProtocolObject<dyn MTLBuffer>,          // buffer(1)
    pub key_cache_buffer: &'a ProtocolObject<dyn MTLBuffer>,    // buffer(2)
    pub value_cache_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(3)
    pub num_groups: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub suffix_length: usize,
    pub segment_prefix_length: usize,
    pub max_sequence_length: usize,
}

pub struct AttentionGemmArguments<'a> {
    pub queries_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)
    pub keys_buffer: &'a ProtocolObject<dyn MTLBuffer>,    // buffer(1)
    pub values_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(2)
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(3)
    pub mask_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(6)
    pub sinks_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(7)
    pub num_heads: usize,
    pub num_groups: usize,
    pub suffix_length: usize,         // qL
    pub sequence_length: usize,       // kL (prefix + suffix)
    pub segment_prefix_length: usize, // qL_off
    pub max_sequence_length: usize,   // stride for K/V cache
    pub head_dim: usize,
    pub is_causal: bool,
    pub scale: f32,
}

fn make_function_constants(
    has_mask_value: bool,
    has_sinks_value: bool,
    is_causal_value: bool,
) -> Retained<MTLFunctionConstantValues> {
    let function_constants = MTLFunctionConstantValues::new();

    let query_transposed_value = false;
    let bool_mask_value = false;
    let float_mask_value = has_mask_value;

    function_constants.set_constant_value_type_at_index(
        NonNull::from(&has_mask_value).cast(),
        MTLDataType::Bool,
        20,
    ); // has_mask
    function_constants.set_constant_value_type_at_index(
        NonNull::from(&query_transposed_value).cast(),
        MTLDataType::Bool,
        21,
    ); // query_transposed
    function_constants.set_constant_value_type_at_index(
        NonNull::from(&bool_mask_value).cast(),
        MTLDataType::Bool,
        23,
    ); // bool_mask
    function_constants.set_constant_value_type_at_index(
        NonNull::from(&float_mask_value).cast(),
        MTLDataType::Bool,
        24,
    ); // float_mask
    function_constants.set_constant_value_type_at_index(
        NonNull::from(&is_causal_value).cast(),
        MTLDataType::Bool,
        22,
    ); // do_causal
    function_constants.set_constant_value_type_at_index(
        NonNull::from(&has_sinks_value).cast(),
        MTLDataType::Bool,
        25,
    ); // has_sinks

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
                for &has_mask_value in &[false, true] {
                    let function_constants = make_function_constants(
                        has_mask_value,
                        has_sinks_value,
                        is_causal_value,
                    );

                    for &head_dim in &supported_head_dims {
                        if let Ok(pipeline) = context.compute_pipeline_state(
                            &format!(
                                "attention_single_pass_{}_{}",
                                data_suffix, head_dim
                            ),
                            Some(&function_constants),
                        ) {
                            single_pass.insert(
                                (
                                    head_dim,
                                    has_sinks_value,
                                    is_causal_value,
                                    has_mask_value,
                                ),
                                pipeline,
                            );
                        }

                        if let Ok(pipeline) = context.compute_pipeline_state(
                            &format!(
                                "attention_2pass_1_{}_{}",
                                data_suffix, head_dim
                            ),
                            Some(&function_constants),
                        ) {
                            two_pass_1.insert(
                                (
                                    head_dim,
                                    has_sinks_value,
                                    is_causal_value,
                                    has_mask_value,
                                ),
                                pipeline,
                            );
                        }

                        if !two_pass_2.contains_key(&head_dim) {
                            if let Ok(pipeline) = context
                                .compute_pipeline_state(
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
        }

        let kv_cache_update = context
            .compute_pipeline_state(
                &format!("update_kv_cache_{}", data_suffix),
                None,
            )
            .ok();

        Ok(Self {
            data_type,
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
        has_mask: bool,
    ) -> bool {
        self.pipelines
            .single_pass
            .contains_key(&(head_dim, false, is_causal, has_mask))
    }

    pub fn supports_two_pass(
        &self,
        head_dim: usize,
        is_causal: bool,
        has_mask: bool,
    ) -> bool {
        self.pipelines
            .two_pass_1
            .contains_key(&(head_dim, false, is_causal, has_mask))
            && self.pipelines.two_pass_2.contains_key(&head_dim)
    }

    pub fn choose_variant(
        &self,
        sequence_length: usize,
        head_dim: usize,
        is_causal: bool,
        has_mask: bool,
    ) -> AttentionKernelVariant {
        if self.supports_two_pass(head_dim, is_causal, has_mask)
            && sequence_length > 1024
        {
            AttentionKernelVariant::TwoPass
        } else {
            AttentionKernelVariant::SinglePass
        }
    }

    pub fn encode_single_pass(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: AttentionSinglePassArguments,
    ) -> Result<(), AttentionError> {
        let has_sinks = args.sinks_buffer.is_some();
        let has_mask = args.mask_buffer.is_some();
        let pipeline = self
            .pipelines
            .single_pass
            .get(&(args.head_dim, has_sinks, args.is_causal, has_mask))
            .ok_or_else(|| AttentionError::UnsupportedHeadDim(args.head_dim))?;

        MTLComputeCommandEncoder::set_compute_pipeline_state(
            compute_encoder,
            pipeline,
        );

        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.queries_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.keys_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.values_buffer),
            0,
            2,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.output_buffer),
            0,
            3,
        );

        unsafe {
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.gqa_factor as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                4,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.sequence_length as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                5,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.k_head_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                6,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.k_seq_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                7,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.v_head_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                8,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.v_seq_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                9,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(&args.scale as *const f32 as *mut _),
                size_of::<f32>(),
                10,
            );
        }

        if let Some(mask_buffer) = args.mask_buffer {
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(mask_buffer),
                0,
                12,
            ); // float_mask
            unsafe {
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_kv_seq_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    13,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_q_seq_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    14,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_head_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    15,
                );
            }
        }

        if let Some(sinks_buffer) = args.sinks_buffer {
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(sinks_buffer),
                0,
                16,
            );
        }

        let threads_per_threadgroup = MTLSize {
            width: 32 * 32, // sequence_block_size * head_block_size
            height: 1,
            depth: 1,
        };

        let threadgroups_per_grid =
            MTLSize::new(args.num_heads, args.suffix_length, 1);

        compute_encoder.dispatch_threadgroups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );
        Ok(())
    }

    pub fn encode_two_pass(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: AttentionTwoPassArguments,
    ) -> Result<(), AttentionError> {
        let has_sinks = args.sinks_buffer.is_some();
        let has_mask = args.mask_buffer.is_some();
        let pass1_pipeline = self
            .pipelines
            .two_pass_1
            .get(&(args.head_dim, has_sinks, args.is_causal, has_mask))
            .ok_or_else(|| AttentionError::UnsupportedHeadDim(args.head_dim))?;

        let pass2_pipeline =
            self.pipelines.two_pass_2.get(&args.head_dim).ok_or_else(|| {
                AttentionError::UnsupportedHeadDim(args.head_dim)
            })?;

        MTLComputeCommandEncoder::set_compute_pipeline_state(
            compute_encoder,
            pass1_pipeline,
        );

        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.queries_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.keys_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.values_buffer),
            0,
            2,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.partials_buffer),
            0,
            3,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.sums_buffer),
            0,
            4,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.maxs_buffer),
            0,
            5,
        );

        unsafe {
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.gqa_factor as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                6,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.sequence_length as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                7,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.k_head_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                8,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.k_seq_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                9,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.v_head_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                10,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &args.v_seq_stride as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                11,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(&args.scale as *const f32 as *mut _),
                size_of::<f32>(),
                12,
            );
        }

        // Set mask buffer if present
        if let Some(mask_buffer) = args.mask_buffer {
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(mask_buffer),
                0,
                14,
            ); // float_mask
            unsafe {
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_kv_seq_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    15,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_q_seq_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    16,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &args.mask_head_stride as *const i32 as *mut _,
                    ),
                    size_of::<i32>(),
                    17,
                );
            }
        }

        if let Some(sinks_buffer) = args.sinks_buffer {
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(sinks_buffer),
                0,
                18,
            );
        }

        let total_blocks_count = 32u64;
        let pass1_threads_per_threadgroup = MTLSize {
            width: 8 * 32, // sequence_block_size * head_block_size
            height: 1,
            depth: 1,
        };
        let pass1_threadgroups_per_grid = MTLSize::new(
            args.num_heads,
            args.suffix_length,
            total_blocks_count as usize,
        );

        compute_encoder.dispatch_threadgroups(
            pass1_threadgroups_per_grid,
            pass1_threads_per_threadgroup,
        );

        MTLComputeCommandEncoder::set_compute_pipeline_state(
            compute_encoder,
            pass2_pipeline,
        );

        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.partials_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.sums_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.maxs_buffer),
            0,
            2,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.output_buffer),
            0,
            3,
        );

        let pass2_threads_per_threadgroup = MTLSize {
            width: 32 * 32,
            height: 1,
            depth: 1,
        };
        let pass2_threadgroups_per_grid =
            MTLSize::new(args.num_heads, args.suffix_length, 1);

        compute_encoder.dispatch_threadgroups(
            pass2_threadgroups_per_grid,
            pass2_threads_per_threadgroup,
        );

        Ok(())
    }

    pub fn encode_kv_cache_update(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: KVCacheUpdateArguments,
    ) -> Result<(), AttentionError> {
        let pipeline =
            self.pipelines.kv_cache_update.as_ref().ok_or_else(|| {
                AttentionError::FunctionNotFound(
                    "KV cache update kernel".to_string(),
                )
            })?;

        MTLComputeCommandEncoder::set_compute_pipeline_state(
            compute_encoder,
            pipeline,
        );

        // Set buffers
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.rotated_keys_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.qkv_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.key_cache_buffer),
            0,
            2,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.value_cache_buffer),
            0,
            3,
        );

        // Set constants
        unsafe {
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.num_groups as i32) as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                4,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.num_heads as i32) as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                5,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.head_dim as i32) as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                6,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.suffix_length as i32) as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                7,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.segment_prefix_length as i32) as *const i32
                        as *mut _,
                ),
                size_of::<i32>(),
                8,
            );
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(
                    &(args.max_sequence_length as i32) as *const i32 as *mut _,
                ),
                size_of::<i32>(),
                9,
            );
        }

        let threads_per_grid = MTLSize::new(
            args.num_groups,
            args.suffix_length,
            args.head_dim as usize,
        );

        let threadgroup_depth =
            std::cmp::min(args.head_dim.max(1), 64) as usize;

        MTLComputeCommandEncoder::dispatch_threads(
            compute_encoder,
            threads_per_grid,
            MTLSize {
                width: 1,
                height: 1,
                depth: threadgroup_depth,
            },
        );
        Ok(())
    }

    pub fn encode_gemm(
        &self,
        context: &MTLContext,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: AttentionGemmArguments,
    ) -> Result<(), AttentionError> {
        const BQ: usize = 32;
        const WM: u64 = 4;
        const WN: u64 = 1;

        if !matches!(args.head_dim, 64 | 128 | 256) {
            return Err(AttentionError::UnsupportedHeadDim(args.head_dim));
        }

        let bk: usize = if args.head_dim < 128 {
            32
        } else {
            16
        };

        let align_q = (args.suffix_length % BQ) == 0;
        let align_k = (args.sequence_length % bk) == 0;
        let has_mask = args.mask_buffer.is_some();
        let has_sinks = args.sinks_buffer.is_some();

        let fcv = MTLFunctionConstantValues::new();
        fcv.set_constant_value_type_at_index(
            NonNull::from(&align_q).cast(),
            MTLDataType::Bool,
            200,
        );
        fcv.set_constant_value_type_at_index(
            NonNull::from(&align_k).cast(),
            MTLDataType::Bool,
            201,
        );
        fcv.set_constant_value_type_at_index(
            NonNull::from(&has_mask).cast(),
            MTLDataType::Bool,
            300,
        );
        fcv.set_constant_value_type_at_index(
            NonNull::from(&args.is_causal).cast(),
            MTLDataType::Bool,
            301,
        );
        fcv.set_constant_value_type_at_index(
            NonNull::from(&has_sinks).cast(),
            MTLDataType::Bool,
            302,
        );

        // Kernel name matches gemm_attention.metal instantiations:
        // attention_gemm_{f16|bf16|f32}_{head_dim}_bk{bk}
        let type_name = match self.data_type {
            KernelDataType::Float16 => "f16",
            KernelDataType::BFloat16 => "bf16",
            KernelDataType::Float32 => "f32",
        };

        let function_name =
            format!("attention_gemm_{}_{}_bk{}", type_name, args.head_dim, bk);

        let cache_key = format!(
            "{}_aq{}_ak{}_m{}_c{}_s{}",
            function_name,
            align_q as u8,
            align_k as u8,
            has_mask as u8,
            args.is_causal as u8,
            has_sinks as u8
        );

        let pipeline = context
            .compute_pipeline_state_cached(
                &cache_key,
                &function_name,
                Some(&fcv),
            )
            .map_err(AttentionError::MetalError)?;

        MTLComputeCommandEncoder::set_compute_pipeline_state(
            compute_encoder,
            &pipeline,
        );

        // Buffers
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.queries_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.keys_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.values_buffer),
            0,
            2,
        );
        MTLComputeCommandEncoder::set_buffer(
            compute_encoder,
            Some(args.output_buffer),
            0,
            3,
        );

        // Params (all strides in elements)
        let q_head_stride = (args.suffix_length * args.head_dim) as i64;
        let q_seq_stride = args.head_dim as i64;

        let kv_head_stride = (args.max_sequence_length * args.head_dim) as i64;
        let kv_seq_stride = args.head_dim as i64;

        let o_head_stride = args.head_dim as i64;
        let o_seq_stride = (args.num_heads * args.head_dim) as i64;

        let nq = (args.suffix_length + BQ - 1) / BQ;
        let nk = (args.sequence_length + bk - 1) / bk;
        let nq_aligned = args.suffix_length / BQ;
        let nk_aligned = args.sequence_length / bk;

        let params = AttnParams {
            q_strides: [0, q_head_stride, q_seq_stride],
            k_strides: [0, kv_head_stride, kv_seq_stride],
            v_strides: [0, kv_head_stride, kv_seq_stride],
            o_strides: [0, o_head_stride, o_seq_stride],
            gqa_factor: (args.num_heads / args.num_groups) as i32,
            scale: args.scale,
            q_len: args.suffix_length as i32,
            k_len: args.sequence_length as i32,
            q_off: args.segment_prefix_length as i32,
            nq_aligned: nq_aligned as i32,
            q_rem: (args.suffix_length - nq_aligned * BQ) as i32,
            nk: nk as i32,
            nk_aligned: nk_aligned as i32,
            k_rem: (args.sequence_length - nk_aligned * bk) as i32,
        };

        unsafe {
            MTLComputeCommandEncoder::set_bytes(
                compute_encoder,
                NonNull::new_unchecked(&params as *const AttnParams as *mut _),
                size_of::<AttnParams>(),
                4,
            );
        }

        if let Some(mask_buffer) = args.mask_buffer {
            let mask_params = AttnMaskParams {
                // We use a shared bias matrix for all heads/batches.
                m_strides: [0, 0, args.sequence_length as i64],
            };
            unsafe {
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &mask_params as *const AttnMaskParams as *mut _,
                    ),
                    size_of::<AttnMaskParams>(),
                    5,
                );
            }
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(mask_buffer),
                0,
                6,
            );
        }

        if let Some(sinks_buffer) = args.sinks_buffer {
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(sinks_buffer),
                0,
                16,
            );
        }

        // Dispatch
        let threadgroups_per_grid =
            MTLSize::new(nq as usize, args.num_heads, 1);
        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: WM as usize,
            depth: WN as usize,
        };

        compute_encoder.dispatch_threadgroups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }
}
