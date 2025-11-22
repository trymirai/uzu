use std::{collections::HashMap, mem::size_of};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use thiserror::Error;

use crate::{
    Array,
    backends::metal::{
        KernelDataType, MTLContext, MTLError,
        forward_pass::{
            ArrayId, ForwardPassState, HashMapId,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub enum AttentionKernelVariant {
    SinglePass,
    TwoPass,
}

type PipelineKey = (usize, bool);

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

fn make_function_constants(has_sinks_value: bool) -> FunctionConstantValues {
    let function_constants = FunctionConstantValues::new();

    let has_mask_value = true;
    let query_transposed_value = false;
    let do_causal_value = false;
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
        &do_causal_value as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        22,
    ); // do_causal
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

        // Pre-generate all supported variants for both sink configurations
        for &has_sinks_value in &[false, true] {
            let function_constants = make_function_constants(has_sinks_value);

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
                    single_pass.insert((head_dim, has_sinks_value), pipeline);
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
                    two_pass_1.insert((head_dim, has_sinks_value), pipeline);
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
    ) -> bool {
        self.pipelines.single_pass.contains_key(&(head_dim, false))
    }

    pub fn supports_two_pass(
        &self,
        head_dim: usize,
    ) -> bool {
        self.pipelines.two_pass_1.contains_key(&(head_dim, false))
            && self.pipelines.two_pass_2.contains_key(&head_dim)
    }

    pub fn choose_variant(
        &self,
        sequence_length: usize,
        head_dim: usize,
    ) -> AttentionKernelVariant {
        if self.supports_two_pass(head_dim) && sequence_length > 1024 {
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
            .get(&(args.head_dim, has_sinks))
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
            .get(&(args.head_dim, has_sinks))
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

pub struct AttentionKernelEncodable {
    kernel: AttentionKernel,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
}

impl AttentionKernelEncodable {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        layer_index: usize,
        attention_scale: Option<f32>,
        has_sinks: bool,
    ) -> Result<Self, AttentionError> {
        let kernel = AttentionKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            layer_index,
            attention_scale,
            has_sinks,
        })
    }
}

impl EncodableWithState for AttentionKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let (
            suffix_length,
            num_heads,
            head_dim,
            num_groups,
            max_sequence_length,
        ) = {
            let suffix_length = state.active_suffix_length();

            let queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
            let queries_array = queries_binding[0].borrow();
            let num_heads = queries_array.shape()[0];
            let head_dim = queries_array.shape()[2];

            let keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
            let keys_array = keys_binding[0].borrow();
            let num_groups = keys_array.shape()[0];

            let key_cache_binding =
                state.arrays(&[ArrayId::Keys(self.layer_index)]);
            let key_cache_array = key_cache_binding[0].borrow();
            let max_sequence_length = key_cache_array.shape()[1];

            (
                suffix_length,
                num_heads,
                head_dim,
                num_groups,
                max_sequence_length,
            )
        };

        let (segment_prefix_length, window_length) = {
            let cache = state.cache_layers.borrow();
            let layer = cache.data[self.layer_index]
                .as_transformer()
                .expect("Attention kernel expects transformer layer state");
            (
                layer.projected_segment_prefix_length(
                    parameters.projection_step.unwrap_or(0),
                ),
                layer.window_length(),
            )
        };

        let sequence_length = segment_prefix_length + suffix_length;

        let gqa_factor = num_heads / num_groups;
        let scale =
            self.attention_scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

        let variant = self.kernel.choose_variant(sequence_length, head_dim);

        let rotated_queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let qkv_binding = state.arrays(&[ArrayId::QKV]);
        let key_cache_binding =
            state.arrays(&[ArrayId::Keys(self.layer_index)]);
        let value_cache_binding =
            state.arrays(&[ArrayId::Values(self.layer_index)]);
        let attention_bias_binding =
            state.hashmaps(&[HashMapId::AttentionBias]);
        let attention_output_binding =
            state.arrays(&[ArrayId::AttentionOutput]);

        let attention_bias_map = attention_bias_binding[0].clone();
        let mask_kv_seq_stride = 1;
        let mask_q_seq_stride = sequence_length as i32;
        let mask_head_stride = 0;
        let sinks_binding = if self.has_sinks {
            Some(state.arrays(&[ArrayId::AttentionSinks(self.layer_index)]))
        } else {
            None
        };

        let mut rotated_keys_array = rotated_keys_binding[0].borrow_mut();
        let rotated_keys_buffer = unsafe { rotated_keys_array.mtl_buffer() };

        let mut qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buffer = unsafe { qkv_array.mtl_buffer() };

        let mut key_cache_array = key_cache_binding[0].borrow_mut();
        let key_cache_buffer = unsafe { key_cache_array.mtl_buffer() };

        let mut value_cache_array = value_cache_binding[0].borrow_mut();
        let value_cache_buffer = unsafe { value_cache_array.mtl_buffer() };

        let mut queries_array = rotated_queries_binding[0].borrow_mut();
        let queries_buffer = unsafe { queries_array.mtl_buffer() };

        let mut attention_output_array =
            attention_output_binding[0].borrow_mut();
        let attention_output_buffer =
            unsafe { attention_output_array.mtl_buffer() };

        let attention_bias_buffer = attention_bias_map
            .get(&window_length)
            .map(|array| {
                let mut array_ref = array.borrow_mut();
                unsafe { array_ref.mtl_buffer().clone() }
            })
            .unwrap_or_else(|| {
                panic!(
                    "Attention bias buffer not found for window length {:?}",
                    window_length
                );
            });

        let partials_binding = state.arrays(&[ArrayId::AttentionPartials]);
        let sums_binding = state.arrays(&[ArrayId::AttentionSums]);
        let maxs_binding = state.arrays(&[ArrayId::AttentionMaxs]);

        let mut partials_array = partials_binding[0].borrow_mut();
        let partials_buffer = unsafe { partials_array.mtl_buffer() };

        let mut sums_array = sums_binding[0].borrow_mut();
        let sums_buffer = unsafe { sums_array.mtl_buffer() };

        let mut maxs_array = maxs_binding[0].borrow_mut();
        let maxs_buffer = unsafe { maxs_array.mtl_buffer() };

        let sinks_buffer = sinks_binding.as_ref().map(|binding| unsafe {
            binding[0].borrow_mut().mtl_buffer().clone()
        });

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();

        let compute_encoder = mtl_command_buffer.new_compute_command_encoder();

        if let Err(e) = self.kernel.encode_kv_cache_update(
            &compute_encoder,
            KVCacheUpdateArguments {
                rotated_keys_buffer: &rotated_keys_buffer,
                qkv_buffer: &qkv_buffer,
                key_cache_buffer: &key_cache_buffer,
                value_cache_buffer: &value_cache_buffer,
                num_groups,
                num_heads,
                head_dim,
                suffix_length,
                segment_prefix_length,
                max_sequence_length,
            },
        ) {
            eprintln!("Failed to encode KV cache update: {:?}", e);
            compute_encoder.end_encoding();
            return;
        }

        let k_head_stride = (max_sequence_length * head_dim) as i32;
        let k_seq_stride = head_dim as i32;
        let v_head_stride = (max_sequence_length * head_dim) as i32;
        let v_seq_stride = head_dim as i32;

        match variant {
            AttentionKernelVariant::SinglePass => {
                if let Err(e) = self.kernel.encode_single_pass(
                    &compute_encoder,
                    AttentionSinglePassArguments {
                        queries_buffer: &queries_buffer,
                        keys_buffer: &key_cache_buffer,
                        values_buffer: &value_cache_buffer,
                        output_buffer: &attention_output_buffer,
                        gqa_factor: gqa_factor as i32,
                        sequence_length: sequence_length as i32,
                        k_head_stride,
                        k_seq_stride,
                        v_head_stride,
                        v_seq_stride,
                        scale,
                        mask_buffer: Some(&attention_bias_buffer),
                        mask_kv_seq_stride,
                        mask_q_seq_stride,
                        mask_head_stride,
                        sinks_buffer: sinks_buffer.as_ref(),
                        num_heads,
                        suffix_length,
                        head_dim,
                    },
                ) {
                    eprintln!(
                        "Failed to encode single-pass attention: {:?}",
                        e
                    );
                }
            },
            AttentionKernelVariant::TwoPass => {
                if let Err(e) = self.kernel.encode_two_pass(
                    &compute_encoder,
                    AttentionTwoPassArguments {
                        queries_buffer: &queries_buffer,
                        keys_buffer: &key_cache_buffer,
                        values_buffer: &value_cache_buffer,
                        partials_buffer: &partials_buffer,
                        sums_buffer: &sums_buffer,
                        maxs_buffer: &maxs_buffer,
                        output_buffer: &attention_output_buffer,
                        gqa_factor: gqa_factor as i32,
                        sequence_length: sequence_length as i32,
                        k_head_stride,
                        k_seq_stride,
                        v_head_stride,
                        v_seq_stride,
                        scale,
                        mask_buffer: Some(&attention_bias_buffer),
                        mask_kv_seq_stride,
                        mask_q_seq_stride,
                        mask_head_stride,
                        sinks_buffer: sinks_buffer.as_ref(),
                        num_heads,
                        suffix_length,
                        head_dim,
                    },
                ) {
                    eprintln!("Failed to encode two-pass attention: {:?}", e);
                }
            },
        }

        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
