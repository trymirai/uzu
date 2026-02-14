//! Attention kernel encodable.

use std::collections::HashMap;

use super::{EncodableBlock, Metal};
use crate::{
    DataType,
    backends::{
        common::kernel::{AttentionSinglePassKernel, AttentionUpdateKVCacheKernel},
        metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext, ProtocolObject, Retained,
            kernel::{
                attention::{
                    AttentionError, AttentionGemmArguments, AttentionKernel, AttentionKernelVariant,
                    AttentionTwoPassArguments,
                },
                dsl::{AttentionSinglePassMetalKernel, AttentionUpdateKVCacheMetalKernel},
            },
        },
    },
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState, HashMapId},
};

fn env_gemm_attention_enabled() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var("UZU_USE_GEMM_ATTENTION").ok();
        let Some(raw) = raw else {
            return true;
        };
        let v = raw.trim().to_ascii_uppercase();
        match v.as_str() {
            "1" | "YES" | "TRUE" | "ON" => true,
            "0" | "NO" | "FALSE" | "OFF" => false,
            _ => true,
        }
    })
}

pub struct Attention {
    kernel: AttentionKernel,
    single_pass_kernels: HashMap<KernelKey, AttentionSinglePassMetalKernel>,
    update_kv_cache_kernel: AttentionUpdateKVCacheMetalKernel,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
    is_causal: bool,
    sliding_window_size: Option<usize>,
}

impl Attention {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        layer_index: usize,
        attention_scale: Option<f32>,
        has_sinks: bool,
        is_causal: bool,
        sliding_window_size: Option<usize>,
    ) -> Result<Self, AttentionError> {
        let kernel = AttentionKernel::new(context, data_type)?;

        let mut single_pass_kernels = HashMap::new();

        let supported_head_dims = [64u32, 128u32, 256u32];
        for head_dim in supported_head_dims {
            for has_mask in [false, true] {
                let key = KernelKey {
                    head_dim,
                    has_mask,
                };
                let kernel = AttentionSinglePassMetalKernel::new(
                    context, data_type, head_dim, has_mask, has_mask, has_sinks, is_causal,
                )?;
                single_pass_kernels.insert(key, kernel);
            }
        }

        let update_kv_cache_kernel = AttentionUpdateKVCacheMetalKernel::new(context, data_type)?;

        Ok(Self {
            kernel,
            single_pass_kernels,
            update_kv_cache_kernel,
            layer_index,
            attention_scale,
            has_sinks,
            is_causal,
            sliding_window_size,
        })
    }
}

impl EncodableBlock<Metal> for Attention {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, parameters, &compute_encoder);
        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        let (suffix_length, num_heads, head_dim, num_groups, max_sequence_length) = {
            let qkv_binding = state.arrays(&[ArrayId::QKV]);
            let qkv_array = qkv_binding[0].borrow();
            let suffix_length = qkv_array.shape()[0];

            let queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
            let queries_array = queries_binding[0].borrow();
            let num_heads = queries_array.shape()[0];
            let head_dim = queries_array.shape()[2];

            let keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
            let keys_array = keys_binding[0].borrow();
            let num_groups = keys_array.shape()[0];

            let max_sequence_length = if let Some(_kv) = state.cache_layers() {
                let key_cache_binding = state.arrays(&[ArrayId::Keys(self.layer_index)]);
                let key_cache_array = key_cache_binding[0].borrow();
                key_cache_array.shape()[1]
            } else {
                // For classifiers without KV cache, max_sequence_length is just suffix_length
                suffix_length
            };

            (suffix_length, num_heads, head_dim, num_groups, max_sequence_length)
        };

        let (segment_prefix_length, window_length) = if let Some(cache_layers) = state.cache_layers() {
            let cache = cache_layers.borrow();
            let layer = cache.data[self.layer_index]
                .as_transformer()
                .expect("Attention kernel expects transformer layer state");
            (layer.projected_segment_prefix_length(parameters.projection_step.unwrap_or(0)), layer.window_length())
        } else {
            // For classifiers without KV cache: no prefix, use configured sliding window
            (0, self.sliding_window_size)
        };

        let use_mask = window_length.is_some();

        let sequence_length = segment_prefix_length + suffix_length;

        let gqa_factor = num_heads / num_groups;
        let scale = self.attention_scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

        let gemm_enabled = env_gemm_attention_enabled();
        let use_gemm = gemm_enabled && suffix_length > 8 && matches!(head_dim, 64 | 128 | 256);
        if !gemm_enabled {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(|| {
                eprintln!("[uzu] Gemm attention disabled via UZU_USE_GEMM_ATTENTION");
            });
        }
        let variant = if use_gemm {
            AttentionKernelVariant::SinglePass
        } else {
            self.kernel.choose_variant(sequence_length, head_dim, self.is_causal, use_mask)
        };

        let rotated_queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let qkv_binding = state.arrays(&[ArrayId::QKV]);
        let attention_bias_binding = state.hashmaps(&[HashMapId::AttentionBias]);
        let attention_output_binding = state.arrays(&[ArrayId::AttentionOutput]);

        let mask_kv_seq_stride = 1;
        let mask_q_seq_stride = sequence_length as i32;
        let mask_head_stride = 0;
        let sinks_binding = if self.has_sinks {
            Some(state.arrays(&[ArrayId::AttentionSinks(self.layer_index)]))
        } else {
            None
        };

        let rotated_keys_array = rotated_keys_binding[0].borrow_mut();
        let rotated_keys_buffer = rotated_keys_array.buffer();

        let qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buffer = qkv_array.buffer();

        // Get KV cache buffers only if KV cache exists (LLM mode)
        let has_kv_cache = state.cache_layers().is_some();
        let (key_cache_buffer, value_cache_buffer) = if has_kv_cache {
            let key_cache_binding = state.arrays(&[ArrayId::Keys(self.layer_index)]);
            let value_cache_binding = state.arrays(&[ArrayId::Values(self.layer_index)]);

            let key_cache_array = key_cache_binding[0].borrow_mut();
            let key_cache_buf = key_cache_array.buffer().clone();

            let value_cache_array = value_cache_binding[0].borrow_mut();
            let value_cache_buf = value_cache_array.buffer().clone();

            (key_cache_buf, value_cache_buf)
        } else {
            // For classifiers, we need a values buffer with [num_groups, suffix_length, head_dim] layout.
            // Use the KV cache update kernel to extract values from QKV into a dedicated extracted_values buffer.
            let extracted_values_binding = state.arrays(&[ArrayId::ExtractedValues]);
            let extracted_values_array = extracted_values_binding[0].borrow_mut();
            let extracted_values_buf = extracted_values_array.buffer().clone();

            // Reuse the KV cache update kernel to write values into extracted_values_buf.
            self.update_kv_cache_kernel.encode(
                rotated_keys_buffer,
                qkv_buffer,
                // keys already in desired layout; harmless overwrite
                rotated_keys_buffer,
                &extracted_values_buf,
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                0u32,
                max_sequence_length as u32,
                &compute_encoder,
            );

            (rotated_keys_buffer.clone(), extracted_values_buf)
        };

        let queries_array = rotated_queries_binding[0].borrow_mut();
        let queries_buffer = queries_array.buffer();

        let attention_output_array = attention_output_binding[0].borrow_mut();
        let attention_output_buffer = attention_output_array.buffer();

        let attention_bias_buffer = if use_mask {
            let attention_bias_map = attention_bias_binding[0].clone();
            Some(
                attention_bias_map
                    .get(&window_length)
                    .map(|array| {
                        let array_ref = array.borrow();
                        array_ref.buffer().clone()
                    })
                    .unwrap_or_else(|| {
                        panic!("Attention bias buffer not found for window length {:?}", window_length);
                    }),
            )
        } else {
            None
        };

        let partials_binding = state.arrays(&[ArrayId::AttentionPartials]);
        let sums_binding = state.arrays(&[ArrayId::AttentionSums]);
        let maxs_binding = state.arrays(&[ArrayId::AttentionMaxs]);

        let partials_array = partials_binding[0].borrow_mut();
        let partials_buffer = partials_array.buffer();

        let sums_array = sums_binding[0].borrow_mut();
        let sums_buffer = sums_array.buffer();

        let maxs_array = maxs_binding[0].borrow_mut();
        let maxs_buffer = maxs_array.buffer();

        let sinks_buffer = sinks_binding.as_ref().map(|binding| binding[0].borrow().buffer().clone());

        // Only update KV cache for LLM mode (not for classifiers)
        if has_kv_cache {
            self.update_kv_cache_kernel.encode(
                rotated_keys_buffer,
                qkv_buffer,
                &key_cache_buffer,
                &value_cache_buffer,
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                segment_prefix_length as u32,
                max_sequence_length as u32,
                &compute_encoder,
            );
        }

        let k_head_stride = (max_sequence_length * head_dim) as i32;
        let k_seq_stride = head_dim as i32;
        let v_head_stride = (max_sequence_length * head_dim) as i32;
        let v_seq_stride = head_dim as i32;

        match variant {
            AttentionKernelVariant::SinglePass => {
                if use_gemm {
                    let mtl = state.mtl_context();
                    if let Err(e) = self.kernel.encode_gemm(
                        mtl,
                        compute_encoder,
                        AttentionGemmArguments {
                            queries_buffer: &queries_buffer,
                            keys_buffer: &key_cache_buffer,
                            values_buffer: &value_cache_buffer,
                            output_buffer: &attention_output_buffer,
                            mask_buffer: attention_bias_buffer.as_deref(),
                            sinks_buffer: sinks_buffer.as_deref(),
                            num_heads,
                            num_groups,
                            suffix_length,
                            sequence_length,
                            segment_prefix_length,
                            max_sequence_length,
                            head_dim,
                            is_causal: self.is_causal,
                            scale,
                        },
                    ) {
                        eprintln!("Failed to encode gemm attention: {:?}", e);
                    }
                } else {
                    let key = KernelKey {
                        head_dim: head_dim as u32,
                        has_mask: use_mask,
                    };
                    if let Some(kernel) = self.single_pass_kernels.get(&key) {
                        let mask_buffer = attention_bias_buffer.as_ref().map(|buffer| buffer);
                        let sinks_buffer = sinks_buffer.as_ref().map(|buffer| buffer);
                        kernel.encode(
                            queries_buffer,
                            &key_cache_buffer,
                            &value_cache_buffer,
                            attention_output_buffer,
                            gqa_factor as u32,
                            sequence_length as u32,
                            k_head_stride as u32,
                            k_seq_stride as u32,
                            v_head_stride as u32,
                            v_seq_stride as u32,
                            scale,
                            mask_buffer,
                            Some(mask_kv_seq_stride as u32),
                            Some(mask_q_seq_stride as u32),
                            Some(mask_head_stride as u32),
                            sinks_buffer,
                            num_heads as u32,
                            suffix_length as u32,
                            &compute_encoder,
                        )
                    }
                }
            },
            AttentionKernelVariant::TwoPass => {
                if let Err(e) = self.kernel.encode_two_pass(
                    compute_encoder,
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
                        mask_buffer: attention_bias_buffer.as_deref(),
                        mask_kv_seq_stride,
                        mask_q_seq_stride,
                        mask_head_stride,
                        sinks_buffer: sinks_buffer.as_deref(),
                        num_heads,
                        suffix_length,
                        head_dim,
                        is_causal: self.is_causal,
                    },
                ) {
                    eprintln!("Failed to encode two-pass attention: {:?}", e);
                }
            },
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
struct KernelKey {
    pub head_dim: u32,
    pub has_mask: bool,
}
