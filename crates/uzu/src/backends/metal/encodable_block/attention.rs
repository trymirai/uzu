//! Attention kernel encodable.

use metal::CommandBufferRef;

use super::{EncodableBlock, EncodingParameters};
use crate::backends::metal::{
    KernelDataType, MTLContext,
    forward_pass::{ArrayId, ForwardPassState, HashMapId},
    kernel::attention::{
        AttentionError, AttentionKernel, AttentionKernelVariant,
        AttentionSinglePassArguments, AttentionTwoPassArguments,
        KVCacheUpdateArguments,
    },
};

pub struct Attention {
    kernel: AttentionKernel,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
    is_causal: bool,
    sliding_window_size: Option<usize>,
}

impl Attention {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        layer_index: usize,
        attention_scale: Option<f32>,
        has_sinks: bool,
        is_causal: bool,
        sliding_window_size: Option<usize>,
    ) -> Result<Self, AttentionError> {
        let kernel = AttentionKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            layer_index,
            attention_scale,
            has_sinks,
            is_causal,
            sliding_window_size,
        })
    }
}

impl EncodableBlock for Attention {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        let (
            suffix_length,
            num_heads,
            head_dim,
            num_groups,
            max_sequence_length,
        ) = {
            use crate::Array;
            let qkv_binding = state.arrays(&[ArrayId::QKV]);
            let qkv_array = qkv_binding[0].borrow();
            let suffix_length = Array::shape(&*qkv_array)[0];

            let queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
            let queries_array = queries_binding[0].borrow();
            let num_heads = Array::shape(&*queries_array)[0];
            let head_dim = Array::shape(&*queries_array)[2];

            let keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
            let keys_array = keys_binding[0].borrow();
            let num_groups = Array::shape(&*keys_array)[0];

            let max_sequence_length = if let Some(_kv) = state.cache_layers() {
                let key_cache_binding =
                    state.arrays(&[ArrayId::Keys(self.layer_index)]);
                let key_cache_array = key_cache_binding[0].borrow();
                Array::shape(&*key_cache_array)[1]
            } else {
                // For classifiers without KV cache, max_sequence_length is just suffix_length
                suffix_length
            };

            (
                suffix_length,
                num_heads,
                head_dim,
                num_groups,
                max_sequence_length,
            )
        };

        let (segment_prefix_length, window_length) =
            if let Some(cache_layers) = state.cache_layers() {
                let cache = cache_layers.borrow();
                let layer = cache.data[self.layer_index]
                    .as_transformer()
                    .expect("Attention kernel expects transformer layer state");
                (
                    layer.projected_segment_prefix_length(
                        parameters.projection_step.unwrap_or(0),
                    ),
                    layer.window_length(),
                )
            } else {
                // For classifiers without KV cache: no prefix, use configured sliding window
                (0, self.sliding_window_size)
            };

        let sequence_length = segment_prefix_length + suffix_length;

        let gqa_factor = num_heads / num_groups;
        let scale =
            self.attention_scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

        let variant = self.kernel.choose_variant(
            sequence_length,
            head_dim,
            self.is_causal,
        );

        let rotated_queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let qkv_binding = state.arrays(&[ArrayId::QKV]);
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

        // Prepare command encoder early (used for optional value extraction)
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Get KV cache buffers only if KV cache exists (LLM mode)
        let has_kv_cache = state.cache_layers().is_some();
        let (key_cache_buffer, value_cache_buffer) = if has_kv_cache {
            let key_cache_binding =
                state.arrays(&[ArrayId::Keys(self.layer_index)]);
            let value_cache_binding =
                state.arrays(&[ArrayId::Values(self.layer_index)]);

            let mut key_cache_array = key_cache_binding[0].borrow_mut();
            let key_cache_buf = unsafe { key_cache_array.mtl_buffer().clone() };

            let mut value_cache_array = value_cache_binding[0].borrow_mut();
            let value_cache_buf =
                unsafe { value_cache_array.mtl_buffer().clone() };

            (key_cache_buf, value_cache_buf)
        } else {
            // For classifiers, we need a values buffer with [num_groups, suffix_length, head_dim] layout.
            // Use the KV cache update kernel to extract values from QKV into a dedicated extracted_values buffer.
            let extracted_values_binding =
                state.arrays(&[ArrayId::ExtractedValues]);
            let mut extracted_values_array =
                extracted_values_binding[0].borrow_mut();
            let extracted_values_buf =
                unsafe { extracted_values_array.mtl_buffer().clone() };

            // Reuse the KV cache update kernel to write values into extracted_values_buf.
            if let Err(e) = self.kernel.encode_kv_cache_update(
                &compute_encoder,
                KVCacheUpdateArguments {
                    rotated_keys_buffer: &rotated_keys_buffer,
                    qkv_buffer: &qkv_buffer,
                    key_cache_buffer: &rotated_keys_buffer, // keys already in desired layout; harmless overwrite
                    value_cache_buffer: &extracted_values_buf,
                    num_groups,
                    num_heads,
                    head_dim,
                    suffix_length,
                    segment_prefix_length: 0,
                    max_sequence_length,
                },
            ) {
                eprintln!("Failed to prepare rotated values buffer: {:?}", e);
            }

            (rotated_keys_buffer.clone(), extracted_values_buf)
        };

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

        // Only update KV cache for LLM mode (not for classifiers)
        if has_kv_cache {
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
                        is_causal: self.is_causal,
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
                        is_causal: self.is_causal,
                    },
                ) {
                    eprintln!("Failed to encode two-pass attention: {:?}", e);
                }
            },
        }

        compute_encoder.end_encoding();
    }
}
