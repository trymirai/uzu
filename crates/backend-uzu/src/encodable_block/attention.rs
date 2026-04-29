//! Attention kernel encodable.

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use itertools::iproduct;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel,
            SigmoidGateKernel,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
        },
    },
    encodable_block::EncodingParameters,
    forward_pass::{
        kv_cache_layer::KVCacheLayerState,
        state::{ArrayId, ForwardPassState},
    },
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

pub struct Attention<B: Backend> {
    single_pass_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionSinglePassKernel>,
    two_pass_1_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionTwoPass1Kernel>,
    two_pass_2_kernels: HashMap<u32, <B::Kernels as Kernels>::AttentionTwoPass2Kernel>,
    update_kv_cache_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    update_kv_cache_inplace_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    gate_kernel: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
    gemm_block: AttentionGemmBlock<B>,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
    is_causal: bool,
    sliding_window_size: Option<usize>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        layer_index: usize,
        attention_scale: Option<f32>,
        has_sinks: bool,
        is_causal: bool,
        sliding_window_size: Option<usize>,
        has_gate: bool,
    ) -> Result<Self, B::Error> {
        let mut single_pass_kernels = HashMap::new();
        let mut two_pass_1_kernels = HashMap::new();
        let mut two_pass_2_kernels = HashMap::new();

        for (head_dim, is_trie, is_kv_cache_ring) in iproduct!([64u32, 128u32, 256u32], [false, true], [false, true]) {
            let key = KernelKey {
                head_dim,
                is_trie,
                is_kv_cache_ring,
            };

            let sp_kernel = <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                context,
                data_type,
                head_dim,
                has_sinks,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                sliding_window_size.is_some(),
            )?;
            single_pass_kernels.insert(key, sp_kernel);

            let tp1_kernel = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
                context,
                data_type,
                head_dim,
                has_sinks,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                sliding_window_size.is_some(),
            )?;
            two_pass_1_kernels.insert(key, tp1_kernel);

            let tp2_kernel = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, data_type, head_dim)?;
            two_pass_2_kernels.insert(head_dim, tp2_kernel);
        }

        let update_kv_cache_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, false)?;
        let update_kv_cache_inplace_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, true)?;
        let gate_kernel = if has_gate {
            Some(<B::Kernels as Kernels>::SigmoidGateKernel::new(context, data_type)?)
        } else {
            None
        };
        let gemm_block = AttentionGemmBlock::new(data_type);

        Ok(Self {
            single_pass_kernels,
            two_pass_1_kernels,
            two_pass_2_kernels,
            update_kv_cache_kernel,
            update_kv_cache_inplace_kernel,
            gate_kernel,
            gemm_block,
            layer_index,
            attention_scale,
            has_sinks,
            is_causal,
            sliding_window_size,
        })
    }

    fn select_variant(
        &self,
        gemm_enabled: bool,
        suffix_length: usize,
        head_dim: usize,
        sequence_length: usize,
        is_trie: bool,
        is_kv_cache_ring: bool,
    ) -> KernelVariant {
        let use_gemm = gemm_enabled && suffix_length > 8 && matches!(head_dim, 64 | 128 | 256);
        if use_gemm {
            return KernelVariant::Gemm;
        }

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };
        if sequence_length > 1024
            && self.two_pass_1_kernels.contains_key(&kernel_key)
            && self.two_pass_2_kernels.contains_key(&(head_dim as u32))
        {
            return KernelVariant::TwoPass;
        }

        KernelVariant::SinglePass
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let qkv_array = state.array(ArrayId::QKV);
        let queries_array = state.array(ArrayId::RotatedQueries);
        let rotated_keys_array = state.array(ArrayId::RotatedKeys);
        let attention_output_array = state.array(ArrayId::AttentionOutput);
        let partials_array = state.array(ArrayId::AttentionPartials);
        let sums_array = state.array(ArrayId::AttentionSums);
        let maxs_array = state.array(ArrayId::AttentionMaxs);

        let suffix_length = qkv_array.shape()[0];
        let num_heads = queries_array.shape()[0];
        let head_dim = queries_array.shape()[2];
        let num_groups = rotated_keys_array.shape()[0];
        let max_sequence_length = if state.cache_layers().is_some() {
            state.array(ArrayId::Keys(self.layer_index)).shape()[1]
        } else {
            // For classifiers without KV cache, max_sequence_length is just suffix_length
            suffix_length
        };

        let is_trie = state.token_subtrie_ranges.is_some();

        let (segment_prefix_length, ring_params) = if let Some(cache_layers) = state.cache_layers() {
            let projection_step = parameters.projection_step.unwrap_or(0);
            let cache = cache_layers.borrow();
            let layer = cache.data[self.layer_index]
                .as_transformer()
                .expect("Attention kernel expects transformer layer state");
            let ring_params = match layer.state {
                KVCacheLayerState::Windowed {
                    ring_offset,
                    ring_length,
                    window_length,
                } => {
                    let overflow = (ring_length + projection_step).saturating_sub(window_length);
                    Some(RingParams {
                        ring_offset: ((ring_offset + overflow) % window_length) as u32,
                        ring_length: (ring_length + projection_step).min(window_length) as u32,
                    })
                },
                _ => None,
            };
            (layer.projected_segment_prefix_length(projection_step), ring_params)
        } else {
            (0, None)
        };

        let is_kv_cache_ring = ring_params.is_some();

        let sequence_length = segment_prefix_length + suffix_length;

        let gqa_factor = num_heads / num_groups;
        let scale = self.attention_scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

        let gemm_enabled = env_gemm_attention_enabled();
        if !gemm_enabled {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(|| {
                eprintln!("[uzu] Gemm attention disabled via UZU_USE_GEMM_ATTENTION");
            });
        }
        let variant =
            self.select_variant(gemm_enabled, suffix_length, head_dim, sequence_length, is_trie, is_kv_cache_ring);

        let gate_array = self.gate_kernel.as_ref().map(|_| state.array(ArrayId::Gate));

        let sinks_array = self.has_sinks.then(|| state.array(ArrayId::AttentionSinks(self.layer_index)));

        let rotated_keys_buf_rc = rotated_keys_array.buffer();
        let mut rotated_keys_buf_borrow = rotated_keys_buf_rc.borrow_mut();

        let qkv_buf_rc = qkv_array.buffer();
        let qkv_buf_borrow = qkv_buf_rc.borrow();

        // Get KV cache buffers only if KV cache exists (LLM mode)
        let has_kv_cache = state.cache_layers().is_some();

        let key_cache_array = has_kv_cache.then(|| state.array(ArrayId::Keys(self.layer_index)));
        let key_cache_buf_rc = key_cache_array.as_ref().map(|a| a.buffer());
        let mut key_cache_buf_borrow = key_cache_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        let value_cache_array = has_kv_cache.then(|| state.array(ArrayId::Values(self.layer_index)));
        let value_cache_buf_rc = value_cache_array.as_ref().map(|a| a.buffer());
        let mut value_cache_buf_borrow = value_cache_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        let extracted_values_array = (!has_kv_cache).then(|| state.array(ArrayId::ExtractedValues));
        let extracted_values_buf_rc = extracted_values_array.as_ref().map(|a| a.buffer());
        let mut extracted_values_buf_borrow = extracted_values_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        // For classifiers (no KV cache): extract values from QKV into a dedicated extracted_values buffer.
        if !has_kv_cache {
            self.update_kv_cache_inplace_kernel.encode(
                None::<&B::Buffer>,
                qkv_buf_borrow.deref(),
                // keys already in desired layout; harmless overwrite
                rotated_keys_buf_borrow.deref_mut(),
                extracted_values_buf_borrow.as_mut().unwrap().deref_mut(),
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                0u32,
                max_sequence_length as u32,
                encoder,
            );
        }

        let queries_buf_rc = queries_array.buffer();
        let queries_buf_borrow = queries_buf_rc.borrow();

        let attention_output_buf_rc = attention_output_array.buffer();
        let mut attention_output_buf_borrow = attention_output_buf_rc.borrow_mut();

        let trie_buf_rc = state.token_subtrie_ranges.as_ref().map(|a| a.buffer());
        let trie_buf_borrow = trie_buf_rc.as_ref().map(|rc| rc.borrow());
        let trie_buffer: Option<&B::Buffer> = trie_buf_borrow.as_ref().map(|b| b.deref());

        let partials_buf_rc = partials_array.buffer();
        let mut partials_buf_borrow = partials_buf_rc.borrow_mut();

        let sums_buf_rc = sums_array.buffer();
        let mut sums_buf_borrow = sums_buf_rc.borrow_mut();

        let maxs_buf_rc = maxs_array.buffer();
        let mut maxs_buf_borrow = maxs_buf_rc.borrow_mut();

        let sinks_buf_rc = sinks_array.as_ref().map(|b| b.buffer());
        let sinks_buf_borrow = sinks_buf_rc.as_ref().map(|rc| rc.borrow());
        let sinks_buffer: Option<&B::Buffer> = sinks_buf_borrow.as_ref().map(|b| b.deref());

        // Only update KV cache for LLM mode (not for classifiers)
        if has_kv_cache {
            self.update_kv_cache_kernel.encode(
                Some(rotated_keys_buf_borrow.deref()),
                qkv_buf_borrow.deref(),
                key_cache_buf_borrow.as_mut().unwrap().deref_mut(),
                value_cache_buf_borrow.as_mut().unwrap().deref_mut(),
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                segment_prefix_length as u32,
                max_sequence_length as u32,
                encoder,
            );
        }

        let key_cache_buffer: &B::Buffer = if has_kv_cache {
            key_cache_buf_borrow.as_ref().unwrap().deref()
        } else {
            rotated_keys_buf_borrow.deref()
        };
        let value_cache_buffer: &B::Buffer = if has_kv_cache {
            value_cache_buf_borrow.as_ref().unwrap().deref()
        } else {
            extracted_values_buf_borrow.as_ref().unwrap().deref()
        };

        // KV cache layout: [max_sequence_length, num_groups, head_dim] (token-major)
        // For classifier mode (no KV cache) keys/values still come in group-major layout
        // [num_groups, suffix_length, head_dim].
        let (k_head_stride, k_seq_stride, v_head_stride, v_seq_stride) = if has_kv_cache {
            (head_dim as u32, (num_groups * head_dim) as u32, head_dim as u32, (num_groups * head_dim) as u32)
        } else {
            (
                (max_sequence_length * head_dim) as u32,
                head_dim as u32,
                (max_sequence_length * head_dim) as u32,
                head_dim as u32,
            )
        };

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };

        match variant {
            KernelVariant::Gemm => {
                let args = AttentionGemmArguments {
                    queries_buffer: queries_buf_borrow.deref(),
                    keys_buffer: key_cache_buffer,
                    values_buffer: value_cache_buffer,
                    output_buffer: attention_output_buf_borrow.deref_mut(),
                    trie_buffer,
                    sinks_buffer,
                    num_heads,
                    num_groups,
                    suffix_length,
                    sequence_length,
                    segment_prefix_length,
                    max_sequence_length,
                    ring_params,
                    head_dim,
                    is_causal: self.is_causal,
                    sliding_window_size: self.sliding_window_size,
                    scale,
                    k_head_stride: k_head_stride as u64,
                    k_seq_stride: k_seq_stride as u64,
                    v_head_stride: v_head_stride as u64,
                    v_seq_stride: v_seq_stride as u64,
                };
                self.gemm_block.encode(state.context(), encoder, args).expect("Failed to encode AttentionGemmBlock");
            },
            KernelVariant::SinglePass => {
                let kernel = match self.single_pass_kernels.get(&kernel_key) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionSinglePassKernel for key {:?}", kernel_key),
                };
                kernel.encode(
                    queries_buf_borrow.deref(),
                    key_cache_buffer,
                    value_cache_buffer,
                    attention_output_buf_borrow.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride,
                    k_seq_stride,
                    v_head_stride,
                    v_seq_stride,
                    ring_params,
                    scale,
                    trie_buffer,
                    self.sliding_window_size.map(|s| s as u32),
                    sinks_buffer,
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                )
            },
            KernelVariant::TwoPass => {
                let kernel_pass1 = match self.two_pass_1_kernels.get(&kernel_key) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionTwoPass1Kernel for key {:?}", kernel_key),
                };
                let kernel_pass2 = match self.two_pass_2_kernels.get(&(head_dim as u32)) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionTwoPass2Kernel for key {:?}", kernel_key),
                };
                kernel_pass1.encode(
                    queries_buf_borrow.deref(),
                    key_cache_buffer,
                    value_cache_buffer,
                    partials_buf_borrow.deref_mut(),
                    sums_buf_borrow.deref_mut(),
                    maxs_buf_borrow.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride,
                    k_seq_stride,
                    v_head_stride,
                    v_seq_stride,
                    ring_params,
                    scale,
                    num_heads as u32,
                    suffix_length as u32,
                    trie_buffer,
                    self.sliding_window_size.map(|s| s as u32),
                    sinks_buffer,
                    encoder,
                );
                kernel_pass2.encode(
                    partials_buf_borrow.deref(),
                    sums_buf_borrow.deref(),
                    maxs_buf_borrow.deref(),
                    attention_output_buf_borrow.deref_mut(),
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                );
            },
        }

        if let Some(gate_kernel) = &self.gate_kernel {
            let gate_array = gate_array.as_ref().unwrap();
            let gate_buf_rc = gate_array.buffer();
            let gate_buf_borrow = gate_buf_rc.borrow();
            let total_elements = (suffix_length * num_heads * head_dim) as u32;
            gate_kernel.encode(
                gate_buf_borrow.deref(),
                attention_output_buf_borrow.deref_mut(),
                total_elements,
                encoder,
            );
        }

        Ok(())
    }
}

enum KernelVariant {
    Gemm,
    SinglePass,
    TwoPass,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct KernelKey {
    pub head_dim: u32,
    pub is_trie: bool,
    pub is_kv_cache_ring: bool,
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/attention_test.rs"]
mod tests;
