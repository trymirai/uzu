//! Attention kernel encodable.

use std::{
    cmp::Ordering,
    collections::HashMap,
    ops::{Deref, DerefMut, Range},
};

use itertools::iproduct;

use crate::{
    ArrayElement, DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionShearSingleDecodeKernel, AttentionSinglePassKernel, AttentionTwoPass1Kernel,
            AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel, SigmoidGateKernel,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
        },
    },
    encodable_block::EncodingParameters,
    forward_pass::{
        kv_cache_layer::{KVCacheLayerState, KvCompressionMode},
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

#[cfg(feature = "tracing")]
fn env_sparse_value_debug_cpu_output_enabled() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var("UZU_DEBUG_SPARSE_VALUE_CPU_OUTPUT").ok();
        let Some(raw) = raw else {
            return false;
        };
        matches!(raw.trim().to_ascii_uppercase().as_str(), "1" | "YES" | "TRUE" | "ON")
    })
}

pub struct Attention<B: Backend> {
    shearkv_single_decode_kernels: HashMap<u32, <B::Kernels as Kernels>::AttentionShearSingleDecodeKernel>,
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
        let mut shearkv_single_decode_kernels = HashMap::new();
        let mut single_pass_kernels = HashMap::new();
        let mut two_pass_1_kernels = HashMap::new();
        let mut two_pass_2_kernels = HashMap::new();

        for (head_dim, is_trie, is_kv_cache_ring) in iproduct!([64u32, 128u32, 256u32], [false, true], [false, true]) {
            if !is_trie && !is_kv_cache_ring {
                let shear_kernel = <B::Kernels as Kernels>::AttentionShearSingleDecodeKernel::new(
                    context, data_type, head_dim, has_sinks,
                )?;
                shearkv_single_decode_kernels.insert(head_dim, shear_kernel);
            }
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
            shearkv_single_decode_kernels,
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

    #[allow(clippy::too_many_arguments)]
    fn try_encode_sparse_value_single_decode_output(
        &self,
        state: &ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        queries: &crate::array::Array<B>,
        rotated_keys: &crate::array::Array<B>,
        qkv: &crate::array::Array<B>,
        key_cache: &crate::array::Array<B>,
        value_cache: &crate::array::Array<B>,
        output: &mut crate::array::Array<B>,
        partials: &crate::array::Array<B>,
        sums: &crate::array::Array<B>,
        maxs: &crate::array::Array<B>,
        sinks: Option<&crate::array::Array<B>>,
        prefix_length: usize,
        max_sequence_length: usize,
        scale: f32,
    ) -> Result<bool, B::Error> {
        let Some(cache_layers) = state.cache_layers() else {
            return Ok(false);
        };
        let cache = cache_layers.borrow();
        let layer =
            cache.data[self.layer_index].as_transformer().expect("sparse value decode requires a transformer layer");
        let Some(sparse_value) = layer.sparse_value.as_ref() else {
            return Ok(false);
        };
        if !matches!(layer.state, KVCacheLayerState::Full { .. }) {
            return Ok(false);
        }

        let num_heads = queries.shape()[0];
        let num_groups = layer.shape[0];
        let head_dim = layer.shape[2];
        let page_size = sparse_value.config.page_size;
        let use_compressed_prefix_values =
            layer.values.is_none() && layer.supports_value_row_decoding_for_single_decode();
        let selection = select_sparse_value_prefix_selection(
            &sparse_value.config,
            sparse_value.shadow_keys.as_ref(),
            &flatten_array_as_f32(queries),
            num_heads,
            num_groups,
            max_sequence_length,
            prefix_length,
            head_dim,
            scale,
        );
        if !use_compressed_prefix_values && selection.selected_pages.len() == selection.recent_start / page_size {
            return Ok(false);
        }
        let compact_prefix_length =
            selection.selected_pages.len() * page_size + (prefix_length - selection.recent_start);
        let compact_sequence_length = compact_prefix_length + 1;
        let (compact_keys, mut compact_values) = state.sparse_value_buffers_for(key_cache, value_cache);
        compact_sparse_value_prefix(
            encoder,
            key_cache,
            &compact_keys,
            num_groups,
            max_sequence_length,
            compact_sequence_length,
            head_dim,
            selection.selected_pages.as_ref(),
            page_size,
            selection.recent_start,
            prefix_length,
        );
        overwrite_sparse_value_self_row(
            encoder,
            rotated_keys,
            &compact_keys,
            num_groups,
            compact_sequence_length,
            head_dim,
        );
        if use_compressed_prefix_values {
            let recent_values = layer
                .sparse_value_recent_values
                .as_ref()
                .expect("SparseValue compressed decode requires dense recent values")
                .borrow()
                .clone();
            compact_sparse_value_prefix_from_compressed_values(
                encoder,
                layer,
                sparse_value,
                &recent_values,
                qkv,
                num_heads,
                &mut compact_values,
                num_groups,
                max_sequence_length,
                compact_sequence_length,
                head_dim,
                selection.selected_pages.as_ref(),
                page_size,
                selection.recent_start,
                prefix_length,
            );
            #[cfg(feature = "tracing")]
            if env_sparse_value_debug_cpu_output_enabled() {
                let expected_output = sparse_value_expected_single_decode_output(
                    layer,
                    sparse_value,
                    queries,
                    rotated_keys,
                    qkv,
                    key_cache,
                    Some(&recent_values),
                    None,
                    num_heads,
                    num_groups,
                    max_sequence_length,
                    compact_sequence_length,
                    head_dim,
                    selection.selected_pages.as_ref(),
                    page_size,
                    selection.recent_start,
                    prefix_length,
                    scale,
                );
                write_array_from_f32(output, &expected_output);
                return Ok(true);
            }
            #[cfg(feature = "tracing")]
            {
                let expected_output = sparse_value_expected_single_decode_output(
                    layer,
                    sparse_value,
                    queries,
                    rotated_keys,
                    qkv,
                    key_cache,
                    Some(&recent_values),
                    None,
                    num_heads,
                    num_groups,
                    max_sequence_length,
                    compact_sequence_length,
                    head_dim,
                    selection.selected_pages.as_ref(),
                    page_size,
                    selection.recent_start,
                    prefix_length,
                    scale,
                );
                let traces = state.traces().borrow();
                let mut layer_trace = traces.layer_results[self.layer_index].borrow_mut();
                write_array_from_f32(&mut layer_trace.sparse_expected_attention, &expected_output);
                layer_trace.has_sparse_expected_attention = true;
            }
        } else {
            compact_sparse_value_prefix(
                encoder,
                value_cache,
                &compact_values,
                num_groups,
                max_sequence_length,
                compact_sequence_length,
                head_dim,
                selection.selected_pages.as_ref(),
                page_size,
                selection.recent_start,
                prefix_length,
            );
            #[cfg(feature = "tracing")]
            {
                let expected_output = sparse_value_expected_single_decode_output(
                    layer,
                    sparse_value,
                    queries,
                    rotated_keys,
                    qkv,
                    key_cache,
                    None,
                    Some(value_cache),
                    num_heads,
                    num_groups,
                    max_sequence_length,
                    compact_sequence_length,
                    head_dim,
                    selection.selected_pages.as_ref(),
                    page_size,
                    selection.recent_start,
                    prefix_length,
                    scale,
                );
                let traces = state.traces().borrow();
                let mut layer_trace = traces.layer_results[self.layer_index].borrow_mut();
                write_array_from_f32(&mut layer_trace.sparse_expected_attention, &expected_output);
                layer_trace.has_sparse_expected_attention = true;
            }
        }
        drop(cache);
        self.encode_single_decode_output(
            encoder,
            queries,
            &compact_keys,
            &compact_values,
            output,
            partials,
            sums,
            maxs,
            sinks,
            num_heads,
            num_groups,
            head_dim,
            compact_sequence_length,
            compact_sequence_length * head_dim,
            compact_sequence_length * head_dim,
            scale,
        );
        Ok(true)
    }

    fn try_encode_shearkv_single_decode_output(
        &self,
        state: &ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        queries: &crate::array::Array<B>,
        key_cache: &crate::array::Array<B>,
        value_cache: &crate::array::Array<B>,
        sinks: Option<&crate::array::Array<B>>,
        num_heads: usize,
        num_groups: usize,
        prefix_length: usize,
        max_sequence_length: usize,
        scale: f32,
        output: &mut crate::array::Array<B>,
    ) -> Result<bool, B::Error> {
        let Some(cache_layers) = state.cache_layers() else {
            return Ok(false);
        };
        let cache = cache_layers.borrow();
        let layer = cache.data[self.layer_index].as_transformer().expect("ShearKV decode requires a transformer layer");
        if layer.compression_mode != KvCompressionMode::ShearKv {
            return Ok(false);
        }
        if !matches!(layer.state, KVCacheLayerState::Full { .. }) {
            return Ok(false);
        }
        if layer.keys.is_none() || layer.values.is_some() {
            return Ok(false);
        }
        let Some(buffers) = layer.value_kernel_buffers_for_single_decode() else {
            return Ok(false);
        };
        let head_dim = layer.shape[2];
        let Some(kernel) = self.shearkv_single_decode_kernels.get(&(head_dim as u32)) else {
            return Ok(false);
        };

        let queries_buffer = queries.buffer();
        let queries_buffer = queries_buffer.borrow();
        let key_cache_buffer = key_cache.buffer();
        let key_cache_buffer = key_cache_buffer.borrow();
        let value_cache_buffer = value_cache.buffer();
        let value_cache_buffer = value_cache_buffer.borrow();
        let output_buffer = output.buffer();
        let mut output_buffer = output_buffer.borrow_mut();
        let sinks_buffer = sinks.map(|array| array.buffer());
        let sinks_buffer = sinks_buffer.as_ref().map(|buffer| buffer.borrow());
        kernel.encode(
            queries_buffer.deref(),
            key_cache_buffer.deref(),
            buffers.codes,
            buffers.scales,
            buffers.biases,
            value_cache_buffer.deref(),
            output_buffer.deref_mut(),
            (num_heads / num_groups) as u32,
            prefix_length as u32,
            max_sequence_length as u32,
            (max_sequence_length * head_dim) as u32,
            head_dim as u32,
            buffers.row_bytes as u32,
            scale,
            sinks_buffer.as_ref().map(|buffer| buffer.deref()),
            num_heads as u32,
            buffers.bits as u32,
            encoder,
        );

        Ok(true)
    }

    fn encode_single_decode_output(
        &self,
        encoder: &mut Encoder<B>,
        queries: &crate::array::Array<B>,
        keys: &crate::array::Array<B>,
        values: &crate::array::Array<B>,
        output: &mut crate::array::Array<B>,
        partials: &crate::array::Array<B>,
        sums: &crate::array::Array<B>,
        maxs: &crate::array::Array<B>,
        sinks: Option<&crate::array::Array<B>>,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        sequence_length: usize,
        key_head_stride: usize,
        value_head_stride: usize,
        scale: f32,
    ) {
        let queries_buffer = queries.buffer();
        let queries_buffer = queries_buffer.borrow();
        let keys_buffer = keys.buffer();
        let keys_buffer = keys_buffer.borrow();
        let values_buffer = values.buffer();
        let values_buffer = values_buffer.borrow();
        let partials_buffer = partials.buffer();
        let mut partials_buffer = partials_buffer.borrow_mut();
        let sums_buffer = sums.buffer();
        let mut sums_buffer = sums_buffer.borrow_mut();
        let maxs_buffer = maxs.buffer();
        let mut maxs_buffer = maxs_buffer.borrow_mut();
        let output_buffer = output.buffer();
        let mut output_buffer = output_buffer.borrow_mut();
        let sinks_buffer = sinks.map(|array| array.buffer());
        let sinks_buffer = sinks_buffer.as_ref().map(|buffer| buffer.borrow());

        let gqa_factor = num_heads / num_groups;
        let variant = self.select_variant(env_gemm_attention_enabled(), 1, head_dim, sequence_length, false, false);
        let k_head_stride = key_head_stride as u32;
        let v_head_stride = value_head_stride as u32;
        let seq_stride = head_dim as u32;
        match variant {
            KernelVariant::Gemm => unreachable!("single-token sparse value decode never uses gemm"),
            KernelVariant::SinglePass => {
                let kernel = self
                    .single_pass_kernels
                    .get(&KernelKey {
                        head_dim: head_dim as u32,
                        is_trie: false,
                        is_kv_cache_ring: false,
                    })
                    .expect("SparseValue single-pass attention kernel must exist");
                kernel.encode(
                    queries_buffer.deref(),
                    keys_buffer.deref(),
                    values_buffer.deref(),
                    output_buffer.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride,
                    seq_stride,
                    v_head_stride,
                    seq_stride,
                    None,
                    scale,
                    None::<&B::Buffer>,
                    None,
                    sinks.zip(sinks_buffer.as_ref()).map(|(_, buffer)| buffer.deref()),
                    num_heads as u32,
                    1,
                    encoder,
                );
            },
            KernelVariant::TwoPass => {
                let kernel = self
                    .two_pass_1_kernels
                    .get(&KernelKey {
                        head_dim: head_dim as u32,
                        is_trie: false,
                        is_kv_cache_ring: false,
                    })
                    .expect("SparseValue two-pass attention kernel must exist");
                kernel.encode(
                    queries_buffer.deref(),
                    keys_buffer.deref(),
                    values_buffer.deref(),
                    partials_buffer.deref_mut(),
                    sums_buffer.deref_mut(),
                    maxs_buffer.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride,
                    seq_stride,
                    v_head_stride,
                    seq_stride,
                    None,
                    scale,
                    num_heads as u32,
                    1,
                    None::<&B::Buffer>,
                    None,
                    sinks.zip(sinks_buffer.as_ref()).map(|(_, buffer)| buffer.deref()),
                    encoder,
                );
                let kernel = self
                    .two_pass_2_kernels
                    .get(&(head_dim as u32))
                    .expect("SparseValue two-pass reduction kernel must exist");
                kernel.encode(
                    partials_buffer.deref(),
                    sums_buffer.deref(),
                    maxs_buffer.deref(),
                    output_buffer.deref_mut(),
                    num_heads as u32,
                    1,
                    encoder,
                );
            },
        }
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
        let mut attention_output_array = state.array(ArrayId::AttentionOutput);
        let partials_array = state.array(ArrayId::AttentionPartials);
        let sums_array = state.array(ArrayId::AttentionSums);
        let maxs_array = state.array(ArrayId::AttentionMaxs);

        let suffix_length = qkv_array.shape()[0];
        let active_row_count = state.active_row_count();
        let num_heads = queries_array.shape()[0];
        let head_dim = queries_array.shape()[2];
        let num_groups = rotated_keys_array.shape()[0];

        let max_sequence_length = if state.cache_layers().is_some() {
            state.array(ArrayId::Keys(self.layer_index)).shape()[1]
        } else {
            // For classifiers without KV cache, max_sequence_length is just suffix_length
            suffix_length
        };
        let projection_step = parameters.projection_step.unwrap_or(0);

        let is_trie = state.token_subtrie_ranges.is_some();

        let (segment_prefix_length, ring_params, has_sparse_value) = if let Some(cache_layers) = state.cache_layers() {
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
            (layer.projected_segment_prefix_length(projection_step), ring_params, layer.sparse_value.is_some())
        } else {
            (0, None, false)
        };
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().borrow();
            let mut layer_trace = traces.layer_results[self.layer_index].borrow_mut();
            layer_trace.sparse_value_single_decode_has_kv_cache = state.cache_layers().is_some();
            layer_trace.sparse_value_single_decode_has_sparse_value = has_sparse_value;
            layer_trace.sparse_value_single_decode_suffix_length = suffix_length;
            layer_trace.sparse_value_single_decode_projection_step = projection_step;
            layer_trace.sparse_value_single_decode_is_trie = is_trie;
            layer_trace.sparse_value_single_decode_is_kv_cache_ring = ring_params.is_some();
        }
        #[cfg(not(feature = "tracing"))]
        let _ = has_sparse_value;

        if let Some(cache_layers) = state.cache_layers() {
            let mut cache = cache_layers.borrow_mut();
            let layer = cache.data[self.layer_index]
                .as_transformer_mut()
                .expect("Attention kernel expects transformer layer state");
            layer.stage_sparse_value_suffix_rows(&rotated_keys_array, &qkv_array, num_heads);
            if segment_prefix_length == 0 || suffix_length > 1 {
                layer.update_triattention_query_stats(&qkv_array, active_row_count);
            }
        }

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

        drop(value_cache_buf_borrow);
        drop(key_cache_buf_borrow);
        drop(rotated_keys_buf_borrow);
        drop(qkv_buf_borrow);
        drop(extracted_values_buf_borrow);

        if has_kv_cache {
            let mut cache = state.cache_layers().expect("KV cache layers must exist").borrow_mut();
            let layer = cache.data[self.layer_index]
                .as_transformer_mut()
                .expect("Attention kernel expects transformer layer state");
            layer.encode_sparse_value_pending_values_from_source(
                value_cache_array.as_ref().expect("KV cache values must exist"),
                segment_prefix_length,
                suffix_length,
                encoder,
            );
        }

        let used_custom_single_decode = if has_kv_cache
            && suffix_length == 1
            && projection_step == 0
            && (!is_trie || active_row_count == 1)
            && !is_kv_cache_ring
        {
            #[cfg(feature = "tracing")]
            {
                let traces = state.traces().borrow();
                let mut layer_trace = traces.layer_results[self.layer_index].borrow_mut();
                layer_trace.attempted_sparse_value_single_decode = true;
            }
            let used_sparse_value_single_decode = self.try_encode_sparse_value_single_decode_output(
                state,
                encoder,
                &queries_array,
                &rotated_keys_array,
                &qkv_array,
                key_cache_array.as_ref().expect("KV cache keys must exist"),
                value_cache_array.as_ref().expect("KV cache values must exist"),
                &mut attention_output_array,
                &partials_array,
                &sums_array,
                &maxs_array,
                sinks_array.as_ref(),
                segment_prefix_length,
                max_sequence_length,
                scale,
            )?;
            #[cfg(feature = "tracing")]
            if used_sparse_value_single_decode {
                let traces = state.traces().borrow();
                let mut layer_trace = traces.layer_results[self.layer_index].borrow_mut();
                layer_trace.used_sparse_value_single_decode = true;
            }
            let used_shearkv_single_decode = used_sparse_value_single_decode
                || self.try_encode_shearkv_single_decode_output(
                    state,
                    encoder,
                    &queries_array,
                    key_cache_array.as_ref().expect("KV cache keys must exist"),
                    value_cache_array.as_ref().expect("KV cache values must exist"),
                    sinks_array.as_ref(),
                    num_heads,
                    num_groups,
                    segment_prefix_length,
                    max_sequence_length,
                    scale,
                    &mut attention_output_array,
                )?;
            used_shearkv_single_decode
                || try_fill_compressed_single_decode_output(
                    state,
                    self.layer_index,
                    &queries_array,
                    &rotated_keys_array,
                    &qkv_array,
                    value_cache_array.as_ref().expect("KV cache values must exist"),
                    sinks_array.as_ref(),
                    num_heads,
                    segment_prefix_length,
                    scale,
                    &mut attention_output_array,
                )
        } else {
            false
        };

        if !used_custom_single_decode {
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

            let rotated_keys_buf_borrow = rotated_keys_buf_rc.borrow();
            let key_cache_buf_rc = key_cache_array.as_ref().map(|a| a.buffer());
            let key_cache_buf_borrow = key_cache_buf_rc.as_ref().map(|rc| rc.borrow());
            let value_cache_buf_rc = value_cache_array.as_ref().map(|a| a.buffer());
            let value_cache_buf_borrow = value_cache_buf_rc.as_ref().map(|rc| rc.borrow());
            let extracted_values_buf_rc = extracted_values_array.as_ref().map(|a| a.buffer());
            let extracted_values_buf_borrow = extracted_values_buf_rc.as_ref().map(|rc| rc.borrow());

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

            let k_head_stride = (max_sequence_length * head_dim) as i32;
            let k_seq_stride = head_dim as i32;
            let v_head_stride = (max_sequence_length * head_dim) as i32;
            let v_seq_stride = head_dim as i32;

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
                    };
                    self.gemm_block
                        .encode(state.context(), encoder, args)
                        .expect("Failed to encode AttentionGemmBlock");
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
                        k_head_stride as u32,
                        k_seq_stride as u32,
                        v_head_stride as u32,
                        v_seq_stride as u32,
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
                        k_head_stride as u32,
                        k_seq_stride as u32,
                        v_head_stride as u32,
                        v_seq_stride as u32,
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
        }

        if let Some(gate_kernel) = &self.gate_kernel {
            let gate_array = gate_array.as_ref().unwrap();
            let gate_buf_rc = gate_array.buffer();
            let gate_buf_borrow = gate_buf_rc.borrow();
            let attention_output_buf_rc = attention_output_array.buffer();
            let mut attention_output_buf_borrow = attention_output_buf_rc.borrow_mut();
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

struct SparseValueSelection {
    selected_pages: Box<[i32]>,
    recent_start: usize,
}

fn select_sparse_value_prefix_selection(
    config: &crate::forward_pass::kv_cache_layer::SparseValueConfig,
    shadow_keys: &[f32],
    queries: &[f32],
    num_heads: usize,
    num_groups: usize,
    sequence_length: usize,
    prefix_length: usize,
    head_dim: usize,
    scale: f32,
) -> SparseValueSelection {
    if prefix_length == 0 {
        return SparseValueSelection {
            selected_pages: Box::new([]),
            recent_start: 0,
        };
    }

    let archive_page_count = prefix_length.saturating_sub(config.recent_window) / config.page_size;
    let archive_token_count = archive_page_count * config.page_size;
    if archive_page_count == 0 {
        return SparseValueSelection {
            selected_pages: Box::new([]),
            recent_start: 0,
        };
    }

    let gqa_factor = num_heads / num_groups;
    let mut archive_page_masses = vec![0.0; archive_page_count];
    let mut scores = vec![0.0; archive_token_count];
    for group_index in 0..num_groups {
        let head_start = group_index * gqa_factor;
        for group_head_index in 0..gqa_factor {
            let head_index = head_start + group_head_index;
            let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for token_index in 0..archive_token_count {
                let row_start = (group_index * sequence_length + token_index) * head_dim;
                let key_row = &shadow_keys[row_start..row_start + head_dim];
                let score = scale
                    * query
                        .iter()
                        .zip(key_row.iter())
                        .map(|(&query_value, &key_value)| query_value * key_value)
                        .sum::<f32>();
                scores[token_index] = score;
                max_score = max_score.max(score);
            }

            let mut archive_exp = 0.0;
            let mut page_exps = vec![0.0; archive_page_count];
            for page_index in 0..archive_page_count {
                let page_start = page_index * config.page_size;
                let page_end = page_start + config.page_size;
                let page_exp = scores[page_start..page_end].iter().map(|score| (*score - max_score).exp()).sum::<f32>();
                page_exps[page_index] = page_exp;
                archive_exp += page_exp;
            }
            if archive_exp == 0.0 {
                continue;
            }
            for (mass, page_exp) in archive_page_masses.iter_mut().zip(page_exps.iter()) {
                *mass += *page_exp / archive_exp;
            }
        }
    }

    let mut selected_pages = vec![false; archive_page_count];
    let mut page_indices = Vec::with_capacity(archive_page_count);
    mark_sparse_value_archive_pages(&archive_page_masses, config.keep_mass, &mut selected_pages, &mut page_indices);
    SparseValueSelection {
        selected_pages: selected_pages
            .into_iter()
            .enumerate()
            .filter_map(|(page_index, selected)| selected.then_some(page_index as i32))
            .collect(),
        recent_start: archive_token_count,
    }
}

fn compact_sparse_value_prefix<B: Backend>(
    encoder: &mut Encoder<B>,
    source: &crate::array::Array<B>,
    destination: &crate::array::Array<B>,
    num_groups: usize,
    source_sequence_length: usize,
    destination_sequence_length: usize,
    head_dim: usize,
    selected_pages: &[i32],
    page_size: usize,
    recent_start: usize,
    prefix_length: usize,
) {
    let row_bytes = head_dim * source.data_type().size_in_bytes();
    let source_head_stride = source_sequence_length * row_bytes;
    let destination_head_stride = destination_sequence_length * row_bytes;
    let source_buffer = source.buffer();
    let source_buffer = source_buffer.borrow();
    let destination_buffer = destination.buffer();
    let mut destination_buffer = destination_buffer.borrow_mut();
    let runs = sparse_value_token_runs(selected_pages, page_size, recent_start, prefix_length);

    for group_index in 0..num_groups {
        let source_head_base = source.offset() + group_index * source_head_stride;
        let destination_head_base = destination.offset() + group_index * destination_head_stride;
        let mut destination_token = 0usize;
        for run in runs.iter() {
            let source_start = source_head_base + run.start * row_bytes;
            let destination_start = destination_head_base + destination_token * row_bytes;
            let byte_len = (run.end - run.start) * row_bytes;
            encoder.encode_copy(
                source_buffer.deref(),
                source_start..source_start + byte_len,
                destination_buffer.deref_mut(),
                destination_start..destination_start + byte_len,
            );
            destination_token += run.end - run.start;
        }
        let source_start = source_head_base + prefix_length * row_bytes;
        let destination_start = destination_head_base + (destination_sequence_length - 1) * row_bytes;
        encoder.encode_copy(
            source_buffer.deref(),
            source_start..source_start + row_bytes,
            destination_buffer.deref_mut(),
            destination_start..destination_start + row_bytes,
        );
    }
}

fn overwrite_sparse_value_self_row<B: Backend>(
    encoder: &mut Encoder<B>,
    source: &crate::array::Array<B>,
    destination: &crate::array::Array<B>,
    num_groups: usize,
    destination_sequence_length: usize,
    head_dim: usize,
) {
    let row_bytes = head_dim * source.data_type().size_in_bytes();
    let source_head_stride = source.shape()[1] * row_bytes;
    let destination_head_stride = destination_sequence_length * row_bytes;
    let source_buffer = source.buffer();
    let source_buffer = source_buffer.borrow();
    let destination_buffer = destination.buffer();
    let mut destination_buffer = destination_buffer.borrow_mut();

    for group_index in 0..num_groups {
        let source_start = source.offset() + group_index * source_head_stride;
        let destination_start = destination.offset()
            + (group_index * destination_head_stride + (destination_sequence_length - 1) * row_bytes);
        encoder.encode_copy(
            source_buffer.deref(),
            source_start..source_start + row_bytes,
            destination_buffer.deref_mut(),
            destination_start..destination_start + row_bytes,
        );
    }
}

fn compact_sparse_value_prefix_from_compressed_values<B: Backend>(
    encoder: &mut Encoder<B>,
    layer: &crate::forward_pass::kv_cache_layer::KVCacheLayer<B>,
    sparse_value: &crate::forward_pass::kv_cache_layer::SparseValueState,
    recent_values: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    num_heads: usize,
    destination: &mut crate::array::Array<B>,
    num_groups: usize,
    source_sequence_length: usize,
    destination_sequence_length: usize,
    head_dim: usize,
    selected_pages: &[i32],
    page_size: usize,
    recent_start: usize,
    prefix_length: usize,
) {
    match destination.data_type() {
        DataType::BF16 => compact_sparse_value_prefix_from_compressed_values_typed::<B, half::bf16>(
            encoder,
            layer,
            sparse_value,
            recent_values,
            qkv,
            num_heads,
            destination,
            num_groups,
            source_sequence_length,
            destination_sequence_length,
            head_dim,
            selected_pages,
            page_size,
            recent_start,
            prefix_length,
            half::bf16::from_f32,
        ),
        DataType::F16 => compact_sparse_value_prefix_from_compressed_values_typed::<B, half::f16>(
            encoder,
            layer,
            sparse_value,
            recent_values,
            qkv,
            num_heads,
            destination,
            num_groups,
            source_sequence_length,
            destination_sequence_length,
            head_dim,
            selected_pages,
            page_size,
            recent_start,
            prefix_length,
            half::f16::from_f32,
        ),
        DataType::F32 => compact_sparse_value_prefix_from_compressed_values_typed::<B, f32>(
            encoder,
            layer,
            sparse_value,
            recent_values,
            qkv,
            num_heads,
            destination,
            num_groups,
            source_sequence_length,
            destination_sequence_length,
            head_dim,
            selected_pages,
            page_size,
            recent_start,
            prefix_length,
            |value| value,
        ),
        dtype => panic!("SparseValue does not support compressed prefix dtype {dtype:?}"),
    }
}

fn sparse_value_token_runs(
    selected_pages: &[i32],
    page_size: usize,
    recent_start: usize,
    prefix_length: usize,
) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::with_capacity(selected_pages.len() + 1);
    for &page_index in selected_pages {
        let start = page_index as usize * page_size;
        let end = start + page_size;
        if let Some(last) = runs.last_mut()
            && last.end == start
        {
            last.end = end;
            continue;
        }
        runs.push(start..end);
    }
    if recent_start < prefix_length {
        if let Some(last) = runs.last_mut()
            && last.end == recent_start
        {
            last.end = prefix_length;
            return runs;
        }
        runs.push(recent_start..prefix_length);
    }
    runs
}

fn compact_sparse_value_prefix_from_compressed_values_typed<B: Backend, T: ArrayElement + Copy>(
    encoder: &mut Encoder<B>,
    layer: &crate::forward_pass::kv_cache_layer::KVCacheLayer<B>,
    sparse_value: &crate::forward_pass::kv_cache_layer::SparseValueState,
    recent_values: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    num_heads: usize,
    destination: &mut crate::array::Array<B>,
    num_groups: usize,
    source_sequence_length: usize,
    destination_sequence_length: usize,
    head_dim: usize,
    selected_pages: &[i32],
    page_size: usize,
    recent_start: usize,
    prefix_length: usize,
    convert: impl Fn(f32) -> T,
) {
    let archive_runs = sparse_value_token_runs(selected_pages, page_size, prefix_length, prefix_length);
    let max_run_tokens = archive_runs.iter().map(|run| run.end - run.start).max().unwrap_or(0);
    let mut decoded_values = vec![0.0f32; max_run_tokens.max(1) * head_dim];
    {
        let destination_rows = destination.as_slice_mut::<T>();
        for group_index in 0..num_groups {
            let mut destination_token = 0usize;
            for run in archive_runs.iter() {
                let row_count = run.end - run.start;
                let decoded_len = row_count * head_dim;
                layer.decode_value_rows_for_single_decode(
                    group_index * source_sequence_length + run.start,
                    row_count,
                    &mut decoded_values[..decoded_len],
                );
                let destination_start = (group_index * destination_sequence_length + destination_token) * head_dim;
                for (dst, &src) in destination_rows[destination_start..destination_start + decoded_len]
                    .iter_mut()
                    .zip(decoded_values[..decoded_len].iter())
                {
                    *dst = convert(src);
                }
                destination_token += row_count;
            }
        }
    }

    let row_bytes = head_dim * destination.data_type().size_in_bytes();
    let recent_head_stride = recent_values.shape()[1] * row_bytes;
    let destination_head_stride = destination_sequence_length * row_bytes;
    let recent_buffer = recent_values.buffer();
    let recent_buffer = recent_buffer.borrow();
    let destination_buffer = destination.buffer();
    let mut destination_buffer = destination_buffer.borrow_mut();
    let value_base = (num_heads + num_groups) * head_dim;

    for group_index in 0..num_groups {
        let archive_len = archive_runs.iter().map(|run| run.end - run.start).sum::<usize>();
        if recent_start < prefix_length {
            let recent_tokens = prefix_length - recent_start;
            assert!(
                recent_tokens <= sparse_value.hot_value_capacity,
                "SparseValue recent window exceeds the dense hot-value capacity"
            );
            let destination_head_base = destination.offset() + group_index * destination_head_stride;
            let recent_head_base = recent_values.offset() + group_index * recent_head_stride;
            let mut copied_tokens = 0usize;
            while copied_tokens < recent_tokens {
                let source_token = (recent_start + copied_tokens) % sparse_value.hot_value_capacity;
                let run_tokens = (sparse_value.hot_value_capacity - source_token).min(recent_tokens - copied_tokens);
                let source_start = recent_head_base + source_token * row_bytes;
                let destination_start = destination_head_base + (archive_len + copied_tokens) * row_bytes;
                let byte_len = run_tokens * row_bytes;
                encoder.encode_copy(
                    recent_buffer.deref(),
                    source_start..source_start + byte_len,
                    destination_buffer.deref_mut(),
                    destination_start..destination_start + byte_len,
                );
                copied_tokens += run_tokens;
            }
        }
    }

    let qkv_buffer = qkv.buffer();
    let qkv_buffer = qkv_buffer.borrow();
    for group_index in 0..num_groups {
        let source_start =
            qkv.offset() + (value_base + group_index * head_dim) * destination.data_type().size_in_bytes();
        let destination_start = destination.offset()
            + (group_index * destination_sequence_length + destination_sequence_length - 1) * row_bytes;
        encoder.encode_copy(
            qkv_buffer.deref(),
            source_start..source_start + row_bytes,
            destination_buffer.deref_mut(),
            destination_start..destination_start + row_bytes,
        );
    }
}

fn mark_sparse_value_archive_pages(
    page_exps: &[f32],
    keep_mass: f32,
    selected_pages: &mut [bool],
    page_indices: &mut Vec<usize>,
) {
    if keep_mass <= 0.0 {
        return;
    }
    let archive_exp = page_exps.iter().sum::<f32>();
    if archive_exp == 0.0 {
        return;
    }

    let required_exp = archive_exp * keep_mass;
    page_indices.clear();
    page_indices.extend(0..page_exps.len());
    page_indices.sort_unstable_by(|&left, &right| {
        page_exps[right].partial_cmp(&page_exps[left]).unwrap_or(Ordering::Equal).then_with(|| left.cmp(&right))
    });

    let mut kept_exp = 0.0;
    for page_index in page_indices.iter().copied() {
        selected_pages[page_index] = true;
        kept_exp += page_exps[page_index];
        if kept_exp >= required_exp {
            break;
        }
    }
}

fn try_fill_compressed_single_decode_output<B: Backend>(
    state: &ForwardPassState<B>,
    layer_index: usize,
    queries: &crate::array::Array<B>,
    rotated_keys: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    values: &crate::array::Array<B>,
    sinks: Option<&crate::array::Array<B>>,
    num_heads: usize,
    prefix_length: usize,
    scale: f32,
    output: &mut crate::array::Array<B>,
) -> bool {
    let Some(cache_layers) = state.cache_layers() else {
        return false;
    };
    let cache = cache_layers.borrow();
    let layer =
        cache.data[layer_index].as_transformer().expect("compressed single decode requires a transformer layer");
    if layer.sparse_value.is_some() {
        panic!(
            "SparseValue fell back to generic compressed decode on layer {layer_index}; \
             window_length={:?}",
            layer.window_length(),
        );
    }
    if !matches!(layer.state, KVCacheLayerState::Full { .. }) {
        return false;
    }
    if !layer.supports_compressed_prefix_attention_scores_for_single_decode() {
        return false;
    }

    let head_dim = layer.shape[2];
    let num_groups = layer.shape[0];
    let sequence_length = layer.shape[1];
    let queries = flatten_array_as_f32(queries);
    let mut prefix_scores = vec![0.0f32; num_heads * prefix_length];
    if !layer.fill_compressed_prefix_attention_scores_for_single_decode(
        &queries,
        num_heads,
        prefix_length,
        &mut prefix_scores,
    ) {
        return false;
    }

    let sinks = sinks.map(|array| array.as_slice::<f32>());
    let mut output_values = vec![0.0f32; num_heads * head_dim];
    let use_compressed_prefix_values = layer.values.is_none() && layer.supports_value_row_decoding_for_single_decode();
    assert_eq!(rotated_keys.data_type(), qkv.data_type(), "compressed single decode self-row dtype mismatch");
    assert_eq!(qkv.data_type(), values.data_type(), "compressed single decode value dtype mismatch");
    match qkv.data_type() {
        DataType::BF16 => {
            if use_compressed_prefix_values {
                fill_compressed_single_decode_output_with_compressed_values_typed::<B, half::bf16>(
                    layer,
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            } else {
                fill_compressed_single_decode_output_typed::<B, half::bf16>(
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            }
        },
        DataType::F16 => {
            if use_compressed_prefix_values {
                fill_compressed_single_decode_output_with_compressed_values_typed::<B, half::f16>(
                    layer,
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            } else {
                fill_compressed_single_decode_output_typed::<B, half::f16>(
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            }
        },
        DataType::F32 => {
            if use_compressed_prefix_values {
                fill_compressed_single_decode_output_with_compressed_values_typed::<B, f32>(
                    layer,
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            } else {
                fill_compressed_single_decode_output_typed::<B, f32>(
                    &queries,
                    &prefix_scores,
                    rotated_keys,
                    qkv,
                    values,
                    sinks,
                    num_heads,
                    num_groups,
                    sequence_length,
                    prefix_length,
                    scale,
                    &mut output_values,
                );
            }
        },
        dtype => panic!("compressed single decode does not support dtype {dtype:?}"),
    }
    write_array_from_f32(output, &output_values);
    true
}

fn fill_compressed_single_decode_output_typed<B: Backend, T: ArrayElement + Copy>(
    queries: &[f32],
    prefix_scores: &[f32],
    rotated_keys: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    values: &crate::array::Array<B>,
    sinks: Option<&[f32]>,
    num_heads: usize,
    num_groups: usize,
    sequence_length: usize,
    prefix_length: usize,
    scale: f32,
    output: &mut [f32],
) where
    f32: From<T>,
{
    let head_dim = rotated_keys.shape()[2];
    let gqa_factor = num_heads / num_groups;
    let rotated_key_rows = rotated_keys.as_slice::<T>();
    let qkv_rows = qkv.as_slice::<T>();
    let value_rows = values.as_slice::<T>();
    let value_base = (num_heads + num_groups) * head_dim;

    for head_index in 0..num_heads {
        let group_index = head_index / gqa_factor;
        let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
        let output_row = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
        output_row.fill(0.0);

        let mut max_score = sinks.map(|sinks| sinks[head_index]).unwrap_or(f32::NEG_INFINITY);
        let mut sum_exp = if sinks.is_some() {
            1.0
        } else {
            0.0
        };

        for token_index in 0..prefix_length {
            let row_index = group_index * sequence_length + token_index;
            let score = prefix_scores[head_index * prefix_length + token_index];
            accumulate_attention_row(
                output_row,
                &value_rows[row_index * head_dim..(row_index + 1) * head_dim],
                score,
                &mut max_score,
                &mut sum_exp,
            );
        }

        let self_key_row = &rotated_key_rows[group_index * head_dim..(group_index + 1) * head_dim];
        let self_value_row = &qkv_rows[value_base + group_index * head_dim..value_base + (group_index + 1) * head_dim];
        let self_score = scale
            * query
                .iter()
                .zip(self_key_row.iter())
                .map(|(&query_value, &key_value)| query_value * f32::from(key_value))
                .sum::<f32>();
        accumulate_attention_row(output_row, self_value_row, self_score, &mut max_score, &mut sum_exp);

        let inv_sum = 1.0 / sum_exp;
        for value in output_row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

fn fill_compressed_single_decode_output_with_compressed_values_typed<B: Backend, T: ArrayElement + Copy>(
    layer: &crate::forward_pass::kv_cache_layer::KVCacheLayer<B>,
    queries: &[f32],
    prefix_scores: &[f32],
    rotated_keys: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    values: &crate::array::Array<B>,
    sinks: Option<&[f32]>,
    num_heads: usize,
    num_groups: usize,
    sequence_length: usize,
    prefix_length: usize,
    scale: f32,
    output: &mut [f32],
) where
    f32: From<T>,
{
    let head_dim = rotated_keys.shape()[2];
    let gqa_factor = num_heads / num_groups;
    let rotated_key_rows = rotated_keys.as_slice::<T>();
    let qkv_rows = qkv.as_slice::<T>();
    let _ = values;
    let value_base = (num_heads + num_groups) * head_dim;
    let mut decoded_value_scratch = vec![0.0f32; head_dim];
    let mut decoded_value_row = vec![0.0f32; head_dim];
    let mut max_scores = vec![0.0f32; num_heads];
    let mut sum_exps = vec![0.0f32; num_heads];

    for group_index in 0..num_groups {
        let head_start = group_index * gqa_factor;
        let head_end = head_start + gqa_factor;

        for head_index in head_start..head_end {
            let output_row = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
            output_row.fill(0.0);
            max_scores[head_index] = sinks.map(|sinks| sinks[head_index]).unwrap_or(f32::NEG_INFINITY);
            sum_exps[head_index] = if sinks.is_some() {
                1.0
            } else {
                0.0
            };
        }

        for token_index in 0..prefix_length {
            let row_index = group_index * sequence_length + token_index;
            layer.decode_value_row_for_single_decode(row_index, &mut decoded_value_scratch, &mut decoded_value_row);
            for head_index in head_start..head_end {
                let output_row = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
                let score = prefix_scores[head_index * prefix_length + token_index];
                accumulate_attention_row_f32(
                    output_row,
                    &decoded_value_row,
                    score,
                    &mut max_scores[head_index],
                    &mut sum_exps[head_index],
                );
            }
        }

        let self_key_row = &rotated_key_rows[group_index * head_dim..(group_index + 1) * head_dim];
        let self_value_row = &qkv_rows[value_base + group_index * head_dim..value_base + (group_index + 1) * head_dim];
        for head_index in head_start..head_end {
            let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
            let output_row = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
            let self_score = scale
                * query
                    .iter()
                    .zip(self_key_row.iter())
                    .map(|(&query_value, &key_value)| query_value * f32::from(key_value))
                    .sum::<f32>();
            accumulate_attention_row(
                output_row,
                self_value_row,
                self_score,
                &mut max_scores[head_index],
                &mut sum_exps[head_index],
            );
            let inv_sum = 1.0 / sum_exps[head_index];
            for value in output_row.iter_mut() {
                *value *= inv_sum;
            }
        }
    }
}

fn accumulate_attention_row_f32(
    output: &mut [f32],
    value_row: &[f32],
    score: f32,
    max_score: &mut f32,
    sum_exp: &mut f32,
) {
    let new_max = f32::max(*max_score, score);
    let factor = (*max_score - new_max).exp();
    let exp_score = (score - new_max).exp();
    *max_score = new_max;
    *sum_exp = *sum_exp * factor + exp_score;
    for (output_value, &value) in output.iter_mut().zip(value_row.iter()) {
        *output_value = *output_value * factor + exp_score * value;
    }
}

fn accumulate_attention_row<T: Copy>(
    output: &mut [f32],
    value_row: &[T],
    score: f32,
    max_score: &mut f32,
    sum_exp: &mut f32,
) where
    f32: From<T>,
{
    let new_max = f32::max(*max_score, score);
    let factor = (*max_score - new_max).exp();
    let exp_score = (score - new_max).exp();
    *max_score = new_max;
    *sum_exp = *sum_exp * factor + exp_score;
    for (output_value, &value) in output.iter_mut().zip(value_row.iter()) {
        *output_value = *output_value * factor + exp_score * f32::from(value);
    }
}

fn flatten_array_as_f32<B: Backend>(array: &crate::array::Array<B>) -> Box<[f32]> {
    match array.data_type() {
        DataType::BF16 => array.as_slice::<half::bf16>().iter().map(|&value| value.to_f32()).collect(),
        DataType::F16 => array.as_slice::<half::f16>().iter().map(|&value| value.to_f32()).collect(),
        DataType::F32 => array.as_slice::<f32>().to_vec().into_boxed_slice(),
        dtype => panic!("compressed single decode does not support dtype {dtype:?}"),
    }
}

fn write_array_from_f32<B: Backend>(
    array: &mut crate::array::Array<B>,
    values: &[f32],
) {
    match array.data_type() {
        DataType::BF16 => {
            for (dst, &src) in array.as_slice_mut::<half::bf16>().iter_mut().zip(values.iter()) {
                *dst = half::bf16::from_f32(src);
            }
        },
        DataType::F16 => {
            for (dst, &src) in array.as_slice_mut::<half::f16>().iter_mut().zip(values.iter()) {
                *dst = half::f16::from_f32(src);
            }
        },
        DataType::F32 => {
            array.as_slice_mut::<f32>().copy_from_slice(values);
        },
        dtype => panic!("compressed single decode does not support dtype {dtype:?}"),
    }
}

#[cfg(feature = "tracing")]
fn sparse_value_expected_single_decode_output<B: Backend>(
    layer: &crate::forward_pass::kv_cache_layer::KVCacheLayer<B>,
    sparse_value: &crate::forward_pass::kv_cache_layer::SparseValueState,
    queries: &crate::array::Array<B>,
    rotated_keys: &crate::array::Array<B>,
    qkv: &crate::array::Array<B>,
    key_cache: &crate::array::Array<B>,
    recent_values: Option<&crate::array::Array<B>>,
    values: Option<&crate::array::Array<B>>,
    num_heads: usize,
    num_groups: usize,
    source_sequence_length: usize,
    compact_sequence_length: usize,
    head_dim: usize,
    selected_pages: &[i32],
    page_size: usize,
    recent_start: usize,
    prefix_length: usize,
    scale: f32,
) -> Vec<f32> {
    let queries = flatten_array_as_f32(queries);
    let rotated_keys = flatten_array_as_f32(rotated_keys);
    let qkv = flatten_array_as_f32(qkv);
    let key_cache = flatten_array_as_f32(key_cache);
    let mut compact_keys = vec![0.0f32; num_groups * compact_sequence_length * head_dim];
    let key_runs = sparse_value_token_runs(selected_pages, page_size, recent_start, prefix_length);

    for group_index in 0..num_groups {
        let source_group = group_index * source_sequence_length * head_dim;
        let destination_group = group_index * compact_sequence_length * head_dim;
        let mut destination_token = 0usize;
        for run in key_runs.iter() {
            let len = (run.end - run.start) * head_dim;
            let source_start = source_group + run.start * head_dim;
            let destination_start = destination_group + destination_token * head_dim;
            compact_keys[destination_start..destination_start + len]
                .copy_from_slice(&key_cache[source_start..source_start + len]);
            destination_token += run.end - run.start;
        }
        let self_start = destination_group + (compact_sequence_length - 1) * head_dim;
        let rotated_key_start = group_index * head_dim;
        compact_keys[self_start..self_start + head_dim]
            .copy_from_slice(&rotated_keys[rotated_key_start..rotated_key_start + head_dim]);
    }

    let mut compact_values = vec![0.0f32; num_groups * compact_sequence_length * head_dim];
    if let Some(recent_values) = recent_values {
        let recent_values = flatten_array_as_f32(recent_values);
        let archive_runs = sparse_value_token_runs(selected_pages, page_size, prefix_length, prefix_length);
        let max_run_tokens = archive_runs.iter().map(|run| run.end - run.start).max().unwrap_or(0);
        let mut decoded_values = vec![0.0f32; max_run_tokens.max(1) * head_dim];
        for group_index in 0..num_groups {
            let destination_group = group_index * compact_sequence_length * head_dim;
            let mut destination_token = 0usize;
            for run in archive_runs.iter() {
                let row_count = run.end - run.start;
                let decoded_len = row_count * head_dim;
                layer.decode_value_rows_for_single_decode(
                    group_index * source_sequence_length + run.start,
                    row_count,
                    &mut decoded_values[..decoded_len],
                );
                let destination_start = destination_group + destination_token * head_dim;
                compact_values[destination_start..destination_start + decoded_len]
                    .copy_from_slice(&decoded_values[..decoded_len]);
                destination_token += row_count;
            }
            if recent_start < prefix_length {
                let recent_tokens = prefix_length - recent_start;
                let recent_group = group_index * sparse_value.hot_value_capacity * head_dim;
                let recent_destination = destination_group + destination_token * head_dim;
                for token_offset in 0..recent_tokens {
                    let recent_index = (recent_start + token_offset) % sparse_value.hot_value_capacity;
                    let source_start = recent_group + recent_index * head_dim;
                    let destination_start = recent_destination + token_offset * head_dim;
                    compact_values[destination_start..destination_start + head_dim]
                        .copy_from_slice(&recent_values[source_start..source_start + head_dim]);
                }
            }
        }
    } else {
        let values = flatten_array_as_f32(values.expect("dense sparse decode expects value cache"));
        let value_runs = sparse_value_token_runs(selected_pages, page_size, recent_start, prefix_length);
        for group_index in 0..num_groups {
            let source_group = group_index * source_sequence_length * head_dim;
            let destination_group = group_index * compact_sequence_length * head_dim;
            let mut destination_token = 0usize;
            for run in value_runs.iter() {
                let len = (run.end - run.start) * head_dim;
                let source_start = source_group + run.start * head_dim;
                let destination_start = destination_group + destination_token * head_dim;
                compact_values[destination_start..destination_start + len]
                    .copy_from_slice(&values[source_start..source_start + len]);
                destination_token += run.end - run.start;
            }
        }
    }

    let value_base = (num_heads + num_groups) * head_dim;
    for group_index in 0..num_groups {
        let destination_start = (group_index * compact_sequence_length + compact_sequence_length - 1) * head_dim;
        let source_start = value_base + group_index * head_dim;
        compact_values[destination_start..destination_start + head_dim]
            .copy_from_slice(&qkv[source_start..source_start + head_dim]);
    }

    let gqa_factor = num_heads / num_groups;
    let mut output = vec![0.0f32; num_heads * head_dim];
    for head_index in 0..num_heads {
        let group_index = head_index / gqa_factor;
        let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
        let output_row = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
        let mut max_score = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        for token_index in 0..compact_sequence_length {
            let key_start = (group_index * compact_sequence_length + token_index) * head_dim;
            let value_start = key_start;
            let score = scale
                * query
                    .iter()
                    .zip(compact_keys[key_start..key_start + head_dim].iter())
                    .map(|(&query_value, &key_value)| query_value * key_value)
                    .sum::<f32>();
            accumulate_attention_row_f32(
                output_row,
                &compact_values[value_start..value_start + head_dim],
                score,
                &mut max_score,
                &mut sum_exp,
            );
        }
        let inv_sum = 1.0 / sum_exp;
        for value in output_row.iter_mut() {
            *value *= inv_sum;
        }
    }
    output
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
mod sparse_value_tests {
    use std::{
        cell::RefCell,
        ops::{Deref, DerefMut},
    };

    use crate::{
        ArrayContextExt, DataType,
        backends::{
            common::{Backend, Context, Encoder, Kernels, kernel::AttentionSinglePassKernel},
            cpu::Cpu,
            metal::Metal,
        },
        forward_pass::kv_cache_layer::{
            KVCacheLayer, KVCacheLayerState, KvCompressionMode, SparseValueConfig, SparseValueState,
        },
    };

    use super::{
        compact_sparse_value_prefix, compact_sparse_value_prefix_from_compressed_values,
        mark_sparse_value_archive_pages, select_sparse_value_prefix_selection,
    };

    #[test]
    fn sparse_value_selection_keeps_top_archive_pages() {
        let mut selected_pages = vec![false; 3];
        let mut page_indices = Vec::new();
        mark_sparse_value_archive_pages(&[9.0, 3.0, 4.0], 0.7, &mut selected_pages, &mut page_indices);
        assert_eq!(selected_pages, vec![true, false, true]);
    }

    #[test]
    fn sparse_value_selection_keeps_full_archive_when_mass_is_one() {
        let mut selected_pages = vec![false; 3];
        let mut page_indices = Vec::new();
        mark_sparse_value_archive_pages(&[3.0, 2.0, 1.0], 1.0, &mut selected_pages, &mut page_indices);
        assert_eq!(selected_pages, vec![true, true, true]);
    }

    #[test]
    fn sparse_value_prefix_selection_uses_recent_window_when_archive_is_empty() {
        let selection = select_sparse_value_prefix_selection(
            &SparseValueConfig {
                recent_window: 2,
                keep_mass: 0.0,
                page_size: 2,
            },
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[1.0],
            1,
            1,
            6,
            6,
            1,
            1.0,
        );
        assert!(selection.selected_pages.is_empty());
        assert_eq!(selection.recent_start, 4);
    }

    #[test]
    fn sparse_value_prefix_selection_keeps_full_archive_when_mass_is_one() {
        let selection = select_sparse_value_prefix_selection(
            &SparseValueConfig {
                recent_window: 0,
                keep_mass: 1.0,
                page_size: 2,
            },
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0],
            1,
            1,
            4,
            4,
            1,
            1.0,
        );
        assert_eq!(selection.selected_pages.as_ref(), &[0, 1]);
        assert_eq!(selection.recent_start, 4);
    }

    #[test]
    fn sparse_value_prefix_selection_rounds_recent_start_down_to_page_boundary() {
        let selection = select_sparse_value_prefix_selection(
            &SparseValueConfig {
                recent_window: 3,
                keep_mass: 0.0,
                page_size: 4,
            },
            &[1.0; 10],
            &[1.0],
            1,
            1,
            10,
            10,
            1,
            1.0,
        );
        assert!(selection.selected_pages.is_empty());
        assert_eq!(selection.recent_start, 4);
    }

    #[test]
    fn sparse_value_compressed_prefix_copy_uses_dense_hot_values() {
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let sparse_value = SparseValueState {
            config: SparseValueConfig {
                recent_window: 8,
                keep_mass: 1.0,
                page_size: 2,
            },
            shadow_keys: vec![0.0; 2 * 8 * 2].into_boxed_slice(),
            hot_value_capacity: 8,
            pending_suffix_len: 0,
            pending_keys: vec![0.0; 2 * 2 * 2].into_boxed_slice(),
        };
        let recent_values = context.create_array_from(
            &[2, 8, 2],
            &[
                10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, 60.0, 61.0, 70.0, 71.0, 80.0, 81.0, 110.0,
                111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0, 160.0, 161.0, 170.0, 171.0, 180.0,
                181.0,
            ]
            .map(|value| value as f32),
            "recent_values",
        );
        let layer = KVCacheLayer::<Cpu> {
            state: KVCacheLayerState::Full {
                prefix_len: 6,
            },
            shape: [2, 8, 2],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(&[2, 8, 2], DataType::F32, "keys"))),
            values: None,
            prefix_token_positions: (0..6).collect(),
            next_token_position: 6,
            max_suffix_length: 2,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(sparse_value.clone()),
            sparse_value_pending_values: None,
            sparse_value_recent_values: Some(RefCell::new(recent_values)),
            triattention: None,
        };
        let qkv = context.create_array_from(
            &[1, 12],
            &vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 900.0, 901.0, 990.0, 991.0],
            "qkv",
        );
        let mut destination = context.create_array_zeros(&[2, 7, 2], DataType::F32, "destination");
        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow().clone();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        compact_sparse_value_prefix_from_compressed_values(
            &mut encoder,
            &layer,
            &sparse_value,
            &recent_values,
            &qkv,
            2,
            &mut destination,
            2,
            8,
            7,
            2,
            &[],
            2,
            0,
            6,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        assert_eq!(
            destination.as_slice::<f32>(),
            &[
                10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, 60.0, 61.0, 900.0, 901.0, 110.0, 111.0,
                120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0, 160.0, 161.0, 990.0, 991.0,
            ]
        );
    }

    #[test]
    fn sparse_value_compressed_prefix_copy_uses_dense_hot_values_on_metal_for_long_prefix() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let head_dim = 2;
        let prefix_length = 218;
        let hot_value_capacity = 512;
        let recent_values = context.create_array_from(
            &[num_groups, hot_value_capacity, head_dim],
            &(0..num_groups)
                .flat_map(|group_index| {
                    (0..hot_value_capacity).flat_map(move |token_index| {
                        let token_base = token_index as f32 * 10.0;
                        let group_base = group_index as f32 * 100.0;
                        [token_base + group_base + 1.0, token_base + group_base + 2.0]
                    })
                })
                .collect::<Vec<_>>(),
            "recent_values",
        );
        let sparse_value = SparseValueState {
            config: SparseValueConfig {
                recent_window: hot_value_capacity,
                keep_mass: 1.0,
                page_size: 32,
            },
            shadow_keys: vec![0.0; num_groups * hot_value_capacity * head_dim].into_boxed_slice(),
            hot_value_capacity,
            pending_suffix_len: 0,
            pending_keys: vec![0.0; num_groups * 2 * head_dim].into_boxed_slice(),
        };
        let layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: prefix_length,
            },
            shape: [num_groups, hot_value_capacity, head_dim],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::F32,
                "keys",
            ))),
            values: None,
            prefix_token_positions: (0..prefix_length).collect(),
            next_token_position: prefix_length,
            max_suffix_length: 1,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(sparse_value.clone()),
            sparse_value_pending_values: None,
            sparse_value_recent_values: Some(RefCell::new(recent_values)),
            triattention: None,
        };
        let qkv = context.create_array_from(
            &[1, 16],
            &[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 900.0, 901.0, 990.0, 991.0],
            "qkv",
        );
        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow().clone();
        let mut destination =
            context.create_array_zeros(&[num_groups, prefix_length + 1, head_dim], DataType::F32, "destination");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        compact_sparse_value_prefix_from_compressed_values(
            &mut encoder,
            &layer,
            &sparse_value,
            &recent_values,
            &qkv,
            4,
            &mut destination,
            num_groups,
            hot_value_capacity,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let destination = destination.as_slice::<f32>();
        for token_index in 0..prefix_length {
            let group0_base = token_index * head_dim;
            let token_base = token_index as f32 * 10.0;
            assert_eq!(&destination[group0_base..group0_base + head_dim], &[token_base + 1.0, token_base + 2.0]);
            let group1_base = (prefix_length + 1) * head_dim + group0_base;
            assert_eq!(&destination[group1_base..group1_base + head_dim], &[token_base + 101.0, token_base + 102.0]);
        }
        let group0_self_base = prefix_length * head_dim;
        assert_eq!(&destination[group0_self_base..group0_self_base + head_dim], &[900.0, 901.0]);
        let group1_self_base = (prefix_length + 1) * head_dim + group0_self_base;
        assert_eq!(&destination[group1_self_base..group1_self_base + head_dim], &[990.0, 991.0]);
    }

    #[test]
    fn sparse_value_compressed_prefix_copy_uses_dense_hot_values_on_metal_for_long_prefix_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let head_dim = 2;
        let prefix_length = 64;
        let hot_value_capacity = 256;
        let recent_values = context.create_array_from(
            &[num_groups, hot_value_capacity, head_dim],
            &(0..num_groups)
                .flat_map(|group_index| {
                    (0..hot_value_capacity).flat_map(move |token_index| {
                        let token_base = token_index as f32 * 8.0;
                        let group_base = group_index as f32 * 128.0;
                        [
                            half::bf16::from_f32(token_base + group_base + 8.0),
                            half::bf16::from_f32(token_base + group_base + 12.0),
                        ]
                    })
                })
                .collect::<Vec<_>>(),
            "recent_values",
        );
        let sparse_value = SparseValueState {
            config: SparseValueConfig {
                recent_window: hot_value_capacity,
                keep_mass: 1.0,
                page_size: 32,
            },
            shadow_keys: vec![0.0; num_groups * hot_value_capacity * head_dim].into_boxed_slice(),
            hot_value_capacity,
            pending_suffix_len: 0,
            pending_keys: vec![0.0; num_groups * 2 * head_dim].into_boxed_slice(),
        };
        let layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: prefix_length,
            },
            shape: [num_groups, hot_value_capacity, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::BF16,
                "keys",
            ))),
            values: None,
            prefix_token_positions: (0..prefix_length).collect(),
            next_token_position: prefix_length,
            max_suffix_length: 1,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(sparse_value.clone()),
            sparse_value_pending_values: None,
            sparse_value_recent_values: Some(RefCell::new(recent_values)),
            triattention: None,
        };
        let qkv = context.create_array_from(
            &[1, 16],
            &[
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(904.0),
                half::bf16::from_f32(908.0),
                half::bf16::from_f32(1000.0),
                half::bf16::from_f32(1004.0),
            ],
            "qkv",
        );
        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow().clone();
        let mut destination =
            context.create_array_zeros(&[num_groups, prefix_length + 1, head_dim], DataType::BF16, "destination");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        compact_sparse_value_prefix_from_compressed_values(
            &mut encoder,
            &layer,
            &sparse_value,
            &recent_values,
            &qkv,
            4,
            &mut destination,
            num_groups,
            hot_value_capacity,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let destination = destination.as_slice::<half::bf16>();
        for token_index in 0..prefix_length {
            let group0_base = token_index * head_dim;
            let token_base = token_index as f32 * 8.0;
            assert_eq!(destination[group0_base].to_f32(), token_base + 8.0);
            assert_eq!(destination[group0_base + 1].to_f32(), token_base + 12.0);
            let group1_base = (prefix_length + 1) * head_dim + group0_base;
            assert_eq!(destination[group1_base].to_f32(), token_base + 136.0);
            assert_eq!(destination[group1_base + 1].to_f32(), token_base + 140.0);
        }
        let group0_self_base = prefix_length * head_dim;
        assert_eq!(destination[group0_self_base].to_f32(), 904.0);
        assert_eq!(destination[group0_self_base + 1].to_f32(), 908.0);
        let group1_self_base = (prefix_length + 1) * head_dim + group0_self_base;
        assert_eq!(destination[group1_self_base].to_f32(), 1000.0);
        assert_eq!(destination[group1_self_base + 1].to_f32(), 1004.0);
    }

    #[test]
    fn sparse_value_recent_only_compaction_matches_dense_attention_on_metal_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_heads = 4usize;
        let num_groups = 2usize;
        let head_dim = 64usize;
        let prefix_length = 8usize;
        let hot_value_capacity = 16usize;
        let qkv_width = (num_heads + num_groups + num_groups) * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let query_data = (0..num_heads)
            .flat_map(|head_index| {
                (0..head_dim).map(move |dim| half::bf16::from_f32(head_index as f32 * 0.05 + dim as f32 * 0.001 - 0.1))
            })
            .collect::<Vec<_>>();
        let key_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..=prefix_length).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        half::bf16::from_f32(group_index as f32 * 0.4 + token_index as f32 * 0.03 + dim as f32 * 0.001)
                    })
                })
            })
            .collect::<Vec<_>>();
        let recent_value_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..hot_value_capacity).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        let value = if token_index < prefix_length {
                            group_index as f32 * 10.0 + token_index as f32 * 0.07 + dim as f32 * 0.001
                        } else {
                            0.0
                        };
                        half::bf16::from_f32(value)
                    })
                })
            })
            .collect::<Vec<_>>();
        let dense_value_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..=prefix_length).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        let value = if token_index < prefix_length {
                            group_index as f32 * 10.0 + token_index as f32 * 0.07 + dim as f32 * 0.001
                        } else {
                            group_index as f32 * 10.0 + 100.0 + dim as f32 * 0.001
                        };
                        half::bf16::from_f32(value)
                    })
                })
            })
            .collect::<Vec<_>>();
        let mut qkv_data = vec![half::bf16::from_f32(0.0); qkv_width];
        let value_base = (num_heads + num_groups) * head_dim;
        for group_index in 0..num_groups {
            for dim in 0..head_dim {
                qkv_data[value_base + group_index * head_dim + dim] =
                    half::bf16::from_f32(group_index as f32 * 10.0 + 100.0 + dim as f32 * 0.001);
            }
        }

        let queries = context.create_array_from(&[num_heads, 1, head_dim], &query_data, "queries");
        let dense_keys = context.create_array_from(&[num_groups, prefix_length + 1, head_dim], &key_data, "dense_keys");
        let recent_values =
            context.create_array_from(&[num_groups, hot_value_capacity, head_dim], &recent_value_data, "recent_values");
        let qkv = context.create_array_from(&[1, qkv_width], &qkv_data, "qkv");
        let dense_values =
            context.create_array_from(&[num_groups, prefix_length + 1, head_dim], &dense_value_data, "dense_values");
        let sparse_value = SparseValueState {
            config: SparseValueConfig {
                recent_window: hot_value_capacity,
                keep_mass: 1.0,
                page_size: 32,
            },
            shadow_keys: vec![0.0; num_groups * hot_value_capacity * head_dim].into_boxed_slice(),
            hot_value_capacity,
            pending_suffix_len: 0,
            pending_keys: vec![0.0; num_groups * head_dim].into_boxed_slice(),
        };
        let layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: prefix_length,
            },
            shape: [num_groups, hot_value_capacity, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(dense_keys.clone())),
            values: None,
            prefix_token_positions: (0..prefix_length).collect(),
            next_token_position: prefix_length,
            max_suffix_length: 1,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(sparse_value.clone()),
            sparse_value_pending_values: None,
            sparse_value_recent_values: Some(RefCell::new(recent_values.clone())),
            triattention: None,
        };
        let compact_keys =
            context.create_array_zeros(&[num_groups, hot_value_capacity, head_dim], DataType::BF16, "compact_keys");
        let mut compact_values =
            context.create_array_zeros(&[num_groups, hot_value_capacity, head_dim], DataType::BF16, "compact_values");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        compact_sparse_value_prefix(
            &mut encoder,
            &dense_keys,
            &compact_keys,
            num_groups,
            prefix_length + 1,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        compact_sparse_value_prefix_from_compressed_values(
            &mut encoder,
            &layer,
            &sparse_value,
            &recent_values,
            &qkv,
            num_heads,
            &mut compact_values,
            num_groups,
            prefix_length + 1,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
            &context,
            DataType::BF16,
            head_dim as u32,
            false,
            false,
            false,
            false,
            false,
        )
        .expect("attention single-pass kernel");
        let dense_output = context.create_array_zeros(&[num_heads, 1, head_dim], DataType::BF16, "dense_output");
        let compact_output = context.create_array_zeros(&[num_heads, 1, head_dim], DataType::BF16, "compact_output");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        {
            let queries_buffer = queries.buffer();
            let queries_buffer = queries_buffer.borrow();
            let dense_keys_buffer = dense_keys.buffer();
            let dense_keys_buffer = dense_keys_buffer.borrow();
            let dense_values_buffer = dense_values.buffer();
            let dense_values_buffer = dense_values_buffer.borrow();
            let compact_keys_buffer = compact_keys.buffer();
            let compact_keys_buffer = compact_keys_buffer.borrow();
            let compact_values_buffer = compact_values.buffer();
            let compact_values_buffer = compact_values_buffer.borrow();
            let dense_output_buffer = dense_output.buffer();
            let mut dense_output_buffer = dense_output_buffer.borrow_mut();
            let compact_output_buffer = compact_output.buffer();
            let mut compact_output_buffer = compact_output_buffer.borrow_mut();

            kernel.encode(
                queries_buffer.deref(),
                dense_keys_buffer.deref(),
                dense_values_buffer.deref(),
                dense_output_buffer.deref_mut(),
                (num_heads / num_groups) as u32,
                (prefix_length + 1) as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                None,
                scale,
                None::<&<Metal as Backend>::Buffer>,
                None,
                None::<&<Metal as Backend>::Buffer>,
                num_heads as u32,
                1,
                &mut encoder,
            );
            kernel.encode(
                queries_buffer.deref(),
                compact_keys_buffer.deref(),
                compact_values_buffer.deref(),
                compact_output_buffer.deref_mut(),
                (num_heads / num_groups) as u32,
                (prefix_length + 1) as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                None,
                scale,
                None::<&<Metal as Backend>::Buffer>,
                None,
                None::<&<Metal as Backend>::Buffer>,
                num_heads as u32,
                1,
                &mut encoder,
            );
        }
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let dense_output = dense_output.as_slice::<half::bf16>().iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let compact_output =
            compact_output.as_slice::<half::bf16>().iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let max_diff = dense_output
            .iter()
            .zip(compact_output.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "SparseValue recent-only compact attention max diff {max_diff}");
    }

    #[test]
    fn sparse_value_recent_only_compaction_matches_dense_attention_on_metal_bf16_long_prefix() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_heads = 16usize;
        let num_groups = 8usize;
        let head_dim = 64usize;
        let prefix_length = 160usize;
        let hot_value_capacity = 256usize;
        let qkv_width = (num_heads + num_groups + num_groups) * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let query_data = (0..num_heads)
            .flat_map(|head_index| {
                (0..head_dim).map(move |dim| half::bf16::from_f32(head_index as f32 * 0.03 + dim as f32 * 0.002 - 0.2))
            })
            .collect::<Vec<_>>();
        let key_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..=prefix_length).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        half::bf16::from_f32(group_index as f32 * 0.5 + token_index as f32 * 0.01 + dim as f32 * 0.001)
                    })
                })
            })
            .collect::<Vec<_>>();
        let recent_value_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..hot_value_capacity).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        let value = if token_index < prefix_length {
                            group_index as f32 * 7.0 + token_index as f32 * 0.05 + dim as f32 * 0.001
                        } else {
                            0.0
                        };
                        half::bf16::from_f32(value)
                    })
                })
            })
            .collect::<Vec<_>>();
        let dense_value_data = (0..num_groups)
            .flat_map(|group_index| {
                (0..=prefix_length).flat_map(move |token_index| {
                    (0..head_dim).map(move |dim| {
                        let value = if token_index < prefix_length {
                            group_index as f32 * 7.0 + token_index as f32 * 0.05 + dim as f32 * 0.001
                        } else {
                            group_index as f32 * 7.0 + 50.0 + dim as f32 * 0.001
                        };
                        half::bf16::from_f32(value)
                    })
                })
            })
            .collect::<Vec<_>>();
        let mut qkv_data = vec![half::bf16::from_f32(0.0); qkv_width];
        let value_base = (num_heads + num_groups) * head_dim;
        for group_index in 0..num_groups {
            for dim in 0..head_dim {
                qkv_data[value_base + group_index * head_dim + dim] =
                    half::bf16::from_f32(group_index as f32 * 7.0 + 50.0 + dim as f32 * 0.001);
            }
        }

        let queries = context.create_array_from(&[num_heads, 1, head_dim], &query_data, "queries");
        let dense_keys = context.create_array_from(&[num_groups, prefix_length + 1, head_dim], &key_data, "dense_keys");
        let recent_values =
            context.create_array_from(&[num_groups, hot_value_capacity, head_dim], &recent_value_data, "recent_values");
        let qkv = context.create_array_from(&[1, qkv_width], &qkv_data, "qkv");
        let dense_values =
            context.create_array_from(&[num_groups, prefix_length + 1, head_dim], &dense_value_data, "dense_values");
        let sparse_value = SparseValueState {
            config: SparseValueConfig {
                recent_window: hot_value_capacity,
                keep_mass: 1.0,
                page_size: 32,
            },
            shadow_keys: vec![0.0; num_groups * hot_value_capacity * head_dim].into_boxed_slice(),
            hot_value_capacity,
            pending_suffix_len: 0,
            pending_keys: vec![0.0; num_groups * head_dim].into_boxed_slice(),
        };
        let layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: prefix_length,
            },
            shape: [num_groups, hot_value_capacity, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(dense_keys.clone())),
            values: None,
            prefix_token_positions: (0..prefix_length).collect(),
            next_token_position: prefix_length,
            max_suffix_length: 1,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(sparse_value.clone()),
            sparse_value_pending_values: None,
            sparse_value_recent_values: Some(RefCell::new(recent_values.clone())),
            triattention: None,
        };
        let compact_keys =
            context.create_array_zeros(&[num_groups, prefix_length + 1, head_dim], DataType::BF16, "compact_keys");
        let mut compact_values =
            context.create_array_zeros(&[num_groups, prefix_length + 1, head_dim], DataType::BF16, "compact_values");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        compact_sparse_value_prefix(
            &mut encoder,
            &dense_keys,
            &compact_keys,
            num_groups,
            prefix_length + 1,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        compact_sparse_value_prefix_from_compressed_values(
            &mut encoder,
            &layer,
            &sparse_value,
            &recent_values,
            &qkv,
            num_heads,
            &mut compact_values,
            num_groups,
            prefix_length + 1,
            prefix_length + 1,
            head_dim,
            &[],
            32,
            0,
            prefix_length,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
            &context,
            DataType::BF16,
            head_dim as u32,
            false,
            false,
            false,
            false,
            false,
        )
        .expect("attention single-pass kernel");
        let dense_output = context.create_array_zeros(&[num_heads, 1, head_dim], DataType::BF16, "dense_output");
        let compact_output = context.create_array_zeros(&[num_heads, 1, head_dim], DataType::BF16, "compact_output");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        {
            let queries_buffer = queries.buffer();
            let queries_buffer = queries_buffer.borrow();
            let dense_keys_buffer = dense_keys.buffer();
            let dense_keys_buffer = dense_keys_buffer.borrow();
            let dense_values_buffer = dense_values.buffer();
            let dense_values_buffer = dense_values_buffer.borrow();
            let compact_keys_buffer = compact_keys.buffer();
            let compact_keys_buffer = compact_keys_buffer.borrow();
            let compact_values_buffer = compact_values.buffer();
            let compact_values_buffer = compact_values_buffer.borrow();
            let dense_output_buffer = dense_output.buffer();
            let mut dense_output_buffer = dense_output_buffer.borrow_mut();
            let compact_output_buffer = compact_output.buffer();
            let mut compact_output_buffer = compact_output_buffer.borrow_mut();

            kernel.encode(
                queries_buffer.deref(),
                dense_keys_buffer.deref(),
                dense_values_buffer.deref(),
                dense_output_buffer.deref_mut(),
                (num_heads / num_groups) as u32,
                (prefix_length + 1) as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                None,
                scale,
                None::<&<Metal as Backend>::Buffer>,
                None,
                None::<&<Metal as Backend>::Buffer>,
                num_heads as u32,
                1,
                &mut encoder,
            );
            kernel.encode(
                queries_buffer.deref(),
                compact_keys_buffer.deref(),
                compact_values_buffer.deref(),
                compact_output_buffer.deref_mut(),
                (num_heads / num_groups) as u32,
                (prefix_length + 1) as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                ((prefix_length + 1) * head_dim) as u32,
                head_dim as u32,
                None,
                scale,
                None::<&<Metal as Backend>::Buffer>,
                None,
                None::<&<Metal as Backend>::Buffer>,
                num_heads as u32,
                1,
                &mut encoder,
            );
        }
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let dense_output = dense_output.as_slice::<half::bf16>().iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let compact_output =
            compact_output.as_slice::<half::bf16>().iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let max_diff = dense_output
            .iter()
            .zip(compact_output.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "SparseValue long recent-only compact attention max diff {max_diff}");
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/attention_test.rs"]
mod tests;
