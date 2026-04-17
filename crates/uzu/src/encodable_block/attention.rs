//! Attention kernel encodable.

use std::collections::HashMap;

use itertools::iproduct;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel,
            SigmoidGateKernel,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
        },
    },
    forward_pass::kv_cache_layer::{KVCacheLayer, KVCacheLayerState},
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
    data_type: DataType,
    attention_scale: Option<f32>,
    is_causal: bool,
    sliding_window_size: Option<usize>,
}

pub struct AttentionArguments<'a, B: Backend> {
    pub context: &'a B::Context,
    pub projection_step: usize,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub attention_sinks: Option<&'a Allocation<B>>,
    pub kv_cache_layer: Option<&'a mut KVCacheLayer<B>>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
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
            data_type,
            attention_scale,
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
        mut args: AttentionArguments<'_, B>,
        qkv: &Allocation<B>,
        queries: &Allocation<B>,
        mut rotated_keys: Allocation<B>,
        gate: Option<&Allocation<B>>,
        suffix_length: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let is_trie = args.token_subtrie_ranges.is_some();

        let (max_sequence_length, segment_prefix_length, ring_params) =
            if let Some(layer) = args.kv_cache_layer.as_deref() {
                let max_sequence_length = layer.shape[1];
                let ring_params = match layer.state {
                    KVCacheLayerState::Windowed {
                        ring_offset,
                        ring_length,
                        window_length,
                    } => {
                        let overflow = (ring_length + args.projection_step).saturating_sub(window_length);
                        Some(RingParams {
                            ring_offset: ((ring_offset + overflow) % window_length) as u32,
                            ring_length: (ring_length + args.projection_step).min(window_length) as u32,
                        })
                    },
                    _ => None,
                };
                (max_sequence_length, layer.projected_segment_prefix_length(args.projection_step), ring_params)
            } else {
                (suffix_length, 0, None)
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

        let sinks_allocation = args.attention_sinks;
        let trie_allocation = args.token_subtrie_ranges;

        // Get KV cache buffers only if KV cache exists (LLM mode)
        let has_kv_cache = args.kv_cache_layer.is_some();
        let mut extracted_values = None;

        // For classifiers (no KV cache): extract values from QKV into a dedicated extracted_values buffer.
        if !has_kv_cache {
            let mut values =
                encoder.allocate_scratch(size_for_shape(&[num_groups, suffix_length, head_dim], self.data_type))?;
            self.update_kv_cache_inplace_kernel.encode(
                None::<&Allocation<B>>,
                qkv,
                &mut rotated_keys,
                &mut values,
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                0u32,
                max_sequence_length as u32,
                encoder,
            );
            extracted_values = Some(values);
        }

        let k_head_stride = (max_sequence_length * head_dim) as i32;
        let k_seq_stride = head_dim as i32;
        let v_head_stride = (max_sequence_length * head_dim) as i32;
        let v_seq_stride = head_dim as i32;

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };

        let mut attention_output =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, num_heads * head_dim], self.data_type))?;

        let (keys, values) = if let Some(layer) = args.kv_cache_layer.as_deref_mut() {
            self.update_kv_cache_kernel.encode(
                Some(&rotated_keys),
                qkv,
                &mut layer.keys,
                &mut layer.values,
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                segment_prefix_length as u32,
                max_sequence_length as u32,
                encoder,
            );
            (&layer.keys, &layer.values)
        } else {
            (&rotated_keys, extracted_values.as_ref().expect("Missing extracted values for classifier attention"))
        };

        Self::encode_attention_variant(
            self,
            args.context,
            variant,
            &kernel_key,
            queries,
            keys,
            values,
            &mut attention_output,
            trie_allocation,
            sinks_allocation,
            gqa_factor,
            sequence_length,
            k_head_stride,
            k_seq_stride,
            v_head_stride,
            v_seq_stride,
            ring_params,
            scale,
            num_heads,
            num_groups,
            suffix_length,
            segment_prefix_length,
            max_sequence_length,
            head_dim,
            encoder,
        );

        if let Some(gate_kernel) = &self.gate_kernel {
            let total_elements = (suffix_length * num_heads * head_dim) as u32;
            let gate = gate.expect("Gate allocation not initialized");
            gate_kernel.encode(gate, &mut attention_output, total_elements, encoder);
        }

        Ok(attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_variant(
        &self,
        context: &B::Context,
        variant: KernelVariant,
        kernel_key: &KernelKey,
        queries: &Allocation<B>,
        keys: &Allocation<B>,
        values: &Allocation<B>,
        attention_output: &mut Allocation<B>,
        trie_allocation: Option<&Allocation<B>>,
        sinks_allocation: Option<&Allocation<B>>,
        gqa_factor: usize,
        sequence_length: usize,
        k_head_stride: i32,
        k_seq_stride: i32,
        v_head_stride: i32,
        v_seq_stride: i32,
        ring_params: Option<RingParams>,
        scale: f32,
        num_heads: usize,
        num_groups: usize,
        suffix_length: usize,
        segment_prefix_length: usize,
        max_sequence_length: usize,
        head_dim: usize,
        encoder: &mut Encoder<B>,
    ) {
        match variant {
            KernelVariant::Gemm => {
                let args = AttentionGemmArguments {
                    queries_buffer: queries,
                    keys_buffer: keys,
                    values_buffer: values,
                    output_buffer: attention_output,
                    trie_buffer: trie_allocation,
                    sinks_buffer: sinks_allocation,
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
                self.gemm_block.encode(context, encoder, args).expect("Failed to encode AttentionGemmBlock");
            },
            KernelVariant::SinglePass => {
                let kernel = self
                    .single_pass_kernels
                    .get(kernel_key)
                    .unwrap_or_else(|| panic!("Can not find AttentionSinglePassKernel for key {:?}", kernel_key));
                kernel.encode(
                    queries,
                    keys,
                    values,
                    attention_output,
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride as u32,
                    k_seq_stride as u32,
                    v_head_stride as u32,
                    v_seq_stride as u32,
                    ring_params,
                    scale,
                    trie_allocation,
                    self.sliding_window_size.map(|s| s as u32),
                    sinks_allocation,
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                );
            },
            KernelVariant::TwoPass => {
                let kernel_pass1 = self
                    .two_pass_1_kernels
                    .get(kernel_key)
                    .unwrap_or_else(|| panic!("Can not find AttentionTwoPass1Kernel for key {:?}", kernel_key));
                let kernel_pass2 = self
                    .two_pass_2_kernels
                    .get(&(head_dim as u32))
                    .unwrap_or_else(|| panic!("Can not find AttentionTwoPass2Kernel for key {:?}", kernel_key));
                let mut partials = encoder
                    .allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32 * head_dim], DataType::F32))
                    .expect("Failed to allocate attention partials");
                let mut sums = encoder
                    .allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32], DataType::F32))
                    .expect("Failed to allocate attention sums");
                let mut maxs = encoder
                    .allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32], DataType::F32))
                    .expect("Failed to allocate attention maxs");
                kernel_pass1.encode(
                    queries,
                    keys,
                    values,
                    &mut partials,
                    &mut sums,
                    &mut maxs,
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
                    trie_allocation,
                    self.sliding_window_size.map(|s| s as u32),
                    sinks_allocation,
                    encoder,
                );
                kernel_pass2.encode(
                    &partials,
                    &sums,
                    &maxs,
                    attention_output,
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                );
            },
        }
    }
}

#[derive(Clone, Copy)]
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
