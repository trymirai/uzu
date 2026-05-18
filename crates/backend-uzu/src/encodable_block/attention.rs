//! Attention kernel encodable.

use std::{cell::RefCell, collections::HashMap};

use itertools::iproduct;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionFallbackScatterScoresKernel, AttentionFallbackScatterValuesKernel, AttentionSinglePassKernel,
            AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel, ManualKernels,
            SigmoidGateKernel, SoftmaxKernel,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
            matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
        },
    },
    config::AttentionConfig,
    forward_pass::{cache_layers::LayerCacheAccess, kv_cache_layer::KVCacheLayerState},
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
    softmax_kernel: <B::Kernels as Kernels>::SoftmaxKernel,
    softmax_sinks_kernel: Option<<B::Kernels as Kernels>::SoftmaxKernel>,
    fallback_scatter_scores_kernels: HashMap<bool, <B::Kernels as Kernels>::AttentionFallbackScatterScoresKernel>,
    fallback_scatter_values_kernel: <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel,
    matmul_kernel: Option<RefCell<<B::Kernels as ManualKernels>::MatmulKernel>>,
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
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub attention_sinks: Option<&'a Allocation<B>>,
    pub cache_access: Option<LayerCacheAccess<'a, B>>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        config: &AttentionConfig,
        has_gate: bool,
    ) -> Result<Self, B::Error> {
        let mut single_pass_kernels = HashMap::new();
        let mut two_pass_1_kernels = HashMap::new();
        let mut two_pass_2_kernels = HashMap::new();
        let mut fallback_scatter_scores_kernels = HashMap::new();

        for (head_dim, is_trie, is_kv_cache_ring) in
            iproduct!([64u32, 128u32, 256u32, 512u32], [false, true], [false, true])
        {
            let key = KernelKey {
                head_dim,
                is_trie,
                is_kv_cache_ring,
            };

            let sp_kernel = <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                context,
                data_type,
                head_dim,
                config.has_sinks,
                is_kv_cache_ring,
                config.is_causal,
                is_trie,
                config.sliding_window_size.is_some(),
            )?;
            single_pass_kernels.insert(key, sp_kernel);

            let tp1_kernel = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
                context,
                data_type,
                head_dim,
                config.has_sinks,
                is_kv_cache_ring,
                config.is_causal,
                is_trie,
                config.sliding_window_size.is_some(),
            )?;
            two_pass_1_kernels.insert(key, tp1_kernel);

            let tp2_kernel = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, data_type, head_dim)?;
            two_pass_2_kernels.insert(head_dim, tp2_kernel);
        }

        for is_kv_cache_ring in [false, true] {
            let scatter = <B::Kernels as Kernels>::AttentionFallbackScatterScoresKernel::new(
                context,
                data_type,
                is_kv_cache_ring,
                config.is_causal,
                false,
                config.sliding_window_size.is_some(),
            )?;
            fallback_scatter_scores_kernels.insert(is_kv_cache_ring, scatter);
        }

        let softmax_kernel = <B::Kernels as Kernels>::SoftmaxKernel::new(context, data_type, false)?;
        let softmax_sinks_kernel = if config.has_sinks {
            Some(<B::Kernels as Kernels>::SoftmaxKernel::new(context, data_type, true)?)
        } else {
            None
        };
        let fallback_scatter_values_kernel =
            <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel::new(context, data_type)?;
        let matmul_kernel =
            <<B as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, data_type).ok().map(RefCell::new);
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
            softmax_kernel,
            softmax_sinks_kernel,
            fallback_scatter_scores_kernels,
            fallback_scatter_values_kernel,
            matmul_kernel,
            update_kv_cache_kernel,
            update_kv_cache_inplace_kernel,
            gate_kernel,
            gemm_block,
            data_type,
            attention_scale: config.scale,
            is_causal: config.is_causal,
            sliding_window_size: config.sliding_window_size,
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
        // head_dim=512 doesn't fit the fused GEMM tile in threadgroup memory.
        if head_dim == 512 && suffix_length > 8 && !is_trie {
            return KernelVariant::Fallback;
        }

        if gemm_enabled && suffix_length > 8 && matches!(head_dim, 64 | 128 | 256) {
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
        args: AttentionArguments<B>,
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

        let cache_layer = args.cache_access.as_ref().map(|a| {
            match a {
                LayerCacheAccess::Owned {
                    entry,
                } => entry.as_transformer(),
                LayerCacheAccess::Shared {
                    source,
                } => source.as_transformer(),
            }
            .expect("attention expects transformer cache")
        });

        let (max_sequence_length, segment_prefix_length, ring_params) = if let Some(layer) = cache_layer {
            let max_sequence_length = layer.shape[0];
            let ring_params = match layer.state {
                KVCacheLayerState::Windowed {
                    ring_offset,
                    ring_length,
                    window_length,
                } => {
                    let overflow = ring_length.saturating_sub(window_length);
                    Some(RingParams {
                        ring_offset: ((ring_offset + overflow) % window_length) as u32,
                        ring_length: ring_length.min(window_length) as u32,
                    })
                },
                _ => None,
            };
            (max_sequence_length, layer.prefix_segment_length(), ring_params)
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

        let has_kv_cache = args.cache_access.is_some();
        let extracted_values = (!has_kv_cache)
            .then(|| {
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
                Ok(values)
            })
            .transpose()?;

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

        let mut attention_output = if let Some(access) = args.cache_access {
            let (keys, values) = match access {
                LayerCacheAccess::Owned {
                    entry,
                } => {
                    let layer = entry.as_transformer_mut().expect("Attention layer expects transformer cache");
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
                },
                LayerCacheAccess::Shared {
                    source,
                } => {
                    let source = source.as_transformer().expect("kv_source must be a transformer cache");
                    (&source.keys, &source.values)
                },
            };

            self.encode_attention_variant(
                variant,
                &kernel_key,
                queries,
                keys,
                values,
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
            )?
        } else {
            let values = extracted_values.as_ref().expect("Missing extracted values for classifier attention");
            self.encode_attention_variant(
                variant,
                &kernel_key,
                queries,
                &rotated_keys,
                values,
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
            )?
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            let total_elements = (suffix_length * num_heads * head_dim) as u32;
            let gate = gate.expect("Gate allocation not initialized");
            gate_kernel.encode(gate, &mut attention_output, total_elements, encoder);
        }

        Ok(attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_variant<
        Keys: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
        Values: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    >(
        &self,
        variant: KernelVariant,
        kernel_key: &KernelKey,
        queries: &Allocation<B>,
        keys: &Keys,
        values: &Values,
        trie_allocation: Option<&Allocation<B>>,
        sinks_allocation: Option<&Allocation<B>>,
        gqa_factor: usize,
        sequence_length: usize,
        k_head_stride: u32,
        k_seq_stride: u32,
        v_head_stride: u32,
        v_seq_stride: u32,
        ring_params: Option<RingParams>,
        scale: f32,
        num_heads: usize,
        num_groups: usize,
        suffix_length: usize,
        segment_prefix_length: usize,
        max_sequence_length: usize,
        head_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut attention_output =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, num_heads * head_dim], self.data_type))?;
        match variant {
            KernelVariant::Gemm => {
                let args = AttentionGemmArguments {
                    queries,
                    keys,
                    values,
                    output: &mut attention_output,
                    trie: trie_allocation,
                    sinks: sinks_allocation,
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
                self.gemm_block.encode(encoder, args)?;
            },
            KernelVariant::Fallback => {
                self.encode_fallback(
                    kernel_key.is_kv_cache_ring,
                    queries,
                    keys,
                    values,
                    &mut attention_output,
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
                    head_dim,
                    encoder,
                )?;
            },
            KernelVariant::SinglePass => {
                let kernel = self.single_pass_kernels.get(kernel_key).expect("single_pass kernel missing");
                kernel.encode(
                    queries,
                    keys,
                    values,
                    &mut attention_output,
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride,
                    k_seq_stride,
                    v_head_stride,
                    v_seq_stride,
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
                let kernel_pass1 = self.two_pass_1_kernels.get(kernel_key).expect("two_pass_1 kernel missing");
                let kernel_pass2 = self.two_pass_2_kernels.get(&(head_dim as u32)).expect("two_pass_2 kernel missing");
                let mut partials = encoder
                    .allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32 * head_dim], DataType::F32))?;
                let mut sums =
                    encoder.allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32], DataType::F32))?;
                let mut maxs =
                    encoder.allocate_scratch(size_for_shape(&[num_heads * suffix_length * 32], DataType::F32))?;
                kernel_pass1.encode(
                    queries,
                    keys,
                    values,
                    &mut partials,
                    &mut sums,
                    &mut maxs,
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
                    trie_allocation,
                    self.sliding_window_size.map(|s| s as u32),
                    sinks_allocation,
                    encoder,
                );
                kernel_pass2.encode(
                    &partials,
                    &sums,
                    &maxs,
                    &mut attention_output,
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                );
            },
        }

        Ok(attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_fallback<
        Keys: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
        Values: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    >(
        &self,
        is_kv_cache_ring: bool,
        queries: &Allocation<B>,
        keys: &Keys,
        values: &Values,
        attention_output: &mut Allocation<B>,
        sinks_allocation: Option<&Allocation<B>>,
        gqa_factor: usize,
        sequence_length: usize,
        k_head_stride: u32,
        k_seq_stride: u32,
        v_head_stride: u32,
        v_seq_stride: u32,
        ring_params: Option<RingParams>,
        scale: f32,
        num_heads: usize,
        num_groups: usize,
        suffix_length: usize,
        head_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let matmul = self.matmul_kernel.as_ref().expect("MatmulKernel required for head_dim=512 fallback");
        let scatter_scores =
            self.fallback_scatter_scores_kernels.get(&is_kv_cache_ring).expect("scatter_scores kernel missing");
        let dt_bytes = self.data_type.size_in_bytes();

        let mut scores =
            encoder.allocate_scratch(size_for_shape(&[num_heads * suffix_length * sequence_length], self.data_type))?;
        let mut group_scores =
            encoder.allocate_scratch(size_for_shape(&[gqa_factor * suffix_length, sequence_length], self.data_type))?;

        for group_index in 0..num_groups {
            matmul.borrow_mut().encode(
                MatmulArguments {
                    a: queries,
                    a_offset: group_index * gqa_factor * suffix_length * head_dim * dt_bytes,
                    b: keys,
                    b_offset: group_index * k_head_stride as usize * dt_bytes,
                    b_leading_dimension: Some(k_seq_stride),
                    b_transpose: true,
                    ab_scale: scale,
                    c: MatmulArgumentC::None,
                    d: &mut group_scores,
                    batch_dim: (gqa_factor * suffix_length) as u32,
                    input_dim: head_dim as u32,
                    output_dim: sequence_length as u32,
                },
                encoder,
            );
            scatter_scores.encode(
                &group_scores,
                &mut scores,
                ring_params,
                None::<&Allocation<B>>,
                self.sliding_window_size.map(|s| s as u32),
                group_index as u32,
                gqa_factor as u32,
                sequence_length as u32,
                suffix_length as u32,
                (gqa_factor * suffix_length * sequence_length) as u32,
                encoder,
            );
        }

        let softmax = if sinks_allocation.is_some() {
            self.softmax_sinks_kernel.as_ref().expect("softmax_sinks_kernel missing but sinks provided")
        } else {
            &self.softmax_kernel
        };
        softmax.encode(
            &mut scores,
            sinks_allocation,
            sequence_length as u32,
            num_heads as u32,
            suffix_length as u32,
            encoder,
        );

        let mut group_output =
            encoder.allocate_scratch(size_for_shape(&[gqa_factor * suffix_length, head_dim], self.data_type))?;

        for group_index in 0..num_groups {
            matmul.borrow_mut().encode(
                MatmulArguments {
                    a: &scores,
                    a_offset: group_index * gqa_factor * suffix_length * sequence_length * dt_bytes,
                    b: values,
                    b_offset: group_index * v_head_stride as usize * dt_bytes,
                    b_leading_dimension: Some(v_seq_stride),
                    b_transpose: false,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: &mut group_output,
                    batch_dim: (gqa_factor * suffix_length) as u32,
                    input_dim: sequence_length as u32,
                    output_dim: head_dim as u32,
                },
                encoder,
            );
            self.fallback_scatter_values_kernel.encode(
                &group_output,
                &mut *attention_output,
                group_index as u32,
                gqa_factor as u32,
                suffix_length as u32,
                num_heads as u32,
                head_dim as u32,
                (gqa_factor * suffix_length * head_dim) as u32,
                encoder,
            );
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
enum KernelVariant {
    Gemm,
    Fallback,
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
