//! Attention kernel encodable.

mod gemm;

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use gemm::{AttentionGemmArguments, AttentionGemmBlock};
use itertools::iproduct;
use thiserror::Error;

use super::{Linear, QKVNorm, QkUnpack, Rope, linear::LinearBlockError, qkv_norm::QKVNormError};
use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionFallbackScatterScoresKernel, AttentionFallbackScatterValuesKernel, AttentionSinglePassKernel,
            AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel, SigmoidGateKernel,
            SoftmaxKernel,
            matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    config::token_mixer::attention::AttentionConfig,
    data_type::DataType,
    forward_pass::{
        cache_layers::LayerCacheAccess,
        kv_cache_layer::{KVCacheLayer, KVCacheLayerState},
        state::RopeBuffers,
    },
    parameters::{ParameterLoaderError, ParameterTree},
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
    qkv_projection: Box<dyn Linear<B>>,
    gate_projection: Option<Box<dyn Linear<B>>>,
    qkv_norm: Option<QKVNorm<B>>,
    rope: Rc<Rope<B>>,
    qk_unpack: Rc<QkUnpack<B>>,
    out_projection: Box<dyn Linear<B>>,
    sinks: Option<Allocation<B>>,
    single_pass_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionSinglePassKernel>,
    two_pass_1_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionTwoPass1Kernel>,
    two_pass_2_kernels: HashMap<u32, <B::Kernels as Kernels>::AttentionTwoPass2Kernel>,
    softmax_kernel: <B::Kernels as Kernels>::SoftmaxKernel,
    softmax_sinks_kernel: Option<<B::Kernels as Kernels>::SoftmaxKernel>,
    fallback_scatter_scores_kernels: HashMap<bool, <B::Kernels as Kernels>::AttentionFallbackScatterScoresKernel>,
    fallback_scatter_values_kernel: <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel,
    matmul_kernel: Option<RefCell<<B::Kernels as Kernels>::MatmulKernel>>,
    update_kv_cache_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    update_kv_cache_inplace_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    gate_kernel: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
    gemm_block: AttentionGemmBlock<B>,
    data_type: DataType,
    attention_scale: Option<f32>,
    is_causal: bool,
    sliding_window_size: Option<usize>,
    model_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
}

#[derive(Debug, Error)]
pub enum AttentionError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear block error: {0}")]
    LinearBlockError(#[from] LinearBlockError<B>),
    #[error("QKV norm error: {0}")]
    QKVNormError(#[from] QKVNormError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

impl<B: Backend> Attention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        model_dim: usize,
        data_type: DataType,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<B>,
        rope: Rc<Rope<B>>,
        qk_unpack: Rc<QkUnpack<B>>,
        extract_input_hadamard: bool,
    ) -> Result<(Self, Option<Allocation<B>>), AttentionError<B>> {
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_groups * config.head_dim;

        let qkv_projection_tree = parameter_tree.subtree("qkv_projection")?;
        let (qkv_projection, input_hadamard_factors) = if extract_input_hadamard {
            <dyn Linear<B>>::new_extracting_input_hadamard(
                model_dim,
                [q_dim, kv_dim, kv_dim],
                config.has_qkv_biases,
                context,
                data_type,
                &qkv_projection_tree,
            )?
        } else {
            (
                <dyn Linear<B>>::new(
                    model_dim,
                    [q_dim, kv_dim, kv_dim],
                    config.has_qkv_biases,
                    context,
                    data_type,
                    &qkv_projection_tree,
                )?,
                None,
            )
        };

        let gate_projection = config
            .gate_projection_config
            .as_ref()
            .map(|_| {
                let gate_projection_tree = parameter_tree.subtree("gate_projection")?;
                <dyn Linear<B>>::new(model_dim, [q_dim], false, context, data_type, &gate_projection_tree)
            })
            .transpose()?;

        let value_norm_config = config.value_norm_config();
        let qkv_norm =
            if config.query_norm_config.is_some() || config.key_norm_config.is_some() || value_norm_config.is_some() {
                Some(QKVNorm::new(
                    context,
                    data_type,
                    config.query_norm_config.clone(),
                    config.key_norm_config.clone(),
                    value_norm_config,
                    parameter_tree,
                    config.num_heads,
                    config.num_groups,
                    config.head_dim,
                )?)
            } else {
                None
            };

        let out_projection_tree = parameter_tree.subtree("out_projection")?;
        let out_projection =
            <dyn Linear<B>>::new(q_dim, [model_dim], config.has_out_biases, context, data_type, &out_projection_tree)?;

        let sinks = if config.has_sinks {
            Some(parameter_tree.leaf("sinks")?.validate(&[config.num_heads], data_type)?.read_allocation()?)
        } else {
            None
        };

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
            )
            .map_err(AttentionError::BackendError)?;
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
            )
            .map_err(AttentionError::BackendError)?;
            two_pass_1_kernels.insert(key, tp1_kernel);

            let tp2_kernel = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, data_type, head_dim)
                .map_err(AttentionError::BackendError)?;
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
            )
            .map_err(AttentionError::BackendError)?;
            fallback_scatter_scores_kernels.insert(is_kv_cache_ring, scatter);
        }

        let softmax_kernel = <B::Kernels as Kernels>::SoftmaxKernel::new(context, data_type, false)
            .map_err(AttentionError::BackendError)?;
        let softmax_sinks_kernel = if config.has_sinks {
            Some(
                <B::Kernels as Kernels>::SoftmaxKernel::new(context, data_type, true)
                    .map_err(AttentionError::BackendError)?,
            )
        } else {
            None
        };
        let fallback_scatter_values_kernel =
            <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel::new(context, data_type)
                .map_err(AttentionError::BackendError)?;
        let matmul_kernel =
            <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                .ok()
                .map(RefCell::new);
        let update_kv_cache_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, false)
                .map_err(AttentionError::BackendError)?;
        let update_kv_cache_inplace_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, true)
                .map_err(AttentionError::BackendError)?;
        let gate_kernel = if gate_projection.is_some() {
            Some(
                <B::Kernels as Kernels>::SigmoidGateKernel::new(context, data_type)
                    .map_err(AttentionError::BackendError)?,
            )
        } else {
            None
        };
        let gemm_block = AttentionGemmBlock::new(data_type);

        Ok((
            Self {
                qkv_projection,
                gate_projection,
                qkv_norm,
                rope,
                qk_unpack,
                out_projection,
                sinks,
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
                model_dim,
                num_heads: config.num_heads,
                num_groups: config.num_groups,
                head_dim: config.head_dim,
            },
            input_hadamard_factors,
        ))
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

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        token_positions: &Allocation<B>,
        token_subtrie_ranges: Option<&Allocation<B>>,
        rope_buffers: Option<&RopeBuffers<B>>,
        cache_access: Option<LayerCacheAccess<B>>,
        hidden: Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let (qkv_input, gate) = if let Some(gate_projection) = &self.gate_projection {
            let mut qkv_input =
                encoder.allocate_scratch(size_for_shape(&[suffix_length, self.model_dim], self.data_type))?;
            encoder.encode_copy(&hidden, .., &mut qkv_input, ..);
            let gate = gate_projection.encode(hidden, suffix_length, encoder)?;
            (qkv_input, Some(gate))
        } else {
            (hidden, None)
        };

        let mut qkv = self.qkv_projection.encode(qkv_input, suffix_length, encoder)?;
        if let Some(qkv_norm) = &self.qkv_norm {
            qkv_norm.encode(&mut qkv, suffix_length, encoder)?;
        }
        let (queries, rotated_keys) = match rope_buffers {
            Some(rope_buffers) => self.rope.encode(
                &qkv,
                token_positions,
                &rope_buffers.cosines,
                &rope_buffers.sines,
                suffix_length,
                self.num_heads,
                self.num_groups,
                self.head_dim,
                rope_buffers.max_sequence_length(),
                rope_buffers.dim(),
                encoder,
            )?,
            None => {
                self.qk_unpack.encode(&qkv, suffix_length, self.num_heads, self.num_groups, self.head_dim, encoder)?
            },
        };
        let attention_output = self.encode_core(
            token_subtrie_ranges,
            cache_access,
            &qkv,
            &queries,
            rotated_keys,
            gate.as_ref(),
            suffix_length,
            encoder,
        )?;
        self.out_projection.encode(attention_output, suffix_length, encoder)
    }

    fn encode_core(
        &self,
        token_subtrie_ranges: Option<&Allocation<B>>,
        cache_access: Option<LayerCacheAccess<B>>,
        qkv: &Allocation<B>,
        queries: &Allocation<B>,
        mut rotated_keys: Allocation<B>,
        gate: Option<&Allocation<B>>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let is_trie = token_subtrie_ranges.is_some();

        let cache_layer = cache_access.as_ref().map(|a| {
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
            let max_sequence_length = layer.shape()[0];
            let ring_params = match layer.state() {
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

        let gqa_factor = self.num_heads / self.num_groups;
        let scale = self.attention_scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt());

        let gemm_enabled = env_gemm_attention_enabled();
        if !gemm_enabled {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(|| {
                eprintln!("[uzu] Gemm attention disabled via UZU_USE_GEMM_ATTENTION");
            });
        }
        let variant =
            self.select_variant(gemm_enabled, suffix_length, self.head_dim, sequence_length, is_trie, is_kv_cache_ring);

        let sinks_allocation = self.sinks.as_ref();
        let trie_allocation = token_subtrie_ranges;

        let has_kv_cache = cache_access.is_some();
        let extracted_values = (!has_kv_cache)
            .then(|| {
                let mut values = encoder.allocate_scratch(size_for_shape(
                    &[self.num_groups, suffix_length, self.head_dim],
                    self.data_type,
                ))?;
                self.update_kv_cache_inplace_kernel.encode(
                    None::<&Allocation<B>>,
                    qkv,
                    &mut rotated_keys,
                    &mut values,
                    self.num_groups as u32,
                    self.num_heads as u32,
                    self.head_dim as u32,
                    suffix_length as u32,
                    0u32,
                    max_sequence_length as u32,
                    encoder,
                );
                Ok::<_, B::Error>(values)
            })
            .transpose()?;

        // KV cache layout: [max_sequence_length, num_groups, head_dim] (token-major)
        // For classifier mode (no KV cache) keys/values still come in group-major layout
        // [num_groups, suffix_length, head_dim].
        let (k_head_stride, k_seq_stride, v_head_stride, v_seq_stride) = if has_kv_cache {
            (
                self.head_dim as u32,
                (self.num_groups * self.head_dim) as u32,
                self.head_dim as u32,
                (self.num_groups * self.head_dim) as u32,
            )
        } else {
            (
                (max_sequence_length * self.head_dim) as u32,
                self.head_dim as u32,
                (max_sequence_length * self.head_dim) as u32,
                self.head_dim as u32,
            )
        };

        let kernel_key = KernelKey {
            head_dim: self.head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };

        macro_rules! encode_cached_attention {
            ($keys:expr, $values:expr) => {
                self.encode_attention_variant(
                    variant,
                    &kernel_key,
                    queries,
                    $keys,
                    $values,
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
                    self.num_heads,
                    self.num_groups,
                    suffix_length,
                    segment_prefix_length,
                    self.head_dim,
                    encoder,
                )?
            };
        }

        let mut attention_output = if let Some(access) = cache_access {
            match access {
                LayerCacheAccess::Owned {
                    entry,
                } => {
                    let layer = entry.as_transformer_mut().expect("Attention layer expects transformer cache");
                    if let Some(layer) = layer.as_any_mut().downcast_mut::<KVCacheLayer<B, B::SparseBuffer>>() {
                        self.update_kv_cache_kernel.encode(
                            Some(&rotated_keys),
                            qkv,
                            &mut layer.keys,
                            &mut layer.values,
                            self.num_groups as u32,
                            self.num_heads as u32,
                            self.head_dim as u32,
                            suffix_length as u32,
                            segment_prefix_length as u32,
                            max_sequence_length as u32,
                            encoder,
                        );
                        encode_cached_attention!(&layer.keys, &layer.values)
                    } else if let Some(layer) = layer.as_any_mut().downcast_mut::<KVCacheLayer<B, B::DenseBuffer>>() {
                        self.update_kv_cache_kernel.encode(
                            Some(&rotated_keys),
                            qkv,
                            &mut layer.keys,
                            &mut layer.values,
                            self.num_groups as u32,
                            self.num_heads as u32,
                            self.head_dim as u32,
                            suffix_length as u32,
                            segment_prefix_length as u32,
                            max_sequence_length as u32,
                            encoder,
                        );
                        encode_cached_attention!(&layer.keys, &layer.values)
                    } else {
                        panic!("Attention layer expects sparse or dense transformer cache")
                    }
                },
                LayerCacheAccess::Shared {
                    source,
                } => {
                    let source = source.as_transformer().expect("kv_source must be a transformer cache");
                    if let Some(source) = source.as_any().downcast_ref::<KVCacheLayer<B, B::SparseBuffer>>() {
                        encode_cached_attention!(&source.keys, &source.values)
                    } else if let Some(source) = source.as_any().downcast_ref::<KVCacheLayer<B, B::DenseBuffer>>() {
                        encode_cached_attention!(&source.keys, &source.values)
                    } else {
                        panic!("kv_source must be a sparse or dense transformer cache")
                    }
                },
            }
        } else {
            let values = extracted_values.as_ref().expect("Missing extracted values for classifier attention");
            encode_cached_attention!(&rotated_keys, values)
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            let total_elements = (suffix_length * self.num_heads * self.head_dim) as u32;
            let gate = gate.expect("Gate allocation not initialized");
            gate_kernel.encode(gate, &mut attention_output, total_elements, encoder);
        }

        Ok(attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_variant<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = B>>>(
        &self,
        variant: KernelVariant,
        kernel_key: &KernelKey,
        queries: &Allocation<B>,
        keys: &KVBuf,
        values: &KVBuf,
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
                    b: MatmulB::FullPrecision {
                        b: keys,
                    },
                    b_offset: group_index * k_head_stride as usize * dt_bytes,
                    b_leading_dimension: Some(k_seq_stride),
                    b_transpose: true,
                    d: &mut group_scores,
                    d_transform: MatmulDOps {
                        ab_scale: scale,
                        accumulate: false,
                        bias: None,
                        rht_factors: None,
                    },
                    m: (gqa_factor * suffix_length) as u32,
                    n: sequence_length as u32,
                    k: head_dim as u32,
                },
                encoder,
            )?;
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
                    b: MatmulB::FullPrecision {
                        b: values,
                    },
                    b_offset: group_index * v_head_stride as usize * dt_bytes,
                    b_leading_dimension: Some(v_seq_stride),
                    b_transpose: false,
                    d: &mut group_output,
                    d_transform: MatmulDOps::none(),
                    m: (gqa_factor * suffix_length) as u32,
                    n: head_dim as u32,
                    k: sequence_length as u32,
                },
                encoder,
            )?;
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
#[path = "../../../tests/unit/encodable_block/attention_test.rs"]
mod tests;
