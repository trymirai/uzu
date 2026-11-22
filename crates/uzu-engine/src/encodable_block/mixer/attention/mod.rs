use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, GpuBuffer, Kernels,
        gpu_types::{
            attention_prepare::{AttentionPrepareParams, SeparatePrepareParams},
            log_scaling::LogScalingParams,
            qkv_norm::QKVNormParams,
        },
        kernel::{
            AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, AttentionPrepareKernel,
            AttentionPrepareSeparateKernel, AttentionRelativeLogitsKernel, attention::KeyBlocks,
        },
    },
    config::{
        normalization::UpcastMode,
        rope::AnyRoPEConfig,
        token_mixer::{
            attention::{AttentionConfig, LogScalingConfig},
            convolutions::CausalConvConfig,
        },
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        block_attention::upload,
        causal_conv::{CausalConv, CausalConvNewError, CausalConvState},
        linear::{InputRhtParameters, Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerState, attention::rope::RoPE},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_mut::MaybeMut, size_for_shape::size_for_shape},
};

mod state;

pub use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState, KVCache, KVCacheState};

pub mod rope;

/// The relative logits kernel keeps one head's states in threadgroup memory.
const MAX_RELATIVE_DIM: u32 = 32;

/// The context that full attention without positional embeddings allocates its cache for.
const MAX_CONTEXT_LENGTH_WITHOUT_ROPE: u32 = 1 << 17;

/// The projection of attention inputs to queries, keys, values and their extras.
enum Projection<B: Backend> {
    /// One input projected to queries, keys, values and the gate, `dim` elements per row.
    Fused {
        projection: Box<dyn Linear<B>>,
        dim: u32,
    },
    /// Separate query, key and value inputs, see `AttentionConfig::separate_inputs`.
    Separate(SeparateProjections<B>),
}

struct SeparateProjections<B: Backend> {
    /// Per row: the queries, then the relative states, the gate and the token sinks.
    query: Box<dyn Linear<B>>,
    query_dim: u32,
    /// Per row: the keys, then the key convolution's dynamic taps.
    key: Box<dyn Linear<B>>,
    key_dim: u32,
    /// Per row: the values, then the value convolution's dynamic taps.
    value: Box<dyn Linear<B>>,
    value_dim: u32,
    key_conv: Option<CausalConv<B>>,
    value_conv: Option<CausalConv<B>>,
}

/// Logit biases by query-to-key distance, see `RelativePositionBiasConfig`.
struct RelativeBias<B: Backend> {
    /// The element offset of the relative states in a query projection row.
    offset: u32,
    extent: u32,
    clip_threshold: f32,
    clip_scales: B::GlobalGpuBuffer,
    profiles: B::GlobalGpuBuffer,
    flat_kernel: <B::Kernels as Kernels>::AttentionRelativeLogitsKernel,
    trie_kernel: <B::Kernels as Kernels>::AttentionRelativeLogitsKernel,
}

/// The one-dispatch preparation of separate-input attention, see `AttentionPrepareSeparate`: for a flat batch of one
/// token, which shifts the convolution states in place, a longer flat batch and a trie.
struct FusedPrepare<B: Backend> {
    decode_kernel: <B::Kernels as Kernels>::AttentionPrepareSeparateKernel,
    flat_kernel: <B::Kernels as Kernels>::AttentionPrepareSeparateKernel,
    trie_kernel: <B::Kernels as Kernels>::AttentionPrepareSeparateKernel,
    /// The relative clip scales `[q head][dim]`, then the profiles `[dim][extent]`.
    relative_params: Option<B::GlobalGpuBuffer>,
    kernel_size: u32,
    value_norm_epsilon: f32,
}

pub struct Attention<B: Backend> {
    head_dim: u32,
    num_q_heads: u32,
    num_groups: u32,
    num_kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    rope: Option<RoPE<B>>,
    data_type: DataType,
    hidden_dim: u32,
    projection: Projection<B>,
    /// The element offsets of the gate and of the token sinks in a projection row.
    gate_offset: Option<u32>,
    token_sink_offset: Option<u32>,
    qkv_norm_params: QKVNormParams,
    qkv_norm_scales: [Option<B::GlobalGpuBuffer>; 3],
    query_clip_threshold: Option<f32>,
    query_clip_scales: Option<B::GlobalGpuBuffer>,
    key_clip_threshold: Option<f32>,
    log_scaling: Option<LogScalingConfig>,
    relative_bias: Option<RelativeBias<B>>,
    fused_prepare: Option<FusedPrepare<B>>,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    prepare_trie: <B::Kernels as Kernels>::AttentionPrepareKernel,
    sinks: Option<B::GlobalGpuBuffer>,
    kernel: <B::Kernels as Kernels>::AttentionKernel,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum AttentionNewError<B: Backend> {
    #[error("Attention kernel error: {0}")]
    Kernel(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Causal convolution error: {0}")]
    CausalConv(#[from] CausalConvNewError<B>),
    #[error("Unsupported attention configuration: {0}")]
    Unsupported(&'static str),
}

#[derive(Debug, Error)]
pub enum AttentionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("Attention kernel error: {0}")]
    Kernel(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
}

fn dynamic_taps(config: Option<&CausalConvConfig>) -> u32 {
    config.and_then(|config| config.num_dynamic_groups.map(|groups| groups * config.kernel_size)).unwrap_or(0)
}

impl<B: Backend> Attention<B> {
    pub async fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<InputRhtParameters<B>>), AttentionNewError<B>> {
        let is_kv_sharing = config.is_kv_sharing;

        let head_dim = config.head_dim;
        let num_groups = config.num_groups;
        let num_q_heads = config.num_heads;
        let num_kv_heads = (!is_kv_sharing).then_some(num_groups);

        let is_causal = config.is_causal;
        let sliding_window_size = config.sliding_window_size;
        let rope = rope_config
            .map(|rope_config| RoPE::new(rope_config, context))
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_groups * head_dim;

        let relative_config = config.relative_position_bias_config.as_ref();
        let (projection, row_dim, gate_offset, token_sink_offset, in_projection_input_hadamard_factors) =
            if config.separate_inputs {
                if is_kv_sharing {
                    return Err(AttentionNewError::Unsupported("separate inputs cannot share another layer's cache"));
                }
                let relative_offset = q_dim;
                let gate_offset = relative_offset + relative_config.map_or(0, |config| num_q_heads * config.dim);
                let token_sink_offset = gate_offset + if config.has_gate {
                    q_dim
                } else {
                    0
                };
                let query_dim = token_sink_offset + if config.has_token_sinks {
                    num_q_heads
                } else {
                    0
                };
                let key_dim = kv_dim + dynamic_taps(config.key_conv_config.as_ref());
                let value_dim = kv_dim + dynamic_taps(config.value_conv_config.as_ref());
                let mut projections = Vec::with_capacity(3);
                for (name, output_dim) in
                    [("query_projection", query_dim), ("key_projection", key_dim), ("value_projection", value_dim)]
                {
                    let tree = parameter_tree.subtree(name);
                    projections.push(
                        <dyn Linear<B>>::new(hidden_dim, output_dim, data_type, config.has_qkvg_biases, &tree, context)
                            .await?,
                    );
                }
                let [query, key, value] = <[_; 3]>::try_from(projections).ok().unwrap();
                let mut convs = Vec::with_capacity(2);
                for (conv_config, name) in
                    [(config.key_conv_config.as_ref(), "key_conv"), (config.value_conv_config.as_ref(), "value_conv")]
                {
                    convs.push(match conv_config {
                        Some(conv_config) => Some(
                            CausalConv::new(kv_dim, data_type, conv_config, &parameter_tree.subtree(name), context)
                                .await?,
                        ),
                        None => None,
                    });
                }
                let value_conv = convs.pop().unwrap();
                let key_conv = convs.pop().unwrap();
                (
                    Projection::Separate(SeparateProjections {
                        query,
                        query_dim,
                        key,
                        key_dim,
                        value,
                        value_dim,
                        key_conv,
                        value_conv,
                    }),
                    query_dim,
                    config.has_gate.then_some(gate_offset),
                    config.has_token_sinks.then_some(token_sink_offset),
                    None,
                )
            } else {
                let qkv_dim = if num_kv_heads.is_some() {
                    q_dim + kv_dim + kv_dim
                } else {
                    q_dim
                };
                let projection_dim = if config.has_gate {
                    qkv_dim + q_dim
                } else {
                    qkv_dim
                };
                let (projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
                    hidden_dim,
                    projection_dim,
                    data_type,
                    config.has_qkvg_biases,
                    &parameter_tree.subtree("qkvg_projection"),
                    context,
                )
                .await?;
                (
                    Projection::Fused {
                        projection,
                        dim: projection_dim,
                    },
                    projection_dim,
                    config.has_gate.then_some(qkv_dim),
                    None,
                    in_projection_input_hadamard_factors,
                )
            };

        let query_norm_config = config.query_norm_config.clone();
        // TODO: Fix lalamo config, those two must be None if kv sharing.
        let key_norm_config = (!is_kv_sharing).then(|| config.key_norm_config.clone()).flatten();
        let value_norm_config = (!is_kv_sharing).then(|| config.value_norm_config()).flatten();
        let mut qkv_norm_params = QKVNormParams::default();
        let mut qkv_norm_scales = [None, None, None];
        for (i, (config, name)) in
            [(query_norm_config, "query_norm"), (key_norm_config, "key_norm"), (value_norm_config, "value_norm")]
                .into_iter()
                .enumerate()
        {
            let Some(config) = config else {
                continue;
            };
            qkv_norm_params.enabled[i] = 1;
            qkv_norm_params.full_layer[i] = u32::from(config.upcast_mode == UpcastMode::FullLayer);
            qkv_norm_params.epsilon[i] = config.epsilon;
            qkv_norm_params.scale_offset[i] = config.scale_offset.unwrap_or(0.0);
            if config.has_scale {
                qkv_norm_scales[i] =
                    Some(parameter_tree.subtree(name).read_buffer("scales", &[head_dim], DataType::F32).await?);
            }
        }

        let query_clip_scales = match &config.query_clip_config {
            Some(clip_config) if clip_config.has_scale => {
                Some(parameter_tree.read_buffer("query_clip_scale", &[num_q_heads, head_dim], DataType::F32).await?)
            },
            _ => None,
        };
        if config.key_clip_config.as_ref().is_some_and(|clip_config| clip_config.has_scale) {
            return Err(AttentionNewError::Unsupported("scaled key clipping"));
        }

        let relative_bias = match relative_config {
            Some(relative_config) => {
                if !relative_config.clip_config.has_scale {
                    return Err(AttentionNewError::Unsupported("relative states without clip scales"));
                }
                if relative_config.dim > MAX_RELATIVE_DIM {
                    return Err(AttentionNewError::Unsupported("relative states wider than 32 per head"));
                }
                let new_kernel = |has_trie| {
                    <B::Kernels as Kernels>::AttentionRelativeLogitsKernel::new(
                        context,
                        data_type,
                        num_q_heads,
                        relative_config.dim,
                        relative_config.extent,
                        config.log_scaling_config.is_some(),
                        has_trie,
                    )
                    .map_err(AttentionNewError::Backend)
                };
                Some(RelativeBias {
                    offset: q_dim,
                    extent: relative_config.extent,
                    clip_threshold: relative_config.clip_config.threshold,
                    clip_scales: parameter_tree
                        .read_buffer("relative_clip_scale", &[num_q_heads, relative_config.dim], DataType::F32)
                        .await?,
                    profiles: parameter_tree
                        .read_buffer(
                            "relative_position_profiles",
                            &[relative_config.dim, relative_config.extent],
                            DataType::F32,
                        )
                        .await?,
                    flat_kernel: new_kernel(false)?,
                    trie_kernel: new_kernel(true)?,
                })
            },
            None => None,
        };

        let sinks = if config.has_sinks {
            Some(parameter_tree.read_buffer("sinks", &[num_q_heads], data_type).await?)
        } else {
            None
        };

        assert!(sliding_window_size.is_none_or(|size| size > 0), "zero sliding window size");
        let ring_capacity = sliding_window_size;
        let is_kv_cache_ring = ring_capacity.is_some();

        let kernel = <B::Kernels as Kernels>::AttentionKernel::new(
            context,
            AttentionKernelConfig {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some() || token_sink_offset.is_some(),
                sink_row_stride: if token_sink_offset.is_some() {
                    row_dim
                } else {
                    0
                },
                gate_row_stride: gate_offset.map(|_| row_dim),
                is_kv_cache_ring,
                is_causal,
                sliding_window_size,
                scale: config.scale,
                data_type,
                sparse_compression_ratio: None,
                relative_extent: relative_bias.as_ref().map(|relative_bias| relative_bias.extent),
            },
        )
        .map_err(AttentionNewError::Kernel)?;
        let new_prepare = |has_trie| {
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(
                context,
                data_type,
                head_dim,
                rope_config.is_some(),
                num_groups,
                kernel.kv_layout().backend_opaque_layout,
                !is_kv_sharing,
                qkv_norm_scales[0].is_some(),
                qkv_norm_scales[1].is_some(),
                qkv_norm_scales[2].is_some(),
                false,
                0,
                has_trie,
                config.separate_inputs,
                config.query_clip_config.is_some(),
                query_clip_scales.is_some(),
                config.key_clip_config.is_some(),
                config.log_scaling_config.is_some(),
            )
            .map_err(AttentionNewError::Backend)
        };
        let prepare = new_prepare(false)?;
        let prepare_trie = new_prepare(true)?;

        let value_norm = config.value_norm_config();
        let fused_prepare = match &projection {
            Projection::Separate(projections)
                if rope_config.is_none()
                    && config.query_norm_config.is_none()
                    && config.key_norm_config.is_none()
                    && value_norm.as_ref().is_none_or(|norm| !norm.has_scale && !norm.subtract_mean)
                    && matches!(head_dim, 64 | 128 | 256) =>
            {
                let convs = [projections.key_conv.as_ref(), projections.value_conv.as_ref()];
                let present = convs.iter().flatten().collect::<Vec<_>>();
                let kernel_size = present.first().map_or(2, |conv| conv.kernel_size());
                let has_taps = present.first().is_some_and(|conv| conv.has_taps());
                let has_identity = present.first().is_some_and(|conv| conv.has_identity());
                // Dynamic taps are shared by the channels of one head. The relative states sit one per lane, and the
                // relative logits are written 4 distances at a time.
                let fusable = present.iter().all(|conv| {
                    conv.kernel_size() == kernel_size
                        && conv.has_taps() == has_taps
                        && conv.has_identity() == has_identity
                        && (!has_taps || conv.group_size() == head_dim)
                }) && relative_config
                    .is_none_or(|relative| relative.dim <= 32 && relative.extent.is_multiple_of(4));
                if fusable {
                    let relative_params = match relative_config {
                        Some(relative_config) => {
                            let mut params = parameter_tree
                                .read_slice::<f32>("relative_clip_scale", &[num_q_heads, relative_config.dim])
                                .await?
                                .into_vec();
                            params.extend_from_slice(
                                &parameter_tree
                                    .read_slice::<f32>(
                                        "relative_position_profiles",
                                        &[relative_config.dim, relative_config.extent],
                                    )
                                    .await?,
                            );
                            Some(upload::<B, _>(&params, context).map_err(AttentionNewError::Backend)?)
                        },
                        None => None,
                    };
                    let new_kernel = |updates_conv_state: bool, has_trie: bool| {
                        <B::Kernels as Kernels>::AttentionPrepareSeparateKernel::new(
                            context,
                            data_type,
                            head_dim,
                            num_q_heads,
                            num_groups,
                            kernel.kv_layout().backend_opaque_layout,
                            kernel_size,
                            projections.key_conv.is_some(),
                            projections.value_conv.is_some(),
                            has_taps,
                            has_identity,
                            updates_conv_state,
                            config.query_clip_config.is_some(),
                            query_clip_scales.is_some(),
                            config.key_clip_config.is_some(),
                            value_norm.is_some(),
                            config.log_scaling_config.is_some(),
                            relative_config.is_some(),
                            relative_config.map_or(0, |relative_config| relative_config.dim),
                            relative_config.map_or(0, |relative_config| relative_config.extent),
                            has_trie,
                        )
                        .map_err(AttentionNewError::Backend)
                    };
                    Some(FusedPrepare {
                        decode_kernel: new_kernel(true, false)?,
                        flat_kernel: new_kernel(false, false)?,
                        trie_kernel: new_kernel(false, true)?,
                        relative_params,
                        kernel_size,
                        value_norm_epsilon: value_norm.as_ref().map_or(0.0, |norm| norm.epsilon),
                    })
                } else {
                    None
                }
            },
            _ => None,
        };

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            hidden_dim,
            data_type,
            config.has_out_biases,
            &parameter_tree.subtree("out_projection"),
            context,
        )
        .await?;

        Ok((
            Self {
                head_dim,
                num_q_heads,
                num_groups,
                num_kv_heads,
                ring_capacity,
                rope,
                data_type,
                hidden_dim,
                projection,
                gate_offset,
                token_sink_offset,
                qkv_norm_params,
                qkv_norm_scales,
                query_clip_threshold: config.query_clip_config.as_ref().map(|clip_config| clip_config.threshold),
                query_clip_scales,
                key_clip_threshold: config.key_clip_config.as_ref().map(|clip_config| clip_config.threshold),
                log_scaling: config.log_scaling_config.clone(),
                relative_bias,
                fused_prepare,
                prepare,
                prepare_trie,
                sinks,
                kernel,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    /// The states of the key and value convolutions, when there are any.
    pub(super) fn create_conv_states(
        &self,
        context: &B::Context,
    ) -> Result<[Option<CausalConvState<B>>; 2], B::Error> {
        let Projection::Separate(projections) = &self.projection else {
            return Ok([None, None]);
        };
        let create = |conv: &Option<CausalConv<B>>| conv.as_ref().map(|conv| conv.create_state(context)).transpose();
        Ok([create(&projections.key_conv)?, create(&projections.value_conv)?])
    }

    fn element_offset(
        &self,
        elements: u32,
    ) -> usize {
        elements as usize * self.data_type.size_in_bytes()
    }

    fn log_scaling_params(
        &self,
        cache: KVCacheState,
    ) -> Option<LogScalingParams> {
        self.log_scaling.as_ref().map(|log_scaling| LogScalingParams {
            n_floor: log_scaling.n_floor as f32,
            alpha: log_scaling.alpha,
            position_offset: cache.context_length(),
        })
    }

    fn attend(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        mut state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let suffix_length = batch_dim.size();
        match &self.projection {
            Projection::Fused {
                projection,
                dim,
            } => {
                let qkvg = projection.encode(hidden, suffix_length, command_buffer)?;
                self.attend_projected(&qkvg, *dim, None, batch_dim, state, command_buffer)
            },
            Projection::Separate(projections) => {
                let (queries, keys, values) = {
                    let command_buffer = &mut *command_buffer.debug_group("attention projections");
                    // The query, key and value inputs, or one input they all share.
                    let input_size = size_for_shape(&[suffix_length, self.hidden_dim], self.data_type);
                    let input_stride = if hidden.size() == input_size {
                        0
                    } else {
                        assert_eq!(hidden.size(), 3 * input_size, "attention expects one or three inputs");
                        input_size
                    };
                    let queries = projections.query.encode_borrowed(&hidden, 0, suffix_length, command_buffer)?;
                    let keys = projections.key.encode_borrowed(&hidden, input_stride, suffix_length, command_buffer)?;
                    let values =
                        projections.value.encode_borrowed(&hidden, 2 * input_stride, suffix_length, command_buffer)?;
                    (queries, keys, values)
                };
                drop(hidden);
                if let Some(fused) = &self.fused_prepare {
                    return self.attend_fused(
                        fused,
                        projections,
                        &queries,
                        &keys,
                        &values,
                        batch_dim,
                        state,
                        command_buffer,
                    );
                }

                let kv_dim = self.num_groups * self.head_dim;
                let [key_state, value_state] = match &mut state {
                    Some(MaybeMut::Mut(state)) => {
                        let [key_state, value_state] = &mut state.conv_states;
                        [key_state.as_mut(), value_state.as_mut()]
                    },
                    Some(MaybeMut::Const(_)) => {
                        assert!(
                            projections.key_conv.is_none() && projections.value_conv.is_none(),
                            "attention convolutions need a mutable state"
                        );
                        [None, None]
                    },
                    None => [None, None],
                };
                let convolve = |conv: &Option<CausalConv<B>>,
                                projected: B::ScratchBuffer,
                                dim: u32,
                                conv_state: Option<&mut CausalConvState<B>>,
                                command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
                 -> Result<(B::ScratchBuffer, u32), AttentionEncodeError<B>> {
                    let Some(conv) = conv else {
                        return Ok((projected, dim));
                    };
                    let taps = (dim > kv_dim).then(|| (projected.subrange(self.element_offset(kv_dim)..), dim));
                    let output = conv
                        .encode(&projected, dim, taps, batch_dim, conv_state, command_buffer)
                        .map_err(AttentionEncodeError::Backend)?;
                    Ok((output, kv_dim))
                };
                let (keys, values) = {
                    let command_buffer = &mut *command_buffer.debug_group("attention convolutions");
                    let keys = convolve(&projections.key_conv, keys, projections.key_dim, key_state, command_buffer)?;
                    let values =
                        convolve(&projections.value_conv, values, projections.value_dim, value_state, command_buffer)?;
                    (keys, values)
                };
                self.attend_projected(
                    &queries,
                    projections.query_dim,
                    Some((&keys.0, keys.1, &values.0, values.1)),
                    batch_dim,
                    state,
                    command_buffer,
                )
            },
        }
    }

    /// Attends with projected rows: `projected` holds `row_dim` elements per token, either the fused queries, keys,
    /// values and gate, or with `key_values` the separate queries and extras, next to keys and values with their
    /// row strides.
    fn attend_projected(
        &self,
        projected: &B::ScratchBuffer,
        row_dim: u32,
        key_values: Option<(&B::ScratchBuffer, u32, &B::ScratchBuffer, u32)>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let suffix_length = batch_dim.size();
        {
            let mut scratch_kv;
            let (mut kv, cache) = match state {
                Some(MaybeMut::Mut(state)) => {
                    let cache = state.cache();
                    (MaybeMut::Mut(&mut state.kv), cache)
                },
                Some(MaybeMut::Const(state)) => (MaybeMut::Const(&state.kv), state.cache()),
                None => {
                    assert!(self.num_kv_heads.is_some(), "stateless query-only attention");
                    assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");
                    scratch_kv = KVCache::allocate_scratch(self.kernel.kv_layout(), batch_dim.size(), command_buffer)
                        .map_err(AttentionEncodeError::Backend)?;
                    (MaybeMut::Mut(&mut scratch_kv), KVCacheState::new(0, self.ring_capacity))
                },
            };
            let trie = batch_dim.nodes();
            let (queries, query_scales, relative_logits) = {
            let command_buffer = &mut *command_buffer.debug_group("attention prepare");
            let (queries, query_scales) = self
                .prepare(
                    projected,
                    row_dim,
                    key_values,
                    &mut kv,
                    cache,
                    self.num_q_heads,
                    suffix_length,
                    trie,
                    None,
                    command_buffer,
                )
                .map_err(AttentionEncodeError::Backend)?;
            let relative_logits = self
                .relative_bias
                .as_ref()
                .map(|relative_bias| {
                    let mut logits = command_buffer.allocate_scratch_for_shape(
                        &[self.num_q_heads, suffix_length, relative_bias.extent],
                        DataType::BF16,
                    )?;
                    let kernel = if trie.is_some() {
                        &relative_bias.trie_kernel
                    } else {
                        &relative_bias.flat_kernel
                    };
                    kernel.encode(
                        projected.subrange(self.element_offset(relative_bias.offset)..),
                        &relative_bias.clip_scales,
                        &relative_bias.profiles,
                        &mut logits,
                        trie,
                        suffix_length,
                        row_dim,
                        relative_bias.clip_threshold,
                        self.log_scaling_params(cache),
                        command_buffer,
                    );
                    Ok(logits)
                })
                .transpose()
                .map_err(AttentionEncodeError::Backend)?;
            (queries, query_scales, relative_logits)
            };
            return self.attend_core(
                projected,
                &queries,
                &query_scales,
                &kv,
                cache,
                relative_logits.as_ref(),
                batch_dim,
                command_buffer,
            );
        };
    }

    /// Attends with separate projections prepared in one dispatch, see `FusedPrepare`.
    fn attend_fused(
        &self,
        fused: &FusedPrepare<B>,
        projections: &SeparateProjections<B>,
        queries: &B::ScratchBuffer,
        keys: &B::ScratchBuffer,
        values: &B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let Some(MaybeMut::Mut(state)) = state else {
            panic!("attention with separate inputs needs a mutable state");
        };
        let backend = AttentionEncodeError::Backend;
        let suffix_length = batch_dim.size();
        let trie = batch_dim.nodes();
        let cache = state.cache();
        let kv_dim = self.num_groups * self.head_dim;
        let history = fused.kernel_size - 1;
        let (relative_logits, query_codes, query_scales) = {
            let command_buffer = &mut *command_buffer.debug_group("attention prepare");
            let kernel = if trie.is_some() {
                &fused.trie_kernel
            } else if suffix_length == 1 {
                &fused.decode_kernel
            } else {
                &fused.flat_kernel
            };
            let mut node_inputs = |has_conv: bool| {
                (trie.is_some() && has_conv)
                    .then(|| command_buffer.allocate_scratch_for_shape(&[suffix_length, kv_dim, history], DataType::F32))
                    .transpose()
            };
            let mut key_nodes = node_inputs(projections.key_conv.is_some()).map_err(backend)?;
            let mut value_nodes = node_inputs(projections.value_conv.is_some()).map_err(backend)?;
            let query_count = (self.num_q_heads * suffix_length) as usize;
            let mut query_codes =
                command_buffer.allocate_scratch(query_count * self.head_dim as usize).map_err(backend)?;
            let mut query_scales =
                command_buffer.allocate_scratch(query_count * DataType::BF16.size_in_bytes()).map_err(backend)?;
            let mut relative_logits = self
                .relative_bias
                .as_ref()
                .map(|relative_bias| {
                    command_buffer.allocate_scratch_for_shape(
                        &[self.num_q_heads, suffix_length, relative_bias.extent],
                        DataType::BF16,
                    )
                })
                .transpose()
                .map_err(backend)?;
            let log_scaling = self.log_scaling_params(cache).unwrap_or_default();
            let params = SeparatePrepareParams {
                query_row_stride: projections.query_dim,
                key_row_stride: projections.key_dim,
                value_row_stride: projections.value_dim,
                relative_offset: self.relative_bias.as_ref().map_or(0, |relative_bias| relative_bias.offset),
                kv_token_offset: cache.prefix_len(),
                query_rows: self.num_q_heads / self.num_groups * suffix_length,
                position_offset: log_scaling.position_offset,
                query_clip_threshold: self.query_clip_threshold.unwrap_or(0.0),
                key_clip_threshold: self.key_clip_threshold.unwrap_or(0.0),
                relative_clip_threshold: self
                    .relative_bias
                    .as_ref()
                    .map_or(0.0, |relative_bias| relative_bias.clip_threshold),
                log_scaling_n_floor: log_scaling.n_floor,
                log_scaling_alpha: log_scaling.alpha,
                key_conv_scale: projections.key_conv.as_ref().map_or(1.0, |conv| conv.output_scale()),
                value_conv_scale: projections.value_conv.as_ref().map_or(1.0, |conv| conv.output_scale()),
                value_norm_epsilon: fused.value_norm_epsilon,
            };
            let AttentionState {
                kv,
                conv_states: [key_state, value_state],
                ..
            } = &mut *state;
            let mut key_inputs = key_state.as_mut().map(|state| state.begin_external(command_buffer));
            let mut value_inputs = value_state.as_mut().map(|state| state.begin_external(command_buffer));
            kernel.encode(
                queries,
                keys,
                values,
                projections.key_conv.as_ref().map(|conv| conv.weights()),
                projections.value_conv.as_ref().map(|conv| conv.weights()),
                key_inputs.as_deref_mut(),
                value_inputs.as_deref_mut(),
                key_nodes.as_mut(),
                value_nodes.as_mut(),
                self.query_clip_scales.as_ref(),
                fused.relative_params.as_ref(),
                relative_logits.as_mut(),
                &mut query_codes,
                &mut query_scales,
                kv.k_codes.as_mut(),
                kv.k_scales.as_mut(),
                kv.v_codes.as_mut(),
                kv.v_scales.as_mut(),
                trie,
                params,
                suffix_length,
                command_buffer,
            );
            // A longer flat batch moves its last inputs into the states once every token read them.
            if trie.is_none() && suffix_length > 1 {
                for (conv, inputs, rows, stride) in [
                    (&projections.key_conv, key_inputs.as_deref_mut(), keys, projections.key_dim),
                    (&projections.value_conv, value_inputs.as_deref_mut(), values, projections.value_dim),
                ] {
                    if let (Some(conv), Some(inputs)) = (conv, inputs) {
                        conv.encode_shift(rows, stride, suffix_length, inputs, command_buffer);
                    }
                }
            }
            if let Some(state) = key_state.as_mut() {
                state.finish_external(key_nodes);
            }
            if let Some(state) = value_state.as_mut() {
                state.finish_external(value_nodes);
            }
            (relative_logits, query_codes, query_scales)
        };
        self.attend_core(
            queries,
            &query_codes,
            &query_scales,
            &state.kv,
            cache,
            relative_logits.as_ref(),
            batch_dim,
            command_buffer,
        )
    }

    /// Attends with prepared queries and the KV cache, then projects the output. `projected` holds the gate and
    /// token sinks of each row.
    fn attend_core(
        &self,
        projected: &B::ScratchBuffer,
        queries: &B::ScratchBuffer,
        query_scales: &B::ScratchBuffer,
        kv: &KVCache<B>,
        cache: KVCacheState,
        relative_logits: Option<&B::ScratchBuffer>,
        batch_dim: &BatchTopology<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let suffix_length = batch_dim.size();
        let gate = self.gate_offset.map(|offset| projected.subrange(self.element_offset(offset)..));
        let projected_rows: &dyn GpuBuffer<Backend = B> = projected;
        let sinks = match (self.token_sink_offset, &self.sinks) {
            (Some(offset), _) => Some(projected_rows.subrange(self.element_offset(offset)..)),
            (None, Some(sinks)) => Some((sinks as &dyn GpuBuffer<Backend = B>).subrange(..)),
            (None, None) => None,
        };
        let input = AttentionInput {
            query_codes: queries,
            query_scales,
            k_codes: kv.k_codes.as_ref(),
            k_scales: kv.k_scales.as_ref(),
            v_codes: kv.v_codes.as_ref(),
            v_scales: kv.v_scales.as_ref(),
        };
        let attention_output = self
            .kernel
            .encode(
                input,
                AttentionArguments {
                    suffix_length,
                    trie: batch_dim.nodes(),
                    key_blocks: None::<KeyBlocks<&B::ConstantBuffer>>,
                    sinks,
                    cache,
                    gate,
                    relative_logits,
                },
                command_buffer,
            )
            .map_err(AttentionEncodeError::Kernel)?;
        Ok(self.out_projection.encode(attention_output, suffix_length, command_buffer)?)
    }

    /// Commits rows of keys, then values, to `state`, normalizing the keys with `key_norm_scales` instead of the
    /// attention's own key norm when given.
    pub fn append_projected_kv(
        &self,
        key_value: impl BufferRef<Backend = B>,
        batch_dim: u32,
        key_norm_scales: Option<&B::GlobalGpuBuffer>,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        assert!(matches!(self.projection, Projection::Fused { .. }), "appending projected keys to separate inputs");
        assert!(
            key_norm_scales.is_none() || self.qkv_norm_scales[1].is_some(),
            "a key norm override needs an attention with a scaled key norm"
        );
        let cache = state.cache();
        self.prepare(
            key_value,
            2 * self.num_groups * self.head_dim,
            None,
            &mut MaybeMut::Mut(&mut state.kv),
            cache,
            0,
            batch_dim,
            None,
            key_norm_scales,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn prepare(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        input_row_stride: u32,
        key_values: Option<(&B::ScratchBuffer, u32, &B::ScratchBuffer, u32)>,
        kv: &mut MaybeMut<'_, KVCache<B>>,
        cache: KVCacheState,
        num_q_heads: u32,
        suffix_length: u32,
        trie: Option<&B::GlobalGpuBuffer>,
        key_norm_scales: Option<&B::GlobalGpuBuffer>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        assert_eq!(kv.layout, self.kernel.kv_layout(), "shared KV layout mismatch");
        assert_eq!(key_values.is_some(), matches!(self.projection, Projection::Separate(_)), "separate key inputs");
        let rope = self.rope.as_ref();
        let (k_codes, k_scales, v_codes, v_scales) = match kv {
            MaybeMut::Mut(kv) => (
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
            ),
            MaybeMut::Const(_) => (None, None, None, None),
        };
        let query_count = num_q_heads as usize * suffix_length as usize;
        // K/V-only calls still need bound query buffers.
        let mut query_codes = command_buffer.allocate_scratch((query_count * self.head_dim as usize).max(1))?;
        let mut query_scales = command_buffer.allocate_scratch(query_count.max(1) * DataType::BF16.size_in_bytes())?;
        let prepare = if trie.is_some() {
            &self.prepare_trie
        } else {
            &self.prepare
        };
        prepare.encode(
            qkvg,
            &mut query_codes,
            &mut query_scales,
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            rope.map(|rope| &rope.inverse_frequencies),
            self.qkv_norm_scales[0].as_ref(),
            key_norm_scales.or(self.qkv_norm_scales[1].as_ref()),
            self.qkv_norm_scales[2].as_ref(),
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&mut B::GlobalGpuBuffer>,
            self.qkv_norm_params,
            num_q_heads,
            rope.map(|rope| rope.dim),
            rope.map(|rope| rope.scale),
            rope.map(|_| cache.context_length()),
            cache.prefix_len(),
            input_row_stride,
            suffix_length,
            num_q_heads / self.num_groups * suffix_length,
            None,
            trie,
            key_values.map(|(keys, ..)| keys),
            key_values.map(|(_, _, values, _)| values),
            self.query_clip_scales.as_ref(),
            (key_values.is_some()
                || self.query_clip_threshold.is_some()
                || self.key_clip_threshold.is_some()
                || self.log_scaling.is_some())
            .then(|| {
                let log_scaling = self.log_scaling_params(cache).unwrap_or_default();
                AttentionPrepareParams {
                    key_row_stride: key_values.map_or(0, |(_, key_row_stride, ..)| key_row_stride),
                    value_row_stride: key_values.map_or(0, |(.., value_row_stride)| value_row_stride),
                    query_clip_threshold: self.query_clip_threshold.unwrap_or(0.0),
                    key_clip_threshold: self.key_clip_threshold.unwrap_or(0.0),
                    log_scaling_n_floor: log_scaling.n_floor,
                    log_scaling_alpha: log_scaling.alpha,
                    position_offset: log_scaling.position_offset,
                }
            }),
            command_buffer,
        );
        Ok((query_codes, query_scales))
    }
}

impl<B: Backend> Mixer<B> for Attention<B> {
    fn speculation_supported(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<u32> {
        match &self.rope {
            Some(rope) => Some(rope.max_sequence_length),
            None => self.ring_capacity.is_none().then_some(MAX_CONTEXT_LENGTH_WITHOUT_ROPE),
        }
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        Ok(Box::new(AttentionState::create_empty(self, max_context_length, context)?))
    }

    fn encode(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MixerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("attention");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));
        let output = self.attend(hidden, batch_dim, state, command_buffer)?;

        Ok(output)
    }
}
