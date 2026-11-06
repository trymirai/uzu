use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::qkv_norm::QKVNormParams,
        kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, AttentionPrepareKernel},
    },
    config::{normalization::UpcastMode, rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::{Mixer, MixerEncodeError, MixerState, attention::rope::PrecalculatedRoPE},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

mod state;

pub use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState, KVCache, KVCacheView};

pub mod rope;

pub struct Attention<B: Backend> {
    head_dim: u32,
    num_q_heads: u32,
    num_groups: u32,
    num_kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    max_rope_length: Option<u32>,
    data_type: DataType,
    has_gate: bool,
    projection_dim: u32,
    projection: Box<dyn Linear<B>>,
    qkv_norm_params: QKVNormParams,
    qkv_norm_scales: [Option<B::GlobalGpuBuffer>; 3],
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

impl<B: Backend> Attention<B> {
    pub async fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalGpuBuffer>), AttentionNewError<B>> {
        let is_kv_sharing = config.is_kv_sharing;

        let head_dim = config.head_dim;
        let num_groups = config.num_groups;
        let num_q_heads = config.num_heads;
        let num_kv_heads = (!is_kv_sharing).then_some(num_groups);

        let is_causal = config.is_causal;
        let sliding_window_size = config.sliding_window_size;
        let max_rope_length = rope_config.map(|rope_config| *rope_config.max_sequence_length());

        let q_dim = num_q_heads * head_dim;

        let projection_tree = parameter_tree.subtree("qkvg_projection");
        let qkv_dim = if let Some(num_kv_heads) = num_kv_heads {
            let kv_dim = num_kv_heads * head_dim;
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
            &projection_tree,
            context,
        )
        .await?;

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
                has_sinks: sinks.is_some(),
                gate_row_stride: config.has_gate.then_some(projection_dim),
                is_kv_cache_ring,
                is_causal,
                sliding_window_size,
                scale: config.scale,
                data_type,
            },
        )
        .map_err(AttentionNewError::Kernel)?;
        let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            context,
            data_type,
            DataType::F32,
            head_dim,
            rope_config.is_some(),
            num_groups,
            kernel.kv_layout().backend_opaque_layout,
            !is_kv_sharing,
            qkv_norm_scales[0].is_some(),
            qkv_norm_scales[1].is_some(),
            qkv_norm_scales[2].is_some(),
            false,
        )
        .map_err(AttentionNewError::Backend)?;
        let prepare_trie = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            context,
            data_type,
            DataType::F32,
            head_dim,
            rope_config.is_some(),
            num_groups,
            kernel.kv_layout().backend_opaque_layout,
            !is_kv_sharing,
            qkv_norm_scales[0].is_some(),
            qkv_norm_scales[1].is_some(),
            qkv_norm_scales[2].is_some(),
            true,
        )
        .map_err(AttentionNewError::Backend)?;

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
                max_rope_length,
                data_type,
                has_gate: config.has_gate,
                projection_dim,
                projection,
                qkv_norm_params,
                qkv_norm_scales,
                prepare,
                prepare_trie,
                sinks,
                kernel,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let qkvg = self.projection.encode(hidden, batch_dim.size(), command_buffer)?;
        let gate = self.has_gate.then(|| {
            qkvg.subrange(
                (self.projection_dim - self.num_q_heads * self.head_dim) as usize * self.data_type.size_in_bytes()..,
            )
        });

        let attention_output = {
            let mut scratch_kv;
            let (mut kv, cache) = match state {
                Some(MaybeMut::Mut(state)) => {
                    let cache = state.view();
                    (MaybeMut::Mut(&mut state.kv), cache)
                },
                Some(MaybeMut::Const(state)) => (MaybeMut::Const(&state.kv), state.view()),
                None => {
                    assert!(self.num_kv_heads.is_some(), "stateless query-only attention");
                    assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");
                    scratch_kv = KVCache::allocate_scratch(self.kernel.kv_layout(), batch_dim.size(), command_buffer)
                        .map_err(AttentionEncodeError::Backend)?;
                    let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));
                    (MaybeMut::Mut(&mut scratch_kv), cache)
                },
            };
            let trie = batch_dim.nodes();
            let (queries, query_scales) = self
                .prepare(
                    &qkvg,
                    &mut kv,
                    cache.prefix_len(),
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    trie,
                    command_buffer,
                )
                .map_err(AttentionEncodeError::Backend)?;
            let input = AttentionInput {
                query_codes: &queries,
                query_scales: &query_scales,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            };
            self.kernel
                .encode(
                    input,
                    AttentionArguments {
                        suffix_length: batch_dim.size(),
                        trie,
                        sinks: self.sinks.as_ref(),
                        cache,
                        gate,
                    },
                    command_buffer,
                )
                .map_err(AttentionEncodeError::Kernel)?
        };

        Ok(self.out_projection.encode(attention_output, batch_dim.size(), command_buffer)?)
    }

    pub fn append_projected_kv(
        &self,
        key_value: impl BufferRef<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let prefix_len = state.view().prefix_len();
        self.prepare(
            key_value,
            &mut MaybeMut::Mut(&mut state.kv),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            None,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn prepare(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        kv: &mut MaybeMut<'_, KVCache<B>>,
        kv_token_offset: u32,
        num_q_heads: u32,
        rope: Option<&PrecalculatedRoPE<B>>,
        suffix_length: u32,
        trie: Option<&B::GlobalGpuBuffer>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        assert_eq!(kv.layout, self.kernel.kv_layout(), "shared KV layout mismatch");
        let (k_codes, k_scales, v_codes, v_scales) = match kv {
            MaybeMut::Mut(kv) => (
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
            ),
            MaybeMut::Const(_) => (None, None, None, None),
        };
        let input_row_stride = if num_q_heads == 0 {
            2 * self.num_groups * self.head_dim
        } else {
            self.projection_dim
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
            rope.map(|rope| &rope.cosines),
            rope.map(|rope| &rope.sines),
            self.qkv_norm_scales[0].as_ref(),
            self.qkv_norm_scales[1].as_ref(),
            self.qkv_norm_scales[2].as_ref(),
            self.qkv_norm_params,
            num_q_heads,
            rope.map(|rope| rope.dim),
            kv_token_offset,
            input_row_stride,
            suffix_length,
            num_q_heads / self.num_groups * suffix_length,
            trie,
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
        self.max_rope_length
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
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology<B>,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MixerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("attention");

        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));
        let output = self.attend(hidden, precalculated_rope, batch_dim, state, command_buffer)?;

        Ok(output)
    }
}
