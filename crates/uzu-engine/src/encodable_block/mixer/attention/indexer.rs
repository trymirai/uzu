use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        gpu_types::qkv_norm::QKVNormParams,
        kernel::{
            AttentionPrepareKernel, IndexerCacheUpdateKernel, attention::KeyBlocks, indexer::IndexerKernel,
            radix_top_k_small::MAX_K,
        },
    },
    config::{
        normalization::{NormalizationConfig, UpcastMode},
        rope::AnyRoPEConfig,
        token_mixer::sparse_attention::SparseAttentionIndexerConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::attention::{ATTENTION_SUFFIX_CAPACITY, AttentionNewError, rope::RoPE},
    },
    safetensors::ParameterTree,
};

pub struct Indexer<B: Backend> {
    pub config: SparseAttentionIndexerConfig,
    norm_params: QKVNormParams,
    query_norm_scales: Option<B::GlobalGpuBuffer>,
    key_norm_scales: Option<B::GlobalGpuBuffer>,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    kernel: <B::Kernels as Kernels>::IndexerKernel,
}

impl<B: Backend> Indexer<B> {
    pub async fn new(
        config: SparseAttentionIndexerConfig,
        rope_config: &AnyRoPEConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, AttentionNewError<B>> {
        if !<B::Kernels as Kernels>::IndexerKernel::is_supported(context) {
            return Err(AttentionNewError::IndexerNotSupported);
        }
        assert!(
            (1..=ATTENTION_SUFFIX_CAPACITY).contains(&config.compression_ratio)
                && config.token_budget.is_multiple_of(config.compression_ratio)
        );
        assert!(
            (1..=MAX_K).contains(&(config.token_budget / config.compression_ratio))
                && *rope_config.head_dim() <= config.head_dim
        );
        let mut norm_params = QKVNormParams::default();
        let mut load_norm = async |slot: usize, norm: &NormalizationConfig, name: &str| {
            assert!(!norm.subtract_mean && !norm.has_biases, "indexer requires RMS normalization");
            norm_params.enabled[slot] = 1;
            norm_params.full_layer[slot] = u32::from(norm.upcast_mode == UpcastMode::FullLayer);
            norm_params.epsilon[slot] = norm.epsilon;
            norm_params.scale_offset[slot] = norm.scale_offset.unwrap_or(0.0);
            Ok::<_, AttentionNewError<B>>(if norm.has_scale {
                Some(parameter_tree.subtree(name).read_buffer("scales", &[config.head_dim], DataType::F32).await?)
            } else {
                None
            })
        };
        let query_norm_scales = load_norm(0, &config.query_norm_config, "query_norm").await?;
        let key_norm_scales = load_norm(1, &config.key_norm_config, "key_norm").await?;
        Ok(Self {
            prepare: <B::Kernels as Kernels>::AttentionPrepareKernel::new(
                context,
                DataType::BF16,
                config.head_dim,
                true,
                1,
                0,
                true,
                query_norm_scales.is_some(),
                key_norm_scales.is_some(),
                false,
                true,
                config.compression_ratio,
                true,
            )
            .map_err(AttentionNewError::Backend)?,
            kernel: <B::Kernels as Kernels>::IndexerKernel::new(
                context,
                config.num_heads,
                config.head_dim,
                config.compression_ratio,
            )
            .map_err(AttentionNewError::Backend)?,
            config,
            norm_params,
            query_norm_scales,
            key_norm_scales,
        })
    }

    /// `projected` starts at the indexer Q/K heads.
    pub fn encode(
        &self,
        keys: &mut IndexerCache<B>,
        projected: impl BufferRef<Backend = B>,
        projected_row_stride: u32,
        rope: &RoPE<B>,
        topology: &BatchTopology<B>,
        prefix: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<KeyBlocks<B::ScratchBuffer>>, B::Error> {
        let block_budget = self.config.token_budget / self.config.compression_ratio;
        let suffix = topology.size();
        assert!(suffix <= ATTENTION_SUFFIX_CAPACITY);
        let requires_block_ranking = (prefix + suffix) / self.config.compression_ratio > block_budget;
        let query_heads = if requires_block_ranking {
            self.config.num_heads
        } else {
            0
        };
        let block_ids_bytes = if requires_block_ranking {
            (suffix * block_budget) as usize * size_of::<i32>()
        } else {
            0
        };
        let draft_rows_bytes = (suffix * self.config.compression_ratio) as usize * size_of::<i32>();
        let mut query_blocks = command_buffer.allocate_scratch(block_ids_bytes + draft_rows_bytes)?;
        let nodes = self.kernel.prepare_draft_blocks(
            topology.nodes(),
            suffix,
            prefix,
            (&mut query_blocks).subrange_mut(block_ids_bytes..),
            command_buffer,
        )?;
        let query_rows = suffix * query_heads;
        let mut query_codes =
            command_buffer.allocate_scratch((query_rows as usize * self.config.head_dim as usize).max(1))?;
        let mut query_scales =
            command_buffer.allocate_scratch((query_rows as usize * DataType::BF16.size_in_bytes()).max(1))?;
        let skipped_query_bytes =
            ((self.config.num_heads - query_heads) * self.config.head_dim) as usize * DataType::BF16.size_in_bytes();
        self.prepare.encode(
            projected.subrange(skipped_query_bytes..),
            &mut query_codes,
            &mut query_scales,
            Some(&mut keys.block_codes),
            Some(&mut keys.block_scales),
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            Some(&rope.inverse_frequencies),
            self.query_norm_scales.as_ref(),
            self.key_norm_scales.as_ref(),
            None::<&B::GlobalGpuBuffer>,
            Some(&nodes),
            Some((&query_blocks).subrange(block_ids_bytes..)),
            Some(&mut keys.unpooled_keys),
            self.norm_params,
            query_heads,
            Some(rope.dim),
            Some(rope.scale),
            Some(prefix),
            0,
            projected_row_stride,
            suffix,
            query_rows,
            Some(prefix),
            Some(&nodes),
            command_buffer,
        );
        if !requires_block_ranking {
            return Ok(None);
        }
        let candidates = if topology.nodes().is_none() {
            (prefix + suffix) / self.config.compression_ratio
        } else {
            prefix / self.config.compression_ratio + suffix
        };
        self.kernel.select_blocks(
            &query_codes,
            &query_scales,
            &keys.block_codes,
            &keys.block_scales,
            suffix,
            prefix,
            ATTENTION_SUFFIX_CAPACITY,
            candidates,
            KeyBlocks {
                query_blocks: &mut query_blocks,
                blocks_per_query: block_budget,
            },
            topology.nodes().is_some().then_some(&nodes),
            command_buffer,
        )?;
        Ok(Some(KeyBlocks {
            query_blocks,
            blocks_per_query: block_budget,
        }))
    }
}

pub struct IndexerCache<B: Backend> {
    unpooled_keys: B::GlobalGpuBuffer,
    block_codes: B::GlobalGpuBuffer,
    block_scales: B::GlobalGpuBuffer,
    accept: <B::Kernels as Kernels>::IndexerCacheUpdateKernel,
}

impl<B: Backend> IndexerCache<B> {
    pub fn new(
        config: &SparseAttentionIndexerConfig,
        max_elements: u32,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let head_dim = config.head_dim as usize;
        let unpooled_rows = (ATTENTION_SUFFIX_CAPACITY + config.compression_ratio - 1) as usize;
        let block_rows = (max_elements / config.compression_ratio + ATTENTION_SUFFIX_CAPACITY) as usize;
        Ok(Self {
            unpooled_keys: context.create_buffer_gpu(unpooled_rows * head_dim * DataType::BF16.size_in_bytes())?,
            block_codes: context.create_buffer_gpu(block_rows * head_dim)?,
            block_scales: context.create_buffer_gpu(block_rows * DataType::BF16.size_in_bytes())?,
            accept: <B::Kernels as Kernels>::IndexerCacheUpdateKernel::new(
                context,
                config.head_dim,
                config.compression_ratio,
            )?,
        })
    }

    pub fn encode_accept(
        &mut self,
        prefix: u32,
        accepted: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        if !accepted.is_empty() {
            self.accept.encode(
                &mut self.block_codes,
                &mut self.block_scales,
                &mut self.unpooled_keys,
                accepted,
                accepted.len() as u32,
                prefix,
                ATTENTION_SUFFIX_CAPACITY,
                command_buffer,
            );
        }
    }
}
