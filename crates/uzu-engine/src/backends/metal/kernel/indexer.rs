use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            kernel::{attention::KeyBlocks, indexer::IndexerKernel},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::kv_layout,
            kernel::{AttentionBlockScoreMetalKernel, PrepareDraftBlocksMetalKernel, RadixTopKSmallMetalKernel},
        },
    },
    data_type::DataType,
};

pub struct IndexerMetalKernel {
    compression_ratio: u32,
    use_mxu: bool,
    prepare_draft: [PrepareDraftBlocksMetalKernel; 2],
    scorers: [AttentionBlockScoreMetalKernel; 2],
    top_k: RadixTopKSmallMetalKernel,
}

impl IndexerMetalKernel {
    pub fn new(
        context: &MetalContext,
        num_heads: u32,
        head_dim: u32,
        compression_ratio: u32,
        use_mxu: bool,
    ) -> Result<Self, MetalError> {
        let scorer = |is_trie| {
            AttentionBlockScoreMetalKernel::new(context, num_heads, head_dim, use_mxu, compression_ratio, is_trie)
        };
        Ok(Self {
            compression_ratio,
            use_mxu,
            prepare_draft: [
                PrepareDraftBlocksMetalKernel::new(context, false)?,
                PrepareDraftBlocksMetalKernel::new(context, true)?,
            ],
            scorers: [scorer(false)?, scorer(true)?],
            top_k: RadixTopKSmallMetalKernel::new(context, DataType::F16, false, false)?,
        })
    }
}

impl IndexerKernel for IndexerMetalKernel {
    type Backend = Metal;

    fn is_supported(_context: &MetalContext) -> bool {
        true
    }

    fn new(
        context: &MetalContext,
        num_heads: u32,
        head_dim: u32,
        compression_ratio: u32,
    ) -> Result<Self, MetalError> {
        Self::new(context, num_heads, head_dim, compression_ratio, context.supports_mxu)
    }

    fn prepare_draft_blocks(
        &self,
        trie: Option<impl BufferRef<Backend = Metal>>,
        rows: u32,
        prefix: u32,
        block_token_rows: impl BufferMut<Backend = Metal>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) {
        self.prepare_draft[usize::from(trie.is_some())].encode(
            trie,
            block_token_rows,
            rows,
            prefix,
            self.compression_ratio,
            command_buffer,
        );
    }

    fn select_blocks(
        &self,
        query_codes: impl BufferRef<Backend = Metal>,
        query_scales: impl BufferRef<Backend = Metal>,
        key_codes: impl BufferRef<Backend = Metal>,
        key_scales: impl BufferRef<Backend = Metal>,
        rows: u32,
        prefix_len: u32,
        cached_block_row: u32,
        candidates: u32,
        selected: KeyBlocks<impl BufferMut<Backend = Metal>>,
        trie: Option<impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let blocks = selected.blocks_per_query;
        assert!(rows > 0 && blocks > 0 && blocks <= candidates);
        assert!(blocks <= crate::backends::common::kernel::radix_top_k_small::MAX_K);
        let mut scores =
            command_buffer.allocate_scratch(rows as usize * candidates as usize * size_of::<half::f16>())?;
        let id_bytes = rows as usize * blocks as usize * size_of::<u32>();
        let (rows_per_group, blocks_per_group) = score_tiling(self.use_mxu, rows);
        self.scorers[usize::from(trie.is_some())].encode(
            query_codes,
            query_scales,
            key_codes,
            key_scales,
            &mut scores,
            0,
            rows,
            rows,
            prefix_len,
            cached_block_row,
            rows_per_group,
            blocks_per_group,
            candidates,
            trie,
            command_buffer,
        );
        self.top_k.encode(
            &scores,
            selected.query_blocks.subrange_mut(..id_bytes),
            None::<&mut <Metal as Backend>::ScratchBuffer>,
            rows,
            candidates,
            blocks,
            command_buffer,
        );
        Ok(())
    }
}

const SIMD_FRAGMENT_ROWS: u32 = 8;
const SCORE_MAX_QUERY_FRAGMENTS: u32 = 4;
const SCORE_BLOCKS_PER_GROUP: u32 = 128;

fn score_tiling(
    use_mxu: bool,
    rows: u32,
) -> (u32, u32) {
    let fragment_rows = if use_mxu {
        kv_layout::TILE_DIMS
    } else {
        SIMD_FRAGMENT_ROWS
    };
    let rows_per_group = rows.next_power_of_two().clamp(fragment_rows, SCORE_MAX_QUERY_FRAGMENTS * fragment_rows);
    (rows_per_group, SCORE_BLOCKS_PER_GROUP)
}
