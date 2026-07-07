use super::AttentionGemmMetalKernel;
use crate::{
    backends::{
        common::{
            Encoder,
            gpu_types::{attention::AttnParams, ring::RingParams},
            kernel::{
                AttentionGemmKernel, BufferArg, BufferArgMut,
                attention_gemm::{AttentionGemmDispatch, retile_params, tile_variant_index},
            },
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct AttentionGemmMetalDispatch {
    simd_tiles: Vec<AttentionGemmMetalKernel>,
    mxu_tiles: Option<Vec<AttentionGemmMetalKernel>>,
    simd_key_tile: u32,
}

impl AttentionGemmMetalDispatch {
    fn build_tiles(
        context: &MetalContext,
        data_type: DataType,
        bk: u32,
        bd: u32,
        use_mxu: bool,
        is_kv_cache_ring: bool,
        is_causal: bool,
        is_trie: bool,
        is_sliding_window: bool,
        has_sinks: bool,
    ) -> Result<Vec<AttentionGemmMetalKernel>, MetalError> {
        let mut tiles = Vec::with_capacity(4);
        for align_q in [false, true] {
            for align_k in [false, true] {
                tiles.push(AttentionGemmMetalKernel::new(
                    context,
                    data_type,
                    bk,
                    bd,
                    use_mxu,
                    align_q,
                    align_k,
                    is_kv_cache_ring,
                    is_causal,
                    is_trie,
                    is_sliding_window,
                    has_sinks,
                )?);
            }
        }
        Ok(tiles)
    }
}

impl AttentionGemmDispatch for AttentionGemmMetalDispatch {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
        bk: u32,
        bd: u32,
        is_kv_cache_ring: bool,
        is_causal: bool,
        is_trie: bool,
        is_sliding_window: bool,
        has_sinks: bool,
    ) -> Result<Self, MetalError> {
        let simd_tiles = Self::build_tiles(
            context,
            data_type,
            bk,
            bd,
            false,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window,
            has_sinks,
        )?;

        let mxu_tiles = if context.supports_mxu()
            && matches!(data_type, DataType::BF16 | DataType::F16)
            && matches!(bd, 64 | 128)
        {
            Some(Self::build_tiles(
                context,
                data_type,
                32,
                bd,
                true,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                is_sliding_window,
                has_sinks,
            )?)
        } else {
            None
        };

        Ok(Self {
            simd_tiles,
            mxu_tiles,
            simd_key_tile: bk,
        })
    }

    fn encode<'q, 'k, 'v, 'o, 'trie, 'sinks, 'encoder>(
        &self,
        q: impl BufferArg<'q, Metal>,
        k: impl BufferArg<'k, Metal>,
        v: impl BufferArg<'v, Metal>,
        o: impl BufferArgMut<'o, Metal>,
        params: AttnParams,
        ring_params: Option<RingParams>,
        trie: Option<impl BufferArg<'trie, Metal>>,
        sliding_window_size: Option<u32>,
        sinks: Option<impl BufferArg<'sinks, Metal>>,
        num_heads: u32,
        suffix_length: u32,
        encoder: &'encoder mut Encoder<Metal>,
    ) {
        let use_mxu = suffix_length >= 64;
        if use_mxu && let Some(mxu_tiles) = &self.mxu_tiles {
            let mxu_params = retile_params(params, 64, 32);
            mxu_tiles[tile_variant_index(mxu_params.q_rem == 0, mxu_params.k_rem == 0)].encode(
                q,
                k,
                v,
                o,
                mxu_params,
                ring_params,
                trie,
                sliding_window_size,
                sinks,
                num_heads,
                suffix_length,
                encoder,
            );
        } else {
            let simd_params = retile_params(params, 32, self.simd_key_tile);
            self.simd_tiles[tile_variant_index(simd_params.q_rem == 0, simd_params.k_rem == 0)].encode(
                q,
                k,
                v,
                o,
                simd_params,
                ring_params,
                trie,
                sliding_window_size,
                sinks,
                num_heads,
                suffix_length,
                encoder,
            );
        }
    }
}
