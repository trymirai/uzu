use std::collections::{HashMap, hash_map::Entry};

use super::AttentionGemmMetalKernel;
use crate::{
    backends::{
        common::{
            Encoder,
            gpu_types::{attention::AttnParams, ring::RingParams},
            kernel::{
                AttentionGemmKernel, BufferArg, BufferArgMut,
                attention_gemm::{AttentionGemmDispatch, retile_params},
            },
        },
        metal::{DeviceExt, Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct AttentionGemmMetalDispatch {
    kernels: HashMap<AttentionGemmKey, AttentionGemmMetalKernel>,
    data_type: DataType,
    bd: u32,
    simd_bk: u32,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct AttentionGemmKey {
    use_mxu: bool,
    align_q: bool,
    align_k: bool,
}

impl AttentionGemmMetalDispatch {
    fn get_or_create(
        &mut self,
        context: &MetalContext,
        key: AttentionGemmKey,
        bk: u32,
    ) -> Result<&AttentionGemmMetalKernel, MetalError> {
        match self.kernels.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = AttentionGemmMetalKernel::new(
                    context,
                    self.data_type,
                    bk,
                    self.bd,
                    key.use_mxu,
                    key.align_q,
                    key.align_k,
                    self.is_kv_cache_ring,
                    self.is_causal,
                    self.is_trie,
                    self.is_sliding_window,
                    self.has_sinks,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }
}

impl AttentionGemmDispatch for AttentionGemmMetalDispatch {
    type Backend = Metal;

    fn new(
        _context: &MetalContext,
        data_type: DataType,
        bk: u32,
        bd: u32,
        is_kv_cache_ring: bool,
        is_causal: bool,
        is_trie: bool,
        is_sliding_window: bool,
        has_sinks: bool,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            kernels: HashMap::new(),
            data_type,
            bd,
            simd_bk: bk,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window,
            has_sinks,
        })
    }

    fn encode<'q, 'k, 'v, 'o, 'trie, 'sinks, 'encoder>(
        &mut self,
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
    ) -> Result<(), MetalError> {
        let use_mxu = suffix_length >= 64
            && encoder.context().device.supports_mxu()
            && matches!(self.data_type, DataType::BF16 | DataType::F16)
            && matches!(self.bd, 64 | 128);
        let params = if use_mxu {
            retile_params(params, 64, 32)
        } else {
            retile_params(params, 32, self.simd_bk)
        };
        let key = AttentionGemmKey {
            use_mxu,
            align_q: params.q_rem == 0,
            align_k: params.k_rem == 0,
        };
        let bk = if use_mxu {
            32
        } else {
            self.simd_bk
        };
        let kernel = self.get_or_create(encoder.context(), key, bk)?;

        kernel.encode(
            q,
            k,
            v,
            o,
            params,
            ring_params,
            trie,
            sliding_window_size,
            sinks,
            num_heads,
            suffix_length,
            encoder,
        );
        Ok(())
    }
}
