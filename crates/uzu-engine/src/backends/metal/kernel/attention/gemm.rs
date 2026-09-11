use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::AttnParams,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        metal::{Metal, context::MetalContext, error::MetalError, kernel::AttentionGemmMetalKernel},
    },
    data_type::DataType,
};

pub struct AttentionGemm {
    kernels: Mutex<HashMap<AttentionGemmKey, AttentionGemmMetalKernel>>,
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    simd_bk: u32,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct AttentionGemmKey {
    use_mxu: bool,
    align_q: bool,
    align_k: bool,
    is_trie: bool,
}

fn retile_params(
    mut params: AttnParams,
    bq: u32,
    bk: u32,
) -> AttnParams {
    params.nq_aligned = params.q_len / bq;
    params.q_rem = params.q_len % bq;
    params.nk = params.k_len.div_ceil(bk);
    params.nk_aligned = params.k_len / bk;
    params.k_rem = params.k_len % bk;
    params
}

impl AttentionGemm {
    pub fn is_supported(config: &AttentionKernelConfig) -> bool {
        matches!(config.head_dim, 64 | 128 | 256) && matches!(config.data_type, DataType::BF16 | DataType::F32)
    }

    pub fn new(config: &AttentionKernelConfig) -> Self {
        let simd_bk = if config.head_dim < 128 {
            32
        } else {
            16
        };
        Self {
            kernels: Mutex::new(HashMap::new()),
            head_dim: config.head_dim,
            num_groups: config.num_groups,
            num_q_heads: config.num_q_heads,
            sliding_window_size: config.sliding_window_size,
            scale: config.scale,
            data_type: config.data_type,
            simd_bk,
            is_kv_cache_ring: config.is_kv_cache_ring,
            is_causal: config.is_causal,
            is_sliding_window: config.sliding_window_size.is_some(),
            has_sinks: config.has_sinks,
        }
    }

    fn get_or_create(
        &self,
        context: &MetalContext,
        key: AttentionGemmKey,
    ) -> Result<MappedMutexGuard<'_, AttentionGemmMetalKernel>, MetalError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(key) {
            let bk = if key.use_mxu {
                32
            } else {
                self.simd_bk
            };
            let kernel = AttentionGemmMetalKernel::new(
                context,
                self.data_type,
                bk,
                self.head_dim,
                key.use_mxu,
                key.align_q,
                key.align_k,
                self.is_kv_cache_ring,
                self.is_causal,
                key.is_trie,
                self.is_sliding_window,
                self.has_sinks,
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&key).expect("kernel was just initialized")))
    }

    pub fn encode<'a, KT: BufferArg<'a, Metal>, VT: BufferArg<'a, Metal>>(
        &self,
        arguments: AttentionArguments<'a, Metal, KT, VT>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let mut output = encoder
            .allocate_constant_for_shape(&[arguments.suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;

        let use_mxu = arguments.suffix_length >= 64
            && matches!(self.data_type, DataType::BF16)
            && matches!(self.head_dim, 64 | 128)
            && encoder.context().supports_mxu;
        let (bq, bk) = if use_mxu {
            (64, 32)
        } else {
            (32, self.simd_bk)
        };
        let params = retile_params(
            AttnParams {
                q_strides: [0, arguments.suffix_length * self.head_dim, self.head_dim],
                k_strides: [0, self.head_dim, self.num_groups * self.head_dim],
                v_strides: [0, self.head_dim, self.num_groups * self.head_dim],
                o_strides: [0, self.head_dim, self.num_q_heads * self.head_dim],
                gqa_factor: self.num_q_heads / self.num_groups,
                scale: self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt()),
                q_len: arguments.suffix_length,
                k_len: arguments.cache.prefix_len() + arguments.suffix_length,
                q_off: arguments.cache.prefix_len(),
                nq_aligned: 0,
                q_rem: 0,
                nk: 0,
                nk_aligned: 0,
                k_rem: 0,
            },
            bq,
            bk,
        );
        let key = AttentionGemmKey {
            use_mxu,
            align_q: params.q_rem == 0,
            align_k: params.k_rem == 0,
            is_trie: arguments.trie.is_some(),
        };
        let kernel = self.get_or_create(encoder.context(), key)?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            params,
            arguments.cache.ring_params(),
            arguments.trie,
            self.sliding_window_size,
            arguments.sinks,
            self.num_q_heads,
            arguments.suffix_length,
            encoder,
        );
        Ok(output)
    }
}
