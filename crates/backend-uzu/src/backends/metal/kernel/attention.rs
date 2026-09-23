use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::AttnParams,
            kernel::{AttentionTwoPass2Kernel, attention_gemm::AttentionGemmCore},
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionGemmMetalKernel, AttentionTwoPass2MetalKernel},
        },
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

/// Split the keys of a long context this many ways when there are few query tiles, then combine the parts with
/// AttentionTwoPass2 (whose block count is also 32).
const SPLITS: u32 = 32;

pub struct AttentionGemmMetalCore {
    kernels: Mutex<HashMap<AttentionGemmKey, AttentionGemmMetalKernel>>,
    merge: AttentionTwoPass2MetalKernel,
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
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
    split: bool,
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

impl AttentionGemmMetalCore {
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
                self.is_trie,
                self.is_sliding_window,
                self.has_sinks,
                key.split,
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&key).expect("kernel was just initialized")))
    }
}

impl AttentionGemmCore<Metal> for AttentionGemmMetalCore {
    fn is_supported(
        arguments: &AttentionCoreNewArguments,
        _context: &MetalContext,
    ) -> Result<bool, MetalError> {
        Ok(matches!(arguments.head_dim, 64 | 128 | 256))
    }

    fn new(
        context: &MetalContext,
        arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, MetalError> {
        let simd_bk = if arguments.head_dim < 128 {
            32
        } else {
            16
        };

        Ok(Self {
            kernels: Mutex::new(HashMap::new()),
            merge: AttentionTwoPass2MetalKernel::new(context, arguments.data_type, arguments.head_dim)?,
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            simd_bk,
            is_kv_cache_ring: arguments.is_kv_cache_ring,
            is_causal: arguments.is_causal,
            is_trie: arguments.is_trie,
            is_sliding_window: arguments.sliding_window_size.is_some(),
            has_sinks: arguments.has_sinks,
        })
    }

    fn encode<'a, KT: BufferArg<'a, Metal>, VT: BufferArg<'a, Metal>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, Metal, KT, VT>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let mut output = encoder
            .allocate_constant_for_shape(&[arguments.suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;

        let use_mxu = arguments.suffix_length >= 64
            && encoder.context().supports_mxu()
            && matches!(self.data_type, DataType::BF16 | DataType::F16)
            && matches!(self.head_dim, 64 | 128);
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
                scale: self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                q_len: arguments.suffix_length,
                k_len: arguments.state_type.physical_prefix_length() + arguments.suffix_length,
                q_off: arguments.state_type.physical_prefix_length(),
                nq_aligned: 0,
                q_rem: 0,
                nk: 0,
                nk_aligned: 0,
                k_rem: 0,
            },
            bq,
            bk,
        );
        // Speculative verification: a handful of query tiles against a long context. Without the split each
        // (tile, head) threadgroup walks every key alone.
        let split = !self.has_sinks && arguments.suffix_length <= 64 && params.k_len > 1024;
        let key = AttentionGemmKey {
            use_mxu,
            align_q: params.q_rem == 0,
            align_k: params.k_rem == 0,
            split,
        };
        let kernel = self.get_or_create(encoder.context(), key)?;
        let parts = [arguments.suffix_length, self.num_q_heads, SPLITS];
        let mut partials = split
            .then(|| encoder.allocate_scratch_for_shape(&[parts[0], parts[1], parts[2], self.head_dim], DataType::F32))
            .transpose()?;
        let mut sums = split.then(|| encoder.allocate_scratch_for_shape(&parts, DataType::F32)).transpose()?;
        let mut maxs = split.then(|| encoder.allocate_scratch_for_shape(&parts, DataType::F32)).transpose()?;

        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            params,
            arguments.state_type.ring_params(),
            arguments.trie,
            self.sliding_window_size,
            arguments.sinks,
            partials.as_mut(),
            sums.as_mut(),
            maxs.as_mut(),
            self.num_q_heads,
            arguments.suffix_length,
            if split { SPLITS } else { 1 },
            encoder,
        );
        if let (Some(partials), Some(sums), Some(maxs)) = (&partials, &sums, &maxs) {
            self.merge.encode(partials, sums, maxs, &mut output, self.num_q_heads, arguments.suffix_length, encoder);
        }
        Ok(output)
    }
}
