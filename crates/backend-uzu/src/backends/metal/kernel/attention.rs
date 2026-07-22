use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    array::size_for_shape,
    backends::{
        common::{Allocation, BufferArg, Encoder, gpu_types::AttnParams, kernel::attention_gemm::AttentionGemmCore},
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionGemmKey, AttentionGemmMetalKernel},
        },
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub struct AttentionGemmMetalCore {
    kernels: Mutex<HashMap<AttentionGemmSpecialization, AttentionGemmMetalKernel>>,
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    simd_bk: u32,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}

/// The template variant, plus the function constants specialized into it.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct AttentionGemmSpecialization {
    key: AttentionGemmKey,
    align_q: bool,
    align_k: bool,
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
        specialization: AttentionGemmSpecialization,
    ) -> Result<MappedMutexGuard<'_, AttentionGemmMetalKernel>, MetalError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(specialization) {
            // A hit is proof the variant exists; only a miss has to ask.
            specialization.key.validate()?;
            let kernel = AttentionGemmMetalKernel::from_key(
                context,
                specialization.key,
                specialization.align_q,
                specialization.align_k,
                self.is_kv_cache_ring,
                self.is_causal,
                self.is_trie,
                self.is_sliding_window,
                self.has_sinks,
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&specialization).expect("kernel was just initialized")))
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
        _context: &MetalContext,
        arguments: &AttentionCoreNewArguments,
    ) -> Result<Self, MetalError> {
        let simd_bk = if arguments.head_dim < 128 {
            32
        } else {
            16
        };

        Ok(Self {
            kernels: Mutex::new(HashMap::new()),
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
        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;

        let mxu_key = AttentionGemmKey {
            t: self.data_type,
            bk: 32,
            bd: self.head_dim as u32,
            use_mxu: true,
        };
        let use_mxu = arguments.suffix_length >= 64 && encoder.context().supports_mxu() && mxu_key.validate().is_ok();
        let (bq, bk) = if use_mxu {
            (64, 32)
        } else {
            (32, self.simd_bk)
        };
        let params = retile_params(
            AttnParams {
                q_strides: [0, (arguments.suffix_length * self.head_dim) as u64, self.head_dim as u64],
                k_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                v_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                o_strides: [0, self.head_dim as u64, (self.num_q_heads * self.head_dim) as u64],
                gqa_factor: (self.num_q_heads / self.num_groups) as u32,
                scale: self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                q_len: arguments.suffix_length as u32,
                k_len: (arguments.state_type.physical_prefix_length() + arguments.suffix_length) as u32,
                q_off: arguments.state_type.physical_prefix_length() as u32,
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
            t: self.data_type,
            bk,
            bd: self.head_dim as u32,
            use_mxu,
        };
        let specialization = AttentionGemmSpecialization {
            key,
            align_q: params.q_rem == 0,
            align_k: params.k_rem == 0,
        };
        let kernel = self.get_or_create(encoder.context(), specialization)?;

        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            params,
            arguments.state_type.ring_params(),
            arguments.trie,
            self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
            arguments.sinks,
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            encoder,
        );
        Ok(output)
    }
}
