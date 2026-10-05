//! Attention: the single-pass and two-pass SIMD kernels of the Metal backend
//! (`backends/metal/kernel/attention/{single_pass,two_pass}.rs`), which cover every suffix length,
//! head size, trie, ring and sink configuration, and its simdgroup-matrix GEMM kernel for suffixes longer
//! than 8 (`gemm.rs`, without the MXU variant). The grouped GEMM kernel is MXU-only and stays Metal's.

use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use super::{
    AttentionGemmAmdgpuKernel, AttentionSinglePassAmdgpuKernel, AttentionTwoPass1AmdgpuKernel,
    AttentionTwoPass2AmdgpuKernel,
};
use crate::{
    backends::{
        amdgpu::{Amdgpu, context::AmdgpuContext, error::AmdgpuError},
        common::{
            Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
            gpu_types::AttnParams,
            kernel::{
                AttentionSinglePassKernel,
                attention::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
            },
        },
    },
    data_type::DataType,
};

type Encoding = <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding;

const SINGLE_PASS_KV_THRESHOLD: u32 = 1_024;
const GEMM_MIN_SUFFIX_LENGTH: u32 = 9;
const PARTIAL_DATA_TYPE: DataType = DataType::F32;
const PARTIAL_BLOCKS: u32 = 32;

struct AttentionSinglePass {
    kernels: Mutex<HashMap<bool, AttentionSinglePassAmdgpuKernel>>,
    config: AttentionKernelConfig,
}

impl AttentionSinglePass {
    fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            config: *config,
        }
    }

    fn get_or_create(
        &self,
        context: &AmdgpuContext,
        is_trie: bool,
    ) -> Result<MappedMutexGuard<'_, AttentionSinglePassAmdgpuKernel>, AmdgpuError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(is_trie) {
            let kernel = AttentionSinglePassAmdgpuKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&is_trie).expect("kernel was just initialized")))
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        command_buffer: &mut Encoding,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        let config = self.config;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let kernel = self.get_or_create(command_buffer.context(), arguments.trie.is_some())?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            config.num_q_heads / config.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            config.head_dim,
            config.num_groups * config.head_dim,
            config.head_dim,
            config.num_groups * config.head_dim,
            arguments.cache.ring_params(),
            config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            arguments.trie,
            config.sliding_window_size,
            arguments.sinks,
            config.num_q_heads,
            arguments.suffix_length,
            command_buffer,
        );
        Ok(output)
    }
}

struct AttentionTwoPass {
    passes: Mutex<HashMap<bool, AttentionTwoPass1AmdgpuKernel>>,
    second: Mutex<Option<AttentionTwoPass2AmdgpuKernel>>,
    config: AttentionKernelConfig,
}

impl AttentionTwoPass {
    fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            passes: Mutex::new(HashMap::new()),
            second: Mutex::new(None),
            config: *config,
        }
    }

    fn get_or_create(
        &self,
        context: &AmdgpuContext,
        is_trie: bool,
    ) -> Result<MappedMutexGuard<'_, AttentionTwoPass1AmdgpuKernel>, AmdgpuError> {
        let mut passes = self.passes.lock();
        if let Entry::Vacant(entry) = passes.entry(is_trie) {
            entry.insert(AttentionTwoPass1AmdgpuKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
            )?);
        }
        Ok(MutexGuard::map(passes, |passes| passes.get_mut(&is_trie).expect("passes were just initialized")))
    }

    fn get_or_create_second(
        &self,
        context: &AmdgpuContext,
    ) -> Result<MappedMutexGuard<'_, AttentionTwoPass2AmdgpuKernel>, AmdgpuError> {
        let mut second = self.second.lock();
        if second.is_none() {
            *second = Some(AttentionTwoPass2AmdgpuKernel::new(context, self.config.data_type, self.config.head_dim)?);
        }
        Ok(MutexGuard::map(second, |second| second.as_mut().expect("pass was just initialized")))
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        command_buffer: &mut Encoding,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        let config = self.config;
        let mut partials = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS, config.head_dim],
            PARTIAL_DATA_TYPE,
        )?;
        let mut sums = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS],
            PARTIAL_DATA_TYPE,
        )?;
        let mut maxs = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS],
            PARTIAL_DATA_TYPE,
        )?;
        let first = self.get_or_create(command_buffer.context(), arguments.trie.is_some())?;
        first.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut partials,
            &mut sums,
            &mut maxs,
            config.num_q_heads / config.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            config.head_dim,
            config.num_groups * config.head_dim,
            config.head_dim,
            config.num_groups * config.head_dim,
            arguments.cache.ring_params(),
            config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            config.num_q_heads,
            arguments.suffix_length,
            arguments.trie,
            config.sliding_window_size,
            arguments.sinks,
            command_buffer,
        );
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let second = self.get_or_create_second(command_buffer.context())?;
        second.encode(
            &partials,
            &sums,
            &maxs,
            &mut output,
            config.num_q_heads,
            arguments.suffix_length,
            command_buffer,
        );
        Ok(output)
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct AttentionGemmKey {
    align_q: bool,
    align_k: bool,
    is_trie: bool,
}

/// Query blocks of 32 rows; key blocks of 32 (head size < 128) or 16 keys, as Metal's simdgroup variant.
struct AttentionGemm {
    kernels: Mutex<HashMap<AttentionGemmKey, AttentionGemmAmdgpuKernel>>,
    config: AttentionKernelConfig,
    bk: u32,
}

impl AttentionGemm {
    const BQ: u32 = 32;

    fn is_supported(config: &AttentionKernelConfig) -> bool {
        matches!(config.head_dim, 64 | 128 | 256) && matches!(config.data_type, DataType::BF16 | DataType::F32)
    }

    fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            config: *config,
            bk: if config.head_dim < 128 {
                32
            } else {
                16
            },
        }
    }

    fn get_or_create(
        &self,
        context: &AmdgpuContext,
        key: AttentionGemmKey,
    ) -> Result<MappedMutexGuard<'_, AttentionGemmAmdgpuKernel>, AmdgpuError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(key) {
            let kernel = AttentionGemmAmdgpuKernel::new(
                context,
                self.config.data_type,
                self.bk,
                self.config.head_dim,
                false,
                key.align_q,
                key.align_k,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                key.is_trie,
                self.config.sliding_window_size.is_some(),
                self.config.has_sinks,
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&key).expect("kernel was just initialized")))
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        command_buffer: &mut Encoding,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        let config = self.config;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let q_len = arguments.suffix_length;
        let k_len = arguments.cache.prefix_len() + arguments.suffix_length;
        let params = AttnParams {
            q_strides: [0, q_len * config.head_dim, config.head_dim],
            k_strides: [0, config.head_dim, config.num_groups * config.head_dim],
            v_strides: [0, config.head_dim, config.num_groups * config.head_dim],
            o_strides: [0, config.head_dim, config.num_q_heads * config.head_dim],
            gqa_factor: config.num_q_heads / config.num_groups,
            scale: config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            q_len,
            k_len,
            q_off: arguments.cache.prefix_len(),
            nq_aligned: q_len / Self::BQ,
            q_rem: q_len % Self::BQ,
            nk: k_len.div_ceil(self.bk),
            nk_aligned: k_len / self.bk,
            k_rem: k_len % self.bk,
        };
        let key = AttentionGemmKey {
            align_q: params.q_rem == 0,
            align_k: params.k_rem == 0,
            is_trie: arguments.trie.is_some(),
        };
        let kernel = self.get_or_create(command_buffer.context(), key)?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            params,
            arguments.cache.ring_params(),
            arguments.trie,
            config.sliding_window_size,
            arguments.sinks,
            config.num_q_heads,
            arguments.suffix_length,
            command_buffer,
        );
        Ok(output)
    }
}

pub struct AmdgpuAttentionKernel {
    single_pass: AttentionSinglePass,
    two_pass: AttentionTwoPass,
    gemm: Option<AttentionGemm>,
}

impl AttentionKernel for AmdgpuAttentionKernel {
    type Backend = Amdgpu;

    fn new(
        _context: &AmdgpuContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, AmdgpuError> {
        Ok(Self {
            single_pass: AttentionSinglePass::new(&config),
            two_pass: AttentionTwoPass::new(&config),
            gemm: AttentionGemm::is_supported(&config).then(|| AttentionGemm::new(&config)),
        })
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        command_buffer: &mut Encoding,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        if arguments.suffix_length >= GEMM_MIN_SUFFIX_LENGTH
            && let Some(gemm) = &self.gemm
        {
            return gemm.encode(arguments, command_buffer);
        }
        let kv_length = arguments.cache.prefix_len() + arguments.suffix_length;
        if kv_length > SINGLE_PASS_KV_THRESHOLD {
            self.two_pass.encode(arguments, command_buffer)
        } else {
            self.single_pass.encode(arguments, command_buffer)
        }
    }
}
