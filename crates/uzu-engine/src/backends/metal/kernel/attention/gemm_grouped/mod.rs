use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::AttnParams,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionGemmGroupedCombineMetalKernel, AttentionGemmGroupedMetalKernel},
        },
    },
    data_type::DataType,
};

mod policy;
use policy::{MAX_TRIE_SUFFIX, choose_splits};

use super::MaskKind;

const SIMDGROUPS_PER_THREADGROUP: u32 = 8;
const SLICE_COLS: u32 = 64;
const TILE_ROWS: u32 = 16;
const BLOCK_K: u32 = 32;

struct AttentionGemmGroupedMetal {
    kernel: AttentionGemmGroupedMetalKernel,
    split_kernel: AttentionGemmGroupedMetalKernel,
    combine: AttentionGemmGroupedCombineMetalKernel,
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    scale: Option<f32>,
    block_rows: u32,
    mask: MaskKind,
}

impl AttentionGemmGroupedMetal {
    fn new_fixed(
        context: &MetalContext,
        head_dim: u32,
        num_groups: u32,
        num_q_heads: u32,
        scale: Option<f32>,
        mask: MaskKind,
    ) -> Result<Self, MetalError> {
        assert!(matches!(head_dim, 128 | 256), "head_dim must be 128 or 256");
        assert_eq!(num_q_heads % num_groups, 0, "num_q_heads must be divisible by num_groups");
        let slices = head_dim / SLICE_COLS;
        let block_rows = SIMDGROUPS_PER_THREADGROUP / slices * TILE_ROWS;
        let new_kernel = |split| {
            AttentionGemmGroupedMetalKernel::new(
                context,
                DataType::BF16,
                BLOCK_K,
                head_dim,
                slices,
                split,
                mask.is_causal(),
                mask.is_trie(),
            )
        };
        let combine = AttentionGemmGroupedCombineMetalKernel::new(context, DataType::BF16, head_dim)?;
        Ok(Self {
            kernel: new_kernel(false)?,
            split_kernel: new_kernel(true)?,
            combine,
            head_dim,
            num_groups,
            num_q_heads,
            scale,
            block_rows,
            mask,
        })
    }

    fn encode(
        &self,
        queries: impl BufferRef<Backend = Metal>,
        keys: impl BufferRef<Backend = Metal>,
        values: impl BufferRef<Backend = Metal>,
        output: impl BufferMut<Backend = Metal>,
        trie: Option<impl BufferRef<Backend = Metal>>,
        suffix_length: u32,
        kv_length: u32,
        q_replicas: u32,
        num_splits: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let params = self.params(suffix_length, kv_length);
        let grouped_rows = self.num_q_heads / self.num_groups * suffix_length;
        let m_tiles = grouped_rows.div_ceil(self.block_rows);
        assert_eq!(trie.is_some(), self.mask.is_trie(), "trie presence must match mask");
        assert!(num_splits >= 1, "num_splits must be at least one");
        assert!(num_splits <= params.nk, "num_splits must not exceed KV blocks");

        if num_splits == 1 {
            self.kernel.encode(
                queries,
                keys,
                values,
                Some(output),
                None::<&mut <Metal as Backend>::ScratchBuffer>,
                None::<&mut <Metal as Backend>::ScratchBuffer>,
                None::<&mut <Metal as Backend>::ScratchBuffer>,
                trie,
                params,
                m_tiles,
                self.num_groups,
                q_replicas,
                1,
                command_buffer,
            );
            return Ok(());
        }

        let partial_rows = q_replicas * self.num_groups * m_tiles * num_splits * self.block_rows;
        let mut partials = command_buffer.allocate_scratch_for_shape(&[partial_rows, self.head_dim], DataType::F32)?;
        let mut partial_maxs = command_buffer.allocate_scratch_for_shape(&[partial_rows], DataType::F32)?;
        let mut partial_sums = command_buffer.allocate_scratch_for_shape(&[partial_rows], DataType::F32)?;

        self.split_kernel.encode(
            queries,
            keys,
            values,
            None::<&mut <Metal as Backend>::ScratchBuffer>,
            Some(&mut partials),
            Some(&mut partial_maxs),
            Some(&mut partial_sums),
            trie,
            params,
            m_tiles,
            self.num_groups,
            q_replicas,
            num_splits,
            command_buffer,
        );

        self.combine.encode(
            &partials,
            &partial_maxs,
            &partial_sums,
            output,
            params,
            grouped_rows,
            self.block_rows,
            m_tiles,
            self.num_groups,
            q_replicas,
            num_splits,
            command_buffer,
        );
        Ok(())
    }

    fn params(
        &self,
        suffix_length: u32,
        kv_length: u32,
    ) -> AttnParams {
        assert!(suffix_length > 0 && kv_length > 0, "lengths must be positive");
        assert!(kv_length >= suffix_length, "kv_length must be at least suffix_length");
        assert!(!self.mask.is_trie() || suffix_length <= MAX_TRIE_SUFFIX, "trie suffix is too long");

        let head_dim = self.head_dim;
        AttnParams {
            q_strides: [self.num_q_heads * suffix_length * head_dim, suffix_length * head_dim, head_dim],
            k_strides: [0, head_dim, self.num_groups * head_dim],
            v_strides: [0, head_dim, self.num_groups * head_dim],
            o_strides: [suffix_length * self.num_q_heads * head_dim, head_dim, self.num_q_heads * head_dim],
            gqa_factor: self.num_q_heads / self.num_groups,
            scale: self.scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt()),
            q_len: suffix_length,
            k_len: kv_length,
            q_off: kv_length - suffix_length,
            nq_aligned: 0,
            q_rem: 0,
            nk: kv_length.div_ceil(BLOCK_K),
            nk_aligned: kv_length / BLOCK_K,
            k_rem: kv_length % BLOCK_K,
        }
    }
}

pub struct AttentionGemmGrouped {
    non_trie: Mutex<Option<AttentionGemmGroupedMetal>>,
    trie: Mutex<Option<AttentionGemmGroupedMetal>>,
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    scale: Option<f32>,
}

impl AttentionGemmGrouped {
    fn get_or_create(
        &self,
        context: &MetalContext,
        mask: MaskKind,
    ) -> Result<MappedMutexGuard<'_, AttentionGemmGroupedMetal>, MetalError> {
        let cache = if mask.is_trie() {
            &self.trie
        } else {
            &self.non_trie
        };
        let mut cache = cache.lock();
        if cache.is_none() {
            *cache = Some(AttentionGemmGroupedMetal::new_fixed(
                context,
                self.head_dim,
                self.num_groups,
                self.num_q_heads,
                self.scale,
                mask,
            )?);
        }
        Ok(MutexGuard::map(cache, |cache| cache.as_mut().expect("attention pipeline was just initialized")))
    }

    pub fn is_supported(
        config: &AttentionKernelConfig,
        context: &MetalContext,
    ) -> bool {
        policy::is_supported(config, context)
    }

    pub fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            non_trie: Mutex::new(None),
            trie: Mutex::new(None),
            head_dim: config.head_dim,
            num_groups: config.num_groups,
            num_q_heads: config.num_q_heads,
            scale: config.scale,
        }
    }

    pub fn should_encode(
        &self,
        mask: MaskKind,
        suffix_length: u32,
        kv_length: u32,
    ) -> bool {
        policy::should_encode(self.head_dim, mask, suffix_length, kv_length)
    }

    pub fn encode(
        &self,
        mask: MaskKind,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let suffix_length = arguments.suffix_length;
        assert!(arguments.cache.ring_params().is_none(), "ring KV cache is unsupported");
        assert!(arguments.sinks.is_none(), "attention sinks are unsupported");

        let kv_length = arguments.cache.prefix_len() + suffix_length;
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[suffix_length, self.num_q_heads, self.head_dim], DataType::BF16)?;
        let gpu_core_count = command_buffer.context().gpu_core_count;
        let core = self.get_or_create(command_buffer.context(), mask)?;
        let num_splits = choose_splits(
            core.head_dim,
            suffix_length,
            kv_length,
            (core.num_q_heads / core.num_groups * suffix_length).div_ceil(core.block_rows) * core.num_groups,
            BLOCK_K,
            gpu_core_count,
        );
        core.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            arguments.trie,
            suffix_length,
            kv_length,
            1,
            num_splits,
            command_buffer,
        )?;
        Ok(output)
    }
}
