use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use super::FullPrecisionInput;
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::attention::AttnParams,
            kernel::{AttentionGemmGroupedCombineMetalKernel, AttentionGemmGroupedMetalKernel},
        },
    },
    data_type::DataType,
};

pub(super) mod policy;
use policy::{MAX_TRIE_SUFFIX, SplitGeometry, choose_splits};

use super::MaskKind;

const SIMDGROUPS_PER_THREADGROUP: u32 = 8;
const SLICE_COLS: u32 = 64;
const MXU_TILE_ROWS: u32 = 16;
const SIMDGROUP_TILE_ROWS: u32 = 8;
const BLOCK_K: u32 = 32;

pub fn query_rows_per_threadgroup(head_dim: u32) -> u32 {
    SIMDGROUPS_PER_THREADGROUP / (head_dim / SLICE_COLS) * MXU_TILE_ROWS
}

struct AttentionGemmGroupedMetal {
    kernel: AttentionGemmGroupedMetalKernel,
    split_kernel: AttentionGemmGroupedMetalKernel,
    combine: AttentionGemmGroupedCombineMetalKernel,
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    scale: Option<f32>,
    gate_row_stride: Option<u32>,
    query_rows_per_threadgroup: u32,
    block_k: u32,
    mask: MaskKind,
}

impl AttentionGemmGroupedMetal {
    fn new_fixed(
        context: &MetalContext,
        config: &AttentionKernelConfig,
        head_dim_slices: u32,
        mask: MaskKind,
    ) -> Result<Self, MetalError> {
        let tile_rows = if context.supports_mxu {
            MXU_TILE_ROWS
        } else {
            SIMDGROUP_TILE_ROWS
        };
        let block_k = if head_dim_slices == 1 {
            16
        } else {
            BLOCK_K
        };
        assert!(matches!(config.head_dim, 128 | 256), "head_dim must be 128 or 256");
        assert_eq!(config.num_q_heads % config.num_groups, 0, "num_q_heads must be divisible by num_groups");
        let new_kernel = |split| {
            AttentionGemmGroupedMetalKernel::new(
                context,
                DataType::BF16,
                block_k,
                config.head_dim,
                head_dim_slices,
                context.supports_mxu,
                split,
                mask.is_causal(),
                mask.is_trie(),
            )
        };
        let combine = AttentionGemmGroupedCombineMetalKernel::new(
            context,
            DataType::BF16,
            config.head_dim,
            config.gate_row_stride.is_some(),
        )?;
        Ok(Self {
            kernel: new_kernel(false)?,
            split_kernel: new_kernel(true)?,
            combine,
            head_dim: config.head_dim,
            num_groups: config.num_groups,
            num_q_heads: config.num_q_heads,
            scale: config.scale,
            gate_row_stride: config.gate_row_stride,
            query_rows_per_threadgroup: SIMDGROUPS_PER_THREADGROUP / head_dim_slices * tile_rows,
            block_k,
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
        gate: Option<impl BufferRef<Backend = Metal>>,
        suffix_length: u32,
        kv_length: u32,
        num_splits: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<bool, MetalError> {
        let params = self.params(suffix_length, kv_length);
        let grouped_rows = self.num_q_heads / self.num_groups * suffix_length;
        let query_tiles = grouped_rows.div_ceil(self.query_rows_per_threadgroup);
        assert_eq!(trie.is_some(), self.mask.is_trie(), "trie presence must match mask");

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
                query_tiles,
                self.num_groups,
                1,
                command_buffer,
            );
            return Ok(false);
        }

        let partial_rows = self.num_groups * query_tiles * num_splits * self.query_rows_per_threadgroup;
        let mut partials = command_buffer.allocate_scratch_for_shape(&[partial_rows, self.head_dim], DataType::BF16)?;
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
            query_tiles,
            self.num_groups,
            num_splits,
            command_buffer,
        );

        self.combine.encode(
            &partials,
            &partial_maxs,
            &partial_sums,
            output,
            gate,
            params,
            grouped_rows,
            query_tiles * self.query_rows_per_threadgroup,
            self.num_groups,
            num_splits,
            self.gate_row_stride,
            command_buffer,
        );
        Ok(gate.is_some())
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
            q_strides: [0, suffix_length * head_dim, head_dim],
            k_strides: [0, head_dim, self.num_groups * head_dim],
            v_strides: [0, head_dim, self.num_groups * head_dim],
            o_strides: [0, head_dim, self.num_q_heads * head_dim],
            gqa_factor: self.num_q_heads / self.num_groups,
            scale: self.scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt()),
            q_len: suffix_length,
            k_len: kv_length,
            q_off: kv_length - suffix_length,
            nq_aligned: 0,
            q_rem: 0,
            nk: kv_length.div_ceil(self.block_k),
            nk_aligned: kv_length / self.block_k,
            k_rem: kv_length % self.block_k,
        }
    }
}

pub struct AttentionGemmGrouped {
    variants: Mutex<HashMap<(MaskKind, u32), AttentionGemmGroupedMetal>>,
    config: AttentionKernelConfig,
}

impl AttentionGemmGrouped {
    fn get_or_create(
        &self,
        context: &MetalContext,
        mask: MaskKind,
        head_dim_slices: u32,
    ) -> Result<MappedMutexGuard<'_, AttentionGemmGroupedMetal>, MetalError> {
        let key = (mask, head_dim_slices);
        let mut variants = self.variants.lock();
        if let Entry::Vacant(slot) = variants.entry(key) {
            slot.insert(AttentionGemmGroupedMetal::new_fixed(context, &self.config, head_dim_slices, mask)?);
        }
        Ok(MutexGuard::map(variants, |variants| {
            variants.get_mut(&key).expect("attention variant was just initialized")
        }))
    }

    pub fn is_supported(config: &AttentionKernelConfig) -> bool {
        policy::is_supported(config)
    }

    pub fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            variants: Mutex::new(HashMap::new()),
            config: *config,
        }
    }

    pub fn should_encode(
        &self,
        mask: MaskKind,
        suffix_length: u32,
        kv_length: u32,
        use_mxu: bool,
    ) -> bool {
        let slices = policy::select_head_dim_slices(self.config.head_dim, mask, suffix_length, kv_length, use_mxu);
        slices.is_some_and(|slices| slices != 1 || self.config.num_q_heads == self.config.num_groups * 4)
    }

    pub fn encode(
        &self,
        mask: MaskKind,
        input: FullPrecisionInput<
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        arguments: AttentionArguments<'_, Metal, impl BufferRef<Backend = Metal>, impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, bool), MetalError> {
        let suffix_length = arguments.suffix_length;
        assert!(arguments.cache.ring_params().is_none(), "ring KV cache is unsupported");
        assert!(arguments.sinks.is_none(), "attention sinks are unsupported");

        let kv_length = arguments.cache.prefix_len() + suffix_length;
        let head_dim_slices = policy::select_head_dim_slices(
            self.config.head_dim,
            mask,
            suffix_length,
            kv_length,
            command_buffer.context().supports_mxu,
        )
        .expect("grouped attention request must be supported");
        assert!(head_dim_slices != 1 || self.config.num_q_heads == self.config.num_groups * 4);
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[suffix_length, self.config.num_q_heads, self.config.head_dim],
            DataType::BF16,
        )?;
        let gpu_core_count = command_buffer.context().gpu_core_count;
        let core = self.get_or_create(command_buffer.context(), mask, head_dim_slices)?;
        let geometry = SplitGeometry {
            head_dim: core.head_dim,
            num_q_heads: core.num_q_heads,
            num_groups: core.num_groups,
            query_rows_per_threadgroup: core.query_rows_per_threadgroup,
            block_k: core.block_k,
        };
        let num_splits =
            choose_splits(geometry, suffix_length, kv_length, gpu_core_count, command_buffer.context().supports_mxu);
        let gate_applied = core.encode(
            input.queries,
            input.keys,
            input.values,
            &mut output,
            arguments.trie,
            arguments.gate,
            suffix_length,
            kv_length,
            num_splits,
            command_buffer,
        )?;
        Ok((output, gate_applied))
    }
}
