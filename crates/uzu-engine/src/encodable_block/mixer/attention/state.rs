use std::any::Any;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Context, DeviceCapabilities, GpuBuffer, Kernels, SparseBuffer,
        gpu_types::{kv_cache_update::Copy, ring::RingParams},
        kernel::{AttentionKernel, KVCacheUpdateKernel, kv_layout::KVLayout},
    },
    encodable_block::{
        batch_topology::{BatchKind, BatchTopology},
        mixer::{Mixer, MixerState, attention::Attention},
    },
};

pub const ATTENTION_SUFFIX_CAPACITY: u32 = 1024; // TODO: remove hardcoded suffix capacity

#[derive(Clone, Copy)]
pub struct KVCacheState {
    context_length: u32,
    ring_capacity: Option<u32>,
    rope_position: u32,
}

impl KVCacheState {
    pub fn new(
        context_length: u32,
        ring_capacity: Option<u32>,
    ) -> Self {
        assert!(ring_capacity != Some(0), "zero ring capacity");
        Self {
            context_length,
            ring_capacity,
            rope_position: context_length,
        }
    }

    pub fn context_length(self) -> u32 {
        self.context_length
    }

    pub fn rope_position(self) -> u32 {
        self.rope_position
    }

    pub fn prefix_len(self) -> u32 {
        self.required_prefix_len(self.context_length)
    }

    pub fn ring_params(self) -> Option<RingParams> {
        let ring_length = self.prefix_len();
        // Rows are written round-robin, so once the ring is full the oldest one is the next to be overwritten.
        self.ring_capacity.map(|capacity| RingParams {
            ring_offset: (self.context_length - ring_length) % capacity,
            ring_length,
        })
    }

    fn required_prefix_len(
        self,
        context_length: u32,
    ) -> u32 {
        self.ring_capacity.map_or(context_length, |capacity| context_length.min(capacity))
    }

    fn accept<B: Backend>(
        &mut self,
        batch_kind: &BatchKind<B>,
        accepted_indices: &[u32],
    ) -> Vec<Copy> {
        assert!(accepted_indices.is_sorted_by(|a, b| a < b), "unsorted accept");
        let suffix_base = self.prefix_len();
        let count = accepted_indices.len() as u32;

        let mut copies = Vec::with_capacity(accepted_indices.len());
        for (index, accepted_index) in accepted_indices.iter().copied().enumerate() {
            let source = suffix_base + accepted_index;
            let destination = self.ring_capacity.map_or(self.context_length, |capacity| self.context_length % capacity);
            // A copy is dead when a later accept in the same batch overwrites its
            // destination; emitting it would alias in a single kernel dispatch.
            let is_dead = self.ring_capacity.is_some_and(|capacity| index as u32 + capacity < count);
            if !is_dead && source != destination {
                copies.push(Copy {
                    source,
                    destination,
                });
            }
            self.context_length += 1;
        }

        self.rope_position += match batch_kind {
            &BatchKind::Image {
                width,
                height,
                first_cell,
                ..
            } if first_cell + count == width * height => width.max(height),
            BatchKind::Image {
                ..
            } => 0,
            _ => count,
        };
        copies
    }
}

pub struct KVCache<B: Backend> {
    pub layout: KVLayout,
    pub k_codes: Box<dyn GpuBuffer<Backend = B>>,
    pub v_codes: Box<dyn GpuBuffer<Backend = B>>,
    pub k_scales: Box<dyn GpuBuffer<Backend = B>>,
    pub v_scales: Box<dyn GpuBuffer<Backend = B>>,
}

impl<B: Backend> KVCache<B> {
    pub fn allocate_scratch(
        layout: KVLayout,
        max_elements: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            k_codes: Box::new(command_buffer.allocate_scratch(layout.code_bytes(max_elements))?),
            v_codes: Box::new(command_buffer.allocate_scratch(layout.code_bytes(max_elements))?),
            k_scales: Box::new(command_buffer.allocate_scratch(layout.scale_bytes(max_elements))?),
            v_scales: Box::new(command_buffer.allocate_scratch(layout.scale_bytes(max_elements))?),
            layout,
        })
    }
}

pub struct AttentionState<B: Backend> {
    cache: KVCacheState,
    pub kv: KVCache<B>,
    kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
}

impl<B: Backend> AttentionState<B> {
    pub fn cache(&self) -> KVCacheState {
        self.cache
    }

    pub fn create_empty(
        attention: &Attention<B>,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let max_rope_length = attention.max_context_length();
        if let Some(max_context_length) = max_context_length {
            assert!(
                max_rope_length.is_none_or(|max_rope_length| max_context_length <= max_rope_length),
                "context exceeds RoPE length"
            );
        }

        let max_prefix_elements = attention.ring_capacity.unwrap_or_else(|| {
            max_context_length
                .or(max_rope_length)
                .expect("Cannot create full attention state with unlimited length for with no RoPE")
        });

        let cache = KVCacheState::new(0, attention.ring_capacity);

        let num_kv_heads = attention.num_kv_heads.unwrap();
        let head_dim = attention.head_dim;
        let layout = attention.kernel.kv_layout();
        let max_elements = max_prefix_elements + ATTENTION_SUFFIX_CAPACITY;
        let code_bytes = layout.code_bytes(max_elements);
        let scale_bytes = layout.scale_bytes(max_elements);
        let allocate = |bytes| -> Result<Box<dyn GpuBuffer<Backend = B>>, B::Error> {
            Ok(if context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
                Box::new(context.create_sparse_buffer(bytes)?)
            } else {
                Box::new(context.create_buffer_gpu(bytes)?)
            })
        };
        let kv = KVCache {
            k_codes: allocate(code_bytes)?,
            v_codes: allocate(code_bytes)?,
            k_scales: allocate(scale_bytes)?,
            v_scales: allocate(scale_bytes)?,
            layout,
        };
        let kv_cache_update = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(
            context,
            head_dim,
            num_kv_heads,
            layout.backend_opaque_layout,
            false,
        )?;

        Ok(Self {
            cache,
            kv,
            kv_cache_update,
        })
    }
}

impl<B: Backend> MixerState<B> for AttentionState<B> {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        assert!(suffix_length <= ATTENTION_SUFFIX_CAPACITY, "suffix exceeds capacity");

        let map = |buffer: &mut dyn GpuBuffer<Backend = B>, bytes: usize| -> Result<(), B::Error> {
            match (buffer as &mut dyn Any).downcast_mut::<B::SparseBuffer>() {
                Some(sparse) => sparse.map(bytes),
                None => Ok(()),
            }
        };
        let elements_required = self.cache.required_prefix_len(context_length) + suffix_length;
        let code_bytes = self.kv.layout.code_bytes(elements_required);
        let scale_bytes = self.kv.layout.scale_bytes(elements_required);
        map(self.kv.k_codes.as_mut(), code_bytes)?;
        map(self.kv.v_codes.as_mut(), code_bytes)?;
        map(self.kv.k_scales.as_mut(), scale_bytes)?;
        map(self.kv.v_scales.as_mut(), scale_bytes)
    }

    fn encode_accept(
        &mut self,
        batch_dim: &BatchTopology<B>,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let copies = self.cache.accept(batch_dim.kind(), accepted_indices);
        assert!(
            copies
                .iter()
                .try_fold(0, |written, copy| (copy.source >= written).then(|| written.max(copy.destination + 1)))
                .is_some(),
            "copy reads a row an earlier copy wrote"
        );

        if !copies.is_empty() {
            self.kv_cache_update.encode(
                self.kv.k_codes.as_mut(),
                Some(self.kv.v_codes.as_mut()),
                self.kv.k_scales.as_mut(),
                Some(self.kv.v_scales.as_mut()),
                &copies,
                copies.len() as u32,
                command_buffer,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/kv_cache_state_test.rs"]
mod tests;
