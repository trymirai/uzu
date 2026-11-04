use std::any::Any;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Context, DeviceCapabilities, GpuBuffer, Kernels, SparseBuffer,
        gpu_types::{kv_cache_update::Copy, ring::RingParams},
        kernel::{AttentionKernel, KVCacheUpdateKernel, kv_layout::KVLayout},
    },
    encodable_block::mixer::{MixerState, attention::Attention},
};

pub const ATTENTION_SUFFIX_CAPACITY: u32 = 1024; // TODO: remove hardcoded suffix capacity

#[derive(Clone, Copy)]
pub struct KVCacheView {
    prefix_len: u32,
    ring: Option<RingParams>,
}

impl KVCacheView {
    pub fn full(prefix_len: u32) -> Self {
        Self {
            prefix_len,
            ring: None,
        }
    }

    pub fn ring(
        prefix_len: u32,
        offset: u32,
    ) -> Self {
        Self {
            prefix_len,
            ring: Some(RingParams {
                ring_offset: offset,
                ring_length: prefix_len,
            }),
        }
    }

    pub fn prefix_len(self) -> u32 {
        self.prefix_len
    }

    pub fn ring_params(self) -> Option<RingParams> {
        self.ring
    }
}

enum KVCacheState {
    Full {
        length: u32,
    },
    Ring {
        offset: u32,
        length: u32,
        capacity: u32,
    },
}

impl KVCacheState {
    fn full() -> Self {
        Self::Full {
            length: 0,
        }
    }

    fn ring(capacity: u32) -> Self {
        assert!(capacity > 0, "zero ring capacity");
        Self::Ring {
            offset: 0,
            length: 0,
            capacity,
        }
    }

    pub fn view(&self) -> KVCacheView {
        match self {
            Self::Full {
                length,
            } => KVCacheView::full(*length),
            Self::Ring {
                offset,
                length,
                capacity: _,
            } => KVCacheView::ring(*length, *offset),
        }
    }

    fn required_prefix_len(
        &self,
        context_length: u32,
    ) -> u32 {
        match self {
            Self::Full {
                ..
            } => context_length,
            Self::Ring {
                capacity,
                ..
            } => context_length.min(*capacity),
        }
    }

    fn accept(
        &mut self,
        accepted_indices: &[u32],
    ) -> Vec<Copy> {
        assert!(accepted_indices.is_sorted_by(|a, b| a < b), "unsorted accept");
        let suffix_base = self.view().prefix_len();

        match self {
            Self::Full {
                length,
            } => {
                let mut copies = Vec::with_capacity(accepted_indices.len());
                for (destination_index, accepted_index) in accepted_indices.iter().copied().enumerate() {
                    let source = suffix_base + accepted_index;
                    let destination = suffix_base + destination_index as u32;
                    if source != destination {
                        copies.push(Copy {
                            source,
                            destination,
                        });
                    }
                }
                *length += accepted_indices.len() as u32;
                copies
            },
            Self::Ring {
                offset,
                length,
                capacity,
            } => {
                let count = accepted_indices.len() as u32;
                let mut copies = Vec::with_capacity(accepted_indices.len().min(*capacity as usize));
                for (index, accepted_index) in accepted_indices.iter().copied().enumerate() {
                    let source = suffix_base + accepted_index;
                    let destination = (*offset + *length) % *capacity;
                    // A copy is dead when a later accept in the same batch overwrites its
                    // destination; emitting it would alias in a single kernel dispatch.
                    if index as u32 + *capacity >= count && source != destination {
                        copies.push(Copy {
                            source,
                            destination,
                        });
                    }

                    if length < capacity {
                        *length += 1;
                    } else {
                        *offset = (*offset + 1) % *capacity;
                    }
                }
                copies
            },
        }
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
    pub fn view(&self) -> KVCacheView {
        self.cache.view()
    }

    pub fn create_empty(
        attention: &Attention<B>,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        if let Some(max_context_length) = max_context_length {
            assert!(
                attention.max_rope_length.is_none_or(|max_rope_length| max_context_length <= max_rope_length),
                "context exceeds RoPE length"
            );
        }

        let max_prefix_elements = attention.ring_capacity.unwrap_or_else(|| {
            max_context_length
                .or(attention.max_rope_length)
                .expect("Cannot create full attention state with unlimited length for with no RoPE")
        });

        let cache = attention.ring_capacity.map_or_else(KVCacheState::full, KVCacheState::ring);

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
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let copies = self.cache.accept(accepted_indices);

        if !copies.is_empty() {
            self.kv_cache_update.encode(
                self.kv.k_codes.as_mut(),
                self.kv.v_codes.as_mut(),
                self.kv.k_scales.as_mut(),
                self.kv.v_scales.as_mut(),
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
