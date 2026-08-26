use std::any::Any;

use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, Context, DeviceCapabilities, Encoder, Kernels, SparseBuffer,
        gpu_types::{Copy, ring::RingParams},
        kernel::KVCacheUpdateKernel,
    },
    data_type::DataType,
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

pub struct AttentionState<B: Backend> {
    pub elements_prepared: u32,
    pub element_dim: u32,
    pub data_type: DataType,
    cache: KVCacheState,
    pub is_sparse: bool,
    pub keys: Box<dyn Buffer<Backend = B>>,
    pub values: Box<dyn Buffer<Backend = B>>,
    pub kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
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

        let data_type = attention.data_type;

        let max_prefix_elements = attention.ring_capacity.unwrap_or_else(|| {
            max_context_length
                .or(attention.max_rope_length)
                .expect("Cannot create full attention state with unlimited length for with no RoPE")
        });

        let cache = attention.ring_capacity.map_or_else(KVCacheState::full, KVCacheState::ring);

        let max_elements = max_prefix_elements + ATTENTION_SUFFIX_CAPACITY;
        let element_size = attention.num_kv_heads.unwrap() * attention.head_dim;
        let kv_buffer_bytes = size_for_shape(&[max_elements, element_size], data_type);

        let is_sparse = context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS);

        let (keys, values): (Box<dyn Buffer<Backend = B>>, Box<dyn Buffer<Backend = B>>) = if is_sparse {
            (
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
            )
        } else {
            (Box::new(context.create_buffer(kv_buffer_bytes)?), Box::new(context.create_buffer(kv_buffer_bytes)?))
        };

        let kv_cache_update = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)?;

        Ok(Self {
            elements_prepared: 0,
            element_dim: element_size,
            data_type,
            cache,
            is_sparse,
            keys,
            values,
            kv_cache_update,
        })
    }
}

impl<B: Backend> MixerState<B> for AttentionState<B> {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        if !self.is_sparse {
            return Ok(());
        }

        assert!(suffix_length <= ATTENTION_SUFFIX_CAPACITY, "suffix exceeds capacity");
        let elements_required = self.cache.required_prefix_len(context_length) + suffix_length;
        let bytes_required = size_for_shape(&[elements_required, self.element_dim], self.data_type);
        let bytes_prepared = size_for_shape(&[self.elements_prepared, self.element_dim], self.data_type);

        let keys = (self.keys.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();
        let values = (self.values.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();

        for buffer in [keys, values] {
            let buffer_page_size = buffer.page_size_bytes();
            let buffer_start_page = bytes_prepared.div_ceil(buffer_page_size);
            let buffer_end_page = bytes_required.div_ceil(buffer_page_size);

            if buffer_end_page > buffer_start_page {
                buffer.map(context, &(buffer_start_page..buffer_end_page))?;
            }
        }

        self.elements_prepared = elements_required;

        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let copies = self.cache.accept(accepted_indices);

        for copies_chunk in copies.chunks(B::MAX_INLINE_BYTES / size_of::<Copy>()) {
            self.kv_cache_update.encode(
                self.keys.as_mut(),
                self.values.as_mut(),
                copies_chunk,
                copies_chunk.len() as u32,
                self.element_dim,
                encoder,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/kv_cache_state_test.rs"]
mod tests;
