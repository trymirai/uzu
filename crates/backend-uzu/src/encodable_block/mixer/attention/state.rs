use std::any::Any;

use crate::{
    backends::common::{
        Backend, Buffer, Context, Encoder, Kernels, SparseBuffer,
        gpu_types::{Copy, ring::RingParams},
        kernel::KVCacheUpdateKernel,
    },
    data_type::DataType,
    encodable_block::mixer::{MixerState, attention::Attention},
};

pub enum AttentionStateType {
    Full {
        length: usize,
    },
    Ring {
        offset: usize,
        length: usize,
        max_length: usize,
    },
}

impl AttentionStateType {
    pub fn physical_prefix_length(&self) -> usize {
        match self {
            Self::Full {
                length,
            } => *length,
            Self::Ring {
                max_length,
                ..
            } => *max_length,
        }
    }

    pub fn ring_params(&self) -> Option<RingParams> {
        let Self::Ring {
            offset,
            length,
            max_length: _,
        } = self
        else {
            return None;
        };

        Some(RingParams {
            ring_offset: *offset as u32,
            ring_length: *length as u32,
        })
    }
}

pub struct AttentionState<B: Backend> {
    pub cur_context_length: usize,
    pub elements_prepared: usize,
    pub element_dim: usize,
    pub data_type: DataType,
    pub state_type: AttentionStateType,
    pub is_sparse: bool,
    pub keys: Box<dyn Buffer<Backend = B>>,
    pub values: Box<dyn Buffer<Backend = B>>,
    pub kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
}

impl<B: Backend> AttentionState<B> {
    pub fn create_empty(
        attention: &Attention<B>,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        if let Some(max_context_length) = max_context_length {
            assert!(
                attention.max_rope_length.is_none_or(|max_rope_length| max_context_length <= max_rope_length),
                "Attention state max_prefix_elements overflows RoPE"
            );
        }

        let data_type = attention.data_type;

        let max_prefix_elements = if attention.is_causal
            && let Some(sliding_window_size) = attention.sliding_window_size
        {
            sliding_window_size
        } else if let Some(max_context_length) = max_context_length {
            max_context_length
        } else {
            attention
                .max_rope_length
                .expect("Cannot create full attention state with unlimited length for with no RoPE")
        };

        let state_type = if attention.is_causal && attention.sliding_window_size.is_some() {
            AttentionStateType::Ring {
                offset: 0,
                length: 0,
                max_length: max_prefix_elements,
            }
        } else {
            AttentionStateType::Full {
                length: 0,
            }
        };

        Self::create_empty_with_type(
            data_type,
            attention.num_kv_heads.unwrap(),
            attention.head_dim,
            max_prefix_elements,
            state_type,
            context,
        )
    }

    fn create_empty_with_type(
        data_type: DataType,
        num_kv_heads: usize,
        head_dim: usize,
        max_prefix_elements: usize,
        state_type: AttentionStateType,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let suffix_capacity = 1024; // TODO: remove hardcoded suffix capacity
        let max_elements = max_prefix_elements + suffix_capacity;
        let element_size = num_kv_heads * head_dim;
        let kv_buffer_bytes = max_elements * element_size * data_type.size_in_bytes();

        let is_ring = matches!(state_type, AttentionStateType::Ring { .. });
        let is_sparse = !is_ring && context.sparse_buffers_supported();

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
            cur_context_length: 0,
            elements_prepared: 0,
            element_dim: element_size,
            data_type,
            state_type,
            is_sparse,
            keys,
            values,
            kv_cache_update,
        })
    }

    #[allow(dead_code)]
    pub(super) fn append_full(
        &mut self,
        length: usize,
    ) {
        let AttentionStateType::Full {
            length: state_length,
        } = &mut self.state_type
        else {
            panic!("append_full requires full attention state");
        };
        *state_length += length;
        self.cur_context_length += length;
    }
}

impl<B: Backend> MixerState<B> for AttentionState<B> {
    fn prepare(
        &mut self,
        context_length: usize,
        suffix_length: usize,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        if !self.is_sparse {
            return Ok(());
        }

        let suffix_capacity = 1024; // TODO: remove hardcoded suffix capacity
        assert!(suffix_length <= suffix_capacity, "attention suffix length exceeds hardcoded capacity");
        let elements_required = context_length + suffix_length;
        let bytes_required = elements_required * self.element_dim * self.data_type.size_in_bytes();
        let bytes_prepared = self.elements_prepared * self.element_dim * self.data_type.size_in_bytes();

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
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        assert!(accepted_indices.is_sorted_by(|a, b| a < b), "invalid accepted indicies");

        let copies = match &mut self.state_type {
            AttentionStateType::Full {
                length,
            } => {
                let copies = accepted_indices
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(index, accepted_index)| index != accepted_index)
                    .map(|(index, accepted_index)| Copy {
                        source: (*length + accepted_index) as u32,
                        destination: (*length + index) as u32,
                    })
                    .collect::<Vec<Copy>>();

                *length += accepted_indices.len();

                copies
            },
            AttentionStateType::Ring {
                offset,
                length,
                max_length,
            } => {
                let mut copies = Vec::new();
                for accepted_index in accepted_indices {
                    copies.push(Copy {
                        source: (*max_length + accepted_index) as u32,
                        destination: ((*offset + *length) % *max_length) as u32,
                    });

                    if length < max_length {
                        *length += 1;
                    } else {
                        *offset = (*offset + 1) % *max_length;
                    }
                }
                copies
            },
        };

        for copies_chunk in copies.chunks(B::MAX_INLINE_BYTES / size_of::<Copy>()) {
            self.kv_cache_update.encode(
                self.keys.as_mut(),
                self.values.as_mut(),
                copies_chunk,
                copies_chunk.len() as u32,
                self.element_dim as u32,
                encoder,
            );
        }

        self.cur_context_length += accepted_indices.len();

        Ok(())
    }
}
