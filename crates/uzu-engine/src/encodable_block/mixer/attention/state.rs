use crate::{
    backends::common::{
        Backend, CommandBuffer,
        gpu_types::ring::RingParams,
        kernel::{AttentionKernel, KVFormat},
    },
    encodable_block::mixer::{
        MixerState,
        attention::{Attention, kv_storage::KVStorage},
    },
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

    fn advance(
        &mut self,
        len_accepted: u32,
    ) {
        match self {
            Self::Full {
                length,
            } => *length += len_accepted,
            Self::Ring {
                offset,
                length,
                capacity,
            } => {
                let len_accepted = len_accepted;
                let grown = len_accepted.min(*capacity - *length);
                *length += grown;
                *offset = (*offset + len_accepted - grown) % *capacity;
            },
        }
    }

    fn ring_capacity(&self) -> u32 {
        match self {
            Self::Full {
                ..
            } => 0,
            Self::Ring {
                capacity,
                ..
            } => *capacity,
        }
    }

    fn ring_write_start(&self) -> u32 {
        match self {
            Self::Full {
                ..
            } => 0,
            Self::Ring {
                offset,
                length,
                capacity,
            } => (*offset + *length) % *capacity,
        }
    }
}

pub struct AttentionState<B: Backend> {
    cache: KVCacheState,
    pub kv: KVStorage<B>,
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

        let format = if attention.kernel.supports_int8() {
            KVFormat::INT8
        } else {
            KVFormat::BF16
        };
        let kv = KVStorage::create_empty(
            format,
            attention.num_kv_heads.unwrap(),
            attention.head_dim,
            max_prefix_elements + ATTENTION_SUFFIX_CAPACITY,
            attention.data_type,
            context,
        )?;

        Ok(Self {
            cache,
            kv,
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

        self.kv.prepare(self.cache.required_prefix_len(context_length) + suffix_length)
    }

    fn encode_accept_all(
        &mut self,
        _len_accepted: u32,
        _command_buffer: &mut <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <B as Backend>::Error> {
        // every suffix row already sits at its committed position; the counter
        // advance happens in `commit_accept`
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &B::GlobalGpuBuffer,
        command_buffer: &mut <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <B as Backend>::Error> {
        let suffix_base = self.cache.view().prefix_len();
        self.kv.encode_compact(
            accepted_indices,
            suffix_base,
            self.cache.ring_write_start(),
            self.cache.ring_capacity(),
            command_buffer,
        );
        Ok(())
    }

    fn commit_accept(
        &mut self,
        len_accepted: u32,
    ) {
        self.cache.advance(len_accepted);
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/kv_cache_state_test.rs"]
mod tests;
