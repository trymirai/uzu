//! Zero-copy thread-safe snapshot of ForwardPassState for parallel encoding.
//!
//! Metal buffers are just pointers with reference counting - sharing them is cheap.
//! Shape data uses Arc to avoid copying.

use std::{collections::HashMap, sync::Arc};

use metal::Buffer;

use super::ArrayId;

/// Buffer pointer with metadata - zero allocation after initial creation
pub struct FrozenBuffer {
    /// Raw Metal buffer pointer (GPU memory, ref-counted by Metal)
    buffer: Buffer,
    pub num_elements: usize,
    /// Shape is shared via Arc - no copy when FrozenState is cloned/shared
    pub shape: Arc<[usize]>,
}

impl FrozenBuffer {
    pub fn new(buffer: Buffer, num_elements: usize, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            num_elements,
            shape: shape.into(),
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

// Metal buffers are thread-safe for read access (GPU memory)
unsafe impl Send for FrozenBuffer {}
unsafe impl Sync for FrozenBuffer {}

/// Zero-copy frozen state for parallel encoding.
/// All data is shared via Arc - cloning is O(1).
pub struct FrozenState {
    /// All buffers indexed by ArrayId (shared, not copied)
    buffers: Arc<HashMap<ArrayId, FrozenBuffer>>,

    /// Active suffix length for this pass
    pub active_suffix_length: usize,

    /// Segment prefix length
    pub segment_prefix_length: usize,

    /// Whether this is a prefill pass
    pub is_prefilling: bool,

    /// Maximum sequence length from cache
    pub max_sequence_length: usize,
}

impl Clone for FrozenState {
    fn clone(&self) -> Self {
        Self {
            buffers: self.buffers.clone(), // Arc clone = pointer copy
            active_suffix_length: self.active_suffix_length,
            segment_prefix_length: self.segment_prefix_length,
            is_prefilling: self.is_prefilling,
            max_sequence_length: self.max_sequence_length,
        }
    }
}

impl FrozenState {
    pub fn new(
        buffers: HashMap<ArrayId, FrozenBuffer>,
        active_suffix_length: usize,
        segment_prefix_length: usize,
        is_prefilling: bool,
        max_sequence_length: usize,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
            active_suffix_length,
            segment_prefix_length,
            is_prefilling,
            max_sequence_length,
        }
    }

    /// Get a buffer by ArrayId. Panics if not found.
    #[inline]
    pub fn buffer(&self, id: &ArrayId) -> &Buffer {
        self.buffers
            .get(id)
            .map(|fb| fb.buffer())
            .unwrap_or_else(|| panic!("Buffer not found for {:?}", id))
    }

    /// Get frozen buffer with metadata. Panics if not found.
    #[inline]
    pub fn frozen_buffer(&self, id: &ArrayId) -> &FrozenBuffer {
        self.buffers
            .get(id)
            .unwrap_or_else(|| panic!("Buffer not found for {:?}", id))
    }

    /// Get a buffer by ArrayId, returns None if not found.
    #[inline]
    pub fn try_buffer(&self, id: &ArrayId) -> Option<&Buffer> {
        self.buffers.get(id).map(|fb| fb.buffer())
    }

    /// Check if a buffer exists
    #[inline]
    pub fn has_buffer(&self, id: &ArrayId) -> bool {
        self.buffers.contains_key(id)
    }

    /// Get number of elements for a buffer
    #[inline]
    pub fn num_elements(&self, id: &ArrayId) -> usize {
        self.frozen_buffer(id).num_elements
    }

    /// Get shape for a buffer
    #[inline]
    pub fn shape(&self, id: &ArrayId) -> &[usize] {
        &self.frozen_buffer(id).shape
    }

    /// Get the number of buffers in this frozen state
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

// FrozenState is Send + Sync because all its contents are thread-safe
unsafe impl Send for FrozenState {}
unsafe impl Sync for FrozenState {}

