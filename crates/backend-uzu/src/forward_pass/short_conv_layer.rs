use std::cell::Cell;

use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
};

pub struct ShortConvLayer<B: Backend> {
    pub conv_state: Allocation<B>,
    pub conv_shape: [usize; 2],
    /// Per-token post-state written during speculative runs.
    ///
    /// Shape: [max_suffix_length, model_dim, state_stride]
    pub suffix_state: Allocation<B>,
    pub suffix_shape: [usize; 3],
    pub data_type: DataType,
    /// Start index (in the suffix batch) for which `suffix_state` contains
    /// valid post-states for this layer.
    pub suffix_state_valid_start: Cell<usize>,
    /// Number of consecutive tokens (from `suffix_state_valid_start`) for which
    /// `suffix_state` contains valid post-states for this layer.
    pub suffix_state_valid_len: Cell<usize>,
}

impl<B: Backend> ShortConvLayer<B> {
    pub fn clear_suffix_state_valid_range(&self) {
        self.suffix_state_valid_start.set(0);
        self.suffix_state_valid_len.set(0);
    }

    pub fn set_suffix_state_valid_range(
        &self,
        start: usize,
        len: usize,
    ) {
        self.suffix_state_valid_start.set(start);
        self.suffix_state_valid_len.set(len);
    }

    pub fn commit_from_suffix_state_if_valid(
        &mut self,
        commit_index: usize,
        encoder: &mut Encoder<B>,
    ) {
        let start = self.suffix_state_valid_start.get();
        let len = self.suffix_state_valid_len.get();
        if len == 0 {
            return;
        }
        if commit_index < start || commit_index >= start + len {
            return;
        }

        let [model_dim, state_stride] = self.conv_shape;
        let [suffix_length, suffix_model_dim, suffix_stride] = self.suffix_shape;
        assert_eq!(suffix_model_dim, model_dim, "ShortConv suffix_state model_dim mismatch");
        assert_eq!(suffix_stride, state_stride, "ShortConv suffix_state stride mismatch");
        assert!(commit_index < suffix_length, "ShortConv commit_index {} out of range {}", commit_index, suffix_length);

        let elem_bytes = self.data_type.size_in_bytes();
        let bytes_per_token = model_dim.saturating_mul(state_stride).saturating_mul(elem_bytes);
        let src_start = commit_index.saturating_mul(bytes_per_token);
        encoder.encode_copy(
            &self.suffix_state,
            src_start..src_start + bytes_per_token,
            &mut self.conv_state,
            0..bytes_per_token,
        );
    }
}
