use std::cell::Cell;

use bytemuck::fill_zeroes;

use crate::{array::ArrayCell, backends::common::Backend};

#[derive(Debug)]
pub struct ShortConvLayer<B: Backend> {
    pub conv_state: ArrayCell<B>,
    /// Per-token post-state written during speculative runs.
    ///
    /// Shape: [max_suffix_length, model_dim, state_stride]
    pub suffix_state: ArrayCell<B>,
    /// Start index (in the suffix batch) for which `suffix_state` contains
    /// valid post-states for this layer.
    pub suffix_state_valid_start: Cell<usize>,
    /// Number of consecutive tokens (from `suffix_state_valid_start`) for which
    /// `suffix_state` contains valid post-states for this layer.
    pub suffix_state_valid_len: Cell<usize>,
}

impl<B: Backend> ShortConvLayer<B> {
    pub fn zero(&self) {
        {
            let mut conv = self.conv_state.borrow_mut();
            fill_zeroes(conv.as_bytes_mut());
        }
        {
            let mut suffix = self.suffix_state.borrow_mut();
            fill_zeroes(suffix.as_bytes_mut());
        }
        self.clear_suffix_state_valid_range();
    }

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
        &self,
        commit_index: usize,
    ) {
        let start = self.suffix_state_valid_start.get();
        let len = self.suffix_state_valid_len.get();
        if len == 0 {
            return;
        }
        if commit_index < start || commit_index >= start + len {
            return;
        }

        let mut conv = self.conv_state.borrow_mut();
        let suffix = self.suffix_state.borrow();

        assert_eq!(
            conv.data_type(),
            suffix.data_type(),
            "ShortConv conv_state / suffix_state dtype mismatch"
        );

        let [model_dim, state_stride] = {
            let shape = conv.shape();
            assert_eq!(
                shape.len(),
                2,
                "ShortConv conv_state expected 2-D [model_dim, state_stride], got {:?}",
                shape
            );
            [shape[0], shape[1]]
        };

        let suffix_shape = suffix.shape();
        assert!(
            suffix_shape.len() == 3
                && suffix_shape[1] == model_dim
                && suffix_shape[2] == state_stride,
            "ShortConv suffix_state expected 3-D [suffix_len, model_dim, state_stride], got {:?}",
            suffix_shape
        );
        assert!(
            commit_index < suffix_shape[0],
            "ShortConv commit_index {} out of suffix_state range {}",
            commit_index,
            suffix_shape[0]
        );

        let elem_bytes = conv.data_type().size_in_bytes();
        let bytes_per_token = model_dim
            .saturating_mul(state_stride)
            .saturating_mul(elem_bytes);
        let src_start = commit_index.saturating_mul(bytes_per_token);
        let src_end = src_start.saturating_add(bytes_per_token);

        let src = &suffix.as_bytes()[src_start..src_end];
        let dst = conv.as_bytes_mut();
        dst.copy_from_slice(src);
    }
}
