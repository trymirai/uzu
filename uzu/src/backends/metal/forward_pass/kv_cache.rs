use std::cell::RefCell;

use super::{
    super::{MTLContext, MetalArray},
    model_shape::ModelShape,
};
use crate::{Array, DeviceContext};

type ArrayCell = RefCell<MetalArray>;

pub struct KVCacheLayer {
    /// [] - scalar u64
    pub prefix_length_buffer: ArrayCell,
    /// [] - scalar u64
    pub ring_offset_buffer: ArrayCell,
    pub window_length: Option<usize>,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: ArrayCell,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: ArrayCell,
}

impl KVCacheLayer {
    pub fn prefix_length(&self) -> usize {
        let buffer_ref = self.prefix_length_buffer.borrow();
        *buffer_ref.item::<i64>().unwrap() as usize
    }

    pub fn set_prefix_length(
        &mut self,
        prefix_length: usize,
    ) {
        *self.prefix_length_buffer.borrow_mut().item_mut::<i64>().unwrap() =
            prefix_length as i64;
    }

    pub fn ring_offset(&self) -> usize {
        let buffer_ref = self.ring_offset_buffer.borrow();
        *buffer_ref.item::<i64>().unwrap() as usize
    }

    pub fn set_ring_offset(
        &mut self,
        ring_offset: usize,
    ) {
        *self.ring_offset_buffer.borrow_mut().item_mut::<i64>().unwrap() =
            ring_offset as i64;
    }

    pub fn effective_prefix_length(&self) -> usize {
        let total = self.prefix_length();
        match self.window_length {
            Some(window) => total.min(window),
            None => total,
        }
    }

    pub fn update_after_acceptance(
        &mut self,
        accepted_count: usize,
    ) {
        let old_prefix = self.prefix_length();
        let old_ring_offset = self.ring_offset();
        let new_prefix = old_prefix + accepted_count;

        match self.window_length {
            Some(window) => {
                let new_prefix_capped = new_prefix.min(window);

                if old_prefix < window {
                    // Still filling the window - no wrap yet
                    self.set_prefix_length(new_prefix_capped);
                    // ring_offset stays 0 until window is full
                } else {
                    // Window is full - ring buffer mode
                    let new_ring_offset =
                        (old_ring_offset + accepted_count) % window;
                    self.set_ring_offset(new_ring_offset);
                    self.set_prefix_length(window); // Always at capacity
                }
            },
            None => {
                self.set_prefix_length(new_prefix);
            },
        }
    }
}

pub struct KVCache {
    max_suffix_length: usize,
    max_prefix_length: usize,
    pub data: Box<[KVCacheLayer]>,
}

impl KVCache {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Self {
        let total_context_length = max_prefix_length + max_suffix_length;
        let data: Box<[KVCacheLayer]> = model_shape
            .kv_cache_layer_shapes(max_prefix_length, max_suffix_length)
            .enumerate()
            .map(|(layer_idx, shape)| {
                // If sliding window size equals context length, treat as no sliding window
                let window_length = model_shape.sliding_window_length_per_layer
                    [layer_idx]
                    .filter(|&window_size| window_size < total_context_length);

                KVCacheLayer {
                    prefix_length_buffer: RefCell::new(
                        context.scalar(0 as i64),
                    ),
                    ring_offset_buffer: RefCell::new(context.scalar(0 as i64)),
                    window_length,
                    keys: RefCell::new(
                        context.array(&shape, model_shape.kv_cache_data_type()),
                    ),
                    values: RefCell::new(
                        context.array(&shape, model_shape.kv_cache_data_type()),
                    ),
                }
            })
            .collect();

        Self {
            max_suffix_length,
            max_prefix_length,
            data,
        }
    }
    pub fn max_effective_prefix_length(&self) -> usize {
        self.data
            .iter()
            .map(|layer| layer.effective_prefix_length())
            .max()
            .unwrap_or(0)
    }

    pub fn effective_prefix_length(
        &self,
        layer_idx: usize,
    ) -> usize {
        self.data[layer_idx].effective_prefix_length()
    }

    pub fn set_prefix_length(
        &mut self,
        prefix_length: usize,
    ) {
        self.data
            .iter_mut()
            .for_each(|layer| layer.set_prefix_length(prefix_length));
    }

    pub fn max_suffix_length(&self) -> usize {
        self.max_suffix_length
    }

    pub fn max_prefix_length(&self) -> usize {
        self.max_prefix_length
    }
}
