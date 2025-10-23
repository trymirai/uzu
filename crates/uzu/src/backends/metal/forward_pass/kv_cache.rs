use std::{cell::RefCell, collections::HashMap};

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    super::{MTLContext, MetalArray},
    model_shape::ModelShape,
};
use crate::{
    DeviceContext,
    array::Array,
    backends::metal::kernel::{KVCacheUpdate, kv_cache_update::KVLayerData},
};

type ArrayCell = RefCell<MetalArray>;

#[derive(Clone, Debug)]
pub enum KVCacheLayerState {
    Full {
        // Prefix length so far (number of tokens in the prefix)
        prefix_len: usize,
    },
    Windowed {
        // Start of the ring buffer (oldest element index)
        ring_offset: usize,
        // Current logical length of the window (<= window_length)
        ring_length: usize,
        window_length: usize,
    },
}

pub const INVALID_POSITION: usize = i32::MAX as usize;

pub struct KVCacheLayer {
    pub state: KVCacheLayerState,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: ArrayCell,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: ArrayCell,

    pub prefix_token_positions: Vec<usize>,
    pub max_suffix_length: usize,
}

impl KVCacheLayer {
    pub fn prefix_segment_length(&self) -> usize {
        match &self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => *prefix_len,
            KVCacheLayerState::Windowed {
                window_length,
                ..
            } => *window_length,
        }
    }

    pub fn projected_segment_prefix_length(
        &self,
        projection_step: usize,
    ) -> usize {
        match &self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => *prefix_len + projection_step,
            KVCacheLayerState::Windowed {
                window_length,
                ..
            } => *window_length,
        }
    }

    pub fn is_sliding_window(&self) -> bool {
        matches!(self.state, KVCacheLayerState::Windowed { .. })
    }

    pub fn window_length(&self) -> Option<usize> {
        match &self.state {
            KVCacheLayerState::Full {
                ..
            } => None,
            KVCacheLayerState::Windowed {
                window_length,
                ..
            } => Some(*window_length),
        }
    }

    pub fn fill_attention_bias(
        &self,
        dst: &mut MetalArray,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        context: &MTLContext,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        let prefix_segment_length = self.prefix_segment_length();

        context.fill_attention_bias(
            dst,
            suffix_length,
            prefix_segment_length,
            |row_index, column_index| {
                if let Some(bias_fn) = external_bias_fn {
                    bias_fn(row_index, column_index)
                } else {
                    let result = self.bias_should_be_neg_inf(
                        row_index,
                        column_index,
                        suffix_token_positions,
                    );
                    result
                }
            },
        );
    }

    pub fn bias_should_be_neg_inf(
        &self,
        row_index: usize,
        column_index: usize,
        suffix_token_positions: &[usize],
    ) -> bool {
        let query_position = suffix_token_positions[row_index];
        if query_position == INVALID_POSITION {
            return true;
        }

        let key_position = if column_index >= self.prefix_segment_length() {
            suffix_token_positions[column_index - self.prefix_segment_length()]
        } else {
            match &self.state {
                KVCacheLayerState::Full {
                    ..
                } => column_index,
                KVCacheLayerState::Windowed {
                    ..
                } => self.prefix_token_positions[column_index],
            }
        };

        if key_position == INVALID_POSITION {
            return true;
        }

        if query_position < key_position {
            return true;
        }

        match &self.state {
            KVCacheLayerState::Windowed {
                window_length,
                ..
            } => query_position >= key_position + window_length,
            _ => false,
        }
    }

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        command_buffer: &MPSCommandBuffer,
        kv_cache_update: &KVCacheUpdate,
    ) {
        assert!(
            !accepted_suffix_indices.is_empty(),
            "update_after_acceptance requires at least one accepted suffix index"
        );

        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                // Absolute positions of the *source* rows (still in suffix part).
                let source_indices: Vec<usize> = accepted_suffix_indices
                    .iter()
                    .map(|i| i + *prefix_len)
                    .collect();

                // Absolute positions of the *destination* rows in the prefix.
                let destination_indices: Vec<usize> = (*prefix_len
                    ..*prefix_len + accepted_suffix_indices.len())
                    .collect();

                self.scatter_if_required(
                    &source_indices,
                    &destination_indices,
                    command_buffer,
                    kv_cache_update,
                );
            },

            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let source_indices: Vec<usize> = accepted_suffix_indices
                    .iter()
                    .map(|i| i + *window_length)
                    .collect();

                // Consecutive slots starting at current ring_offset.
                let mut destination_indices =
                    Vec::with_capacity(accepted_suffix_indices.len());

                for i in 0..accepted_suffix_indices.len() {
                    destination_indices.push(
                        (*ring_length + *ring_offset + i) % *window_length,
                    );
                }

                self.scatter_if_required(
                    &source_indices,
                    &destination_indices,
                    command_buffer,
                    kv_cache_update,
                );
            },
        }
    }

    fn scatter_if_required(
        &self,
        source_indices: &[usize],
        destination_indices: &[usize],
        command_buffer: &MPSCommandBuffer,
        kv_cache_update: &KVCacheUpdate,
    ) {
        if source_indices == destination_indices {
            return;
        }

        let root_cb = command_buffer.root_command_buffer().to_owned();
        let key_buffer = {
            let mut k = self.keys.borrow_mut();
            unsafe { k.mtl_buffer() }.clone()
        };
        let value_buffer = {
            let mut v = self.values.borrow_mut();
            unsafe { v.mtl_buffer() }.clone()
        };

        let k_shape = self.keys.borrow().shape().to_vec();
        let v_shape = self.values.borrow().shape().to_vec();

        let layer_data = KVLayerData {
            key_buffer,
            key_shape: [k_shape[0], k_shape[1], k_shape[2]],
            value_buffer,
            value_shape: [v_shape[0], v_shape[1], v_shape[2]],
        };

        let _ = kv_cache_update.encode(
            &[layer_data],
            source_indices,
            destination_indices,
            &root_cb,
        );
    }

    pub fn register_accepted_tokens(
        &mut self,
        token_positions: &[usize],
    ) {
        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                self.prefix_token_positions.extend_from_slice(token_positions);
                *prefix_len = self.prefix_token_positions.len();
            },
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                for &token_pos in token_positions {
                    if *ring_length < *window_length {
                        let dst =
                            (*ring_offset + *ring_length) % *window_length;

                        self.prefix_token_positions[dst] = token_pos;
                        *ring_length += 1;
                    } else {
                        self.prefix_token_positions[*ring_offset] = token_pos;

                        *ring_offset = (*ring_offset + 1) % *window_length;
                    }
                }
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
                let window_length = model_shape.sliding_window_length_per_layer
                    [layer_idx]
                    .filter(|&window_size| window_size < total_context_length);

                let state = if let Some(w) = window_length {
                    KVCacheLayerState::Windowed {
                        ring_offset: 0,
                        ring_length: 0,
                        window_length: w,
                    }
                } else {
                    KVCacheLayerState::Full {
                        prefix_len: 0,
                    }
                };

                KVCacheLayer {
                    state: state.clone(),
                    keys: RefCell::new(
                        context.array(&shape, model_shape.kv_cache_data_type()),
                    ),
                    values: RefCell::new(
                        context.array(&shape, model_shape.kv_cache_data_type()),
                    ),
                    prefix_token_positions: match &state {
                        KVCacheLayerState::Full {
                            ..
                        } => Vec::with_capacity(max_prefix_length),
                        KVCacheLayerState::Windowed {
                            window_length,
                            ..
                        } => (0..*window_length)
                            .map(|_| INVALID_POSITION)
                            .collect(),
                    },
                    max_suffix_length: max_suffix_length,
                }
            })
            .collect();

        Self {
            max_suffix_length,
            max_prefix_length,
            data,
        }
    }

    pub fn clear(&mut self) {
        for layer in self.data.iter_mut() {
            match &mut layer.state {
                KVCacheLayerState::Full {
                    prefix_len,
                } => {
                    *prefix_len = 0;
                },
                KVCacheLayerState::Windowed {
                    ring_offset,
                    ring_length,
                    ..
                } => {
                    *ring_offset = 0;
                    *ring_length = 0;
                },
            }
            layer.prefix_token_positions.fill(INVALID_POSITION);
        }
    }

    pub fn max_suffix_length(&self) -> usize {
        self.max_suffix_length
    }

    pub fn max_prefix_length(&self) -> usize {
        self.max_prefix_length
    }

    pub fn fill_attention_bias(
        &self,
        dst: &mut HashMap<Option<usize>, MetalArray>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        context: &MTLContext,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        for layer in self.data.iter() {
            let key: Option<usize> = match &layer.state {
                KVCacheLayerState::Full {
                    ..
                } => None,
                KVCacheLayerState::Windowed {
                    window_length,
                    ..
                } => Some(*window_length),
            };

            if let Some(array) = dst.get_mut(&key) {
                layer.fill_attention_bias(
                    array,
                    suffix_token_positions,
                    suffix_length,
                    context,
                    external_bias_fn,
                );
            }
        }
    }

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        command_buffer: &MPSCommandBuffer,
        kv_cache_update: &KVCacheUpdate,
    ) {
        for layer in self.data.iter_mut() {
            layer.update_after_acceptance(
                accepted_suffix_indices,
                command_buffer,
                kv_cache_update,
            );
        }
    }

    pub fn register_accepted_tokens(
        &mut self,
        token_positions: &[usize],
    ) {
        for layer in self.data.iter_mut() {
            layer.register_accepted_tokens(token_positions);
        }
    }

    pub fn clone_with_prefix_len(
        &self,
        context: &MTLContext,
        prefix_len: usize,
    ) -> Self {
        let new_total_len = prefix_len + self.max_suffix_length;
        let data: Box<[KVCacheLayer]> = self
            .data
            .iter()
            .map(|layer| {
                let shape = layer.keys.borrow().shape().to_vec();
                let num_groups = shape[0];
                let head_dim = shape[2];
                let dtype = layer.keys.borrow().data_type();
                let new_shape = [num_groups, new_total_len, head_dim];

                let mut new_keys = context.array(&new_shape, dtype);
                let mut new_values = context.array(&new_shape, dtype);

                let mut copy_rows = layer.prefix_segment_length();
                if let Some(window_length) = layer.window_length() {
                    copy_rows = std::cmp::min(copy_rows, window_length);
                }

                if copy_rows > 0 {
                    new_keys.copy_slice(
                        &layer.keys.borrow(),
                        1,
                        0..copy_rows,
                        0,
                    );
                    new_values.copy_slice(
                        &layer.values.borrow(),
                        1,
                        0..copy_rows,
                        0,
                    );
                }

                KVCacheLayer {
                    state: layer.state.clone(),
                    keys: RefCell::new(new_keys),
                    values: RefCell::new(new_values),
                    prefix_token_positions: layer
                        .prefix_token_positions
                        .clone(),
                    max_suffix_length: layer.max_suffix_length,
                }
            })
            .collect();

        Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: prefix_len,
            data,
        }
    }
}
