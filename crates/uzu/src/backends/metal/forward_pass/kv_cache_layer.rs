use std::cell::RefCell;

use super::super::{MTLContext, MetalArray};
use crate::{
    array::ArrayContextExt,
    backends::metal::{
        MTLCommandBuffer, ProtocolObject, Retained,
        kernel::{KVCacheUpdate, kv_cache_update::KVLayerData},
    },
    utils::attention::fill_attention_bias,
};

pub type ArrayCell = RefCell<MetalArray>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBiasUpdate {
    pub key: Option<usize>,
    pub unmask_col: i32,
    pub mask_col: i32,
}

#[derive(Clone)]
pub enum KVSlice {
    Full {
        base_prefix_len: usize,
        base_positions_len: usize,
        positions: Vec<usize>,
    },
    Window {
        window_length: usize,
        base_ring_offset: usize,
        base_ring_length: usize,
        slots: Vec<usize>,
        positions: Vec<usize>, // per slot
        keys: MetalArray,      // [num_groups, slots.len(), head_dim]
        values: MetalArray,    // [num_groups, slots.len(), head_dim]
    },
}

#[derive(Clone, Debug)]
pub enum KVCacheLayerState {
    Full {
        /// Prefix length so far (number of tokens in the prefix)
        prefix_len: usize,
    },
    Windowed {
        /// Start of the ring buffer (oldest element index)
        ring_offset: usize,
        /// Current logical length of the window (<= window_length)
        ring_length: usize,
        window_length: usize,
    },
}

pub const INVALID_POSITION: usize = i32::MAX as usize;

#[derive(Debug)]
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
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        let prefix_segment_length = self.prefix_segment_length();

        fill_attention_bias(
            dst,
            suffix_length,
            prefix_segment_length,
            |row_index, column_index| {
                if let Some(bias_fn) = external_bias_fn {
                    bias_fn(row_index, column_index)
                } else {
                    self.bias_should_be_neg_inf(
                        row_index,
                        column_index,
                        suffix_token_positions,
                    )
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
        suffix_start: Option<usize>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        kv_cache_update: &KVCacheUpdate,
    ) {
        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                if accepted_suffix_indices.is_empty() {
                    return;
                }

                // Absolute positions of the *source* rows.
                let source_indices: Vec<usize> = accepted_suffix_indices
                    .iter()
                    .map(|i| i + suffix_start.unwrap_or(*prefix_len))
                    .collect();

                // Absolute positions of the *destination* rows.
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
                let suffix_indices: Vec<usize> =
                    if accepted_suffix_indices.is_empty() {
                        vec![0]
                    } else {
                        accepted_suffix_indices.to_vec()
                    };

                let source_indices: Vec<usize> =
                    suffix_indices.iter().map(|i| i + *window_length).collect();

                let mut destination_indices =
                    Vec::with_capacity(suffix_indices.len());

                for i in 0..suffix_indices.len() {
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
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        kv_cache_update: &KVCacheUpdate,
    ) {
        if source_indices == destination_indices {
            return;
        }

        let key_buffer = self.keys.borrow().buffer().clone();
        let value_buffer = self.values.borrow().buffer().clone();

        let k_shape = self.keys.borrow().shape().to_vec();
        let v_shape = self.values.borrow().shape().to_vec();

        let layer_data = KVLayerData {
            key_buffer,
            key_shape: [k_shape[0], k_shape[1], k_shape[2]],
            value_buffer,
            value_shape: [v_shape[0], v_shape[1], v_shape[2]],
        };

        let cmd_buf = command_buffer.clone();
        let _ = kv_cache_update.encode(
            &[layer_data],
            source_indices,
            destination_indices,
            &cmd_buf,
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

    pub fn attention_bias_update_after_acceptance(
        &self,
        accepted_len: usize,
    ) -> Option<AttentionBiasUpdate> {
        if accepted_len != 1 {
            return None;
        }

        match self.state {
            KVCacheLayerState::Full {
                ..
            } => None,
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let newest_slot = (ring_length > 0)
                    .then_some(
                        (ring_offset + ring_length + window_length - 1)
                            % window_length,
                    )
                    .unwrap_or(0);
                let unmask_col = (ring_length > 0)
                    .then_some(newest_slot as i32)
                    .unwrap_or(-1);
                let mask_col = (ring_length == window_length)
                    .then_some(ring_offset as i32)
                    .unwrap_or(-1);

                Some(AttentionBiasUpdate {
                    key: Some(window_length),
                    unmask_col,
                    mask_col,
                })
            },
        }
    }

    pub fn slice(
        &self,
        context: &MTLContext,
        range: std::ops::Range<usize>,
    ) -> Option<KVSlice> {
        match self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => Some(KVSlice::Full {
                base_prefix_len: prefix_len,
                base_positions_len: self.prefix_token_positions.len(),
                positions: self.prefix_token_positions.clone(),
            }),
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let len = range.end.saturating_sub(range.start);
                if len == 0 || len > window_length {
                    return None;
                }
                let keys = self.keys.borrow();
                let values = self.values.borrow();
                let shape = keys.shape();
                let num_groups = shape[0];
                let head_dim = shape[2];
                let dtype = keys.data_type();

                let slice_shape = [num_groups, len, head_dim];
                let mut slice_keys = context.create_array(
                    &slice_shape,
                    dtype,
                    "kv_cache_layer_slice_keys",
                );
                let mut slice_values = context.create_array(
                    &slice_shape,
                    dtype,
                    "kv_cache_layer_slice_values",
                );

                let slots: Vec<usize> = (range.start..range.end)
                    .enumerate()
                    .map(|(offset, _)| {
                        let x = ring_length + offset;
                        if x < window_length {
                            (ring_offset + x) % window_length
                        } else {
                            (ring_offset + (x - window_length)) % window_length
                        }
                    })
                    .collect();

                let positions: Vec<usize> = slots
                    .iter()
                    .map(|&s| self.prefix_token_positions[s])
                    .collect();

                for (i, &slot) in slots.iter().enumerate() {
                    slice_keys.copy_slice(&keys, 1, slot..slot + 1, i);
                    slice_values.copy_slice(&values, 1, slot..slot + 1, i);
                }

                Some(KVSlice::Window {
                    window_length,
                    base_ring_offset: ring_offset,
                    base_ring_length: ring_length,
                    slots,
                    positions,
                    keys: slice_keys,
                    values: slice_values,
                })
            },
        }
    }

    pub fn apply_slice(
        &mut self,
        slice: &KVSlice,
        range: Option<std::ops::Range<usize>>,
    ) {
        match (slice, &mut self.state) {
            (
                KVSlice::Full {
                    base_prefix_len,
                    base_positions_len,
                    positions,
                },
                KVCacheLayerState::Full {
                    prefix_len,
                },
            ) => match range {
                None => {
                    *prefix_len = *base_prefix_len;
                    self.prefix_token_positions.clone_from(positions);
                    self.prefix_token_positions.truncate(*base_positions_len);
                },
                Some(r) => {
                    let accepted = r.start;
                    *prefix_len = base_prefix_len.saturating_add(accepted);
                    let keep_positions =
                        base_positions_len.saturating_add(accepted);
                    self.prefix_token_positions.truncate(keep_positions);
                },
            },
            (
                KVSlice::Window {
                    window_length,
                    base_ring_offset,
                    base_ring_length,
                    slots,
                    positions,
                    keys,
                    values,
                },
                KVCacheLayerState::Windowed {
                    ring_offset,
                    ring_length,
                    window_length: w_len,
                },
            ) => {
                *w_len = *window_length;
                match range {
                    None => {
                        *ring_offset = *base_ring_offset;
                        *ring_length = *base_ring_length;

                        for (i, &slot) in slots.iter().enumerate() {
                            self.prefix_token_positions[slot] = positions[i];
                        }

                        let mut dst_keys = self.keys.borrow_mut();
                        let mut dst_values = self.values.borrow_mut();
                        for (i, &slot) in slots.iter().enumerate() {
                            dst_keys.copy_slice(keys, 1, i..i + 1, slot);
                            dst_values.copy_slice(values, 1, i..i + 1, slot);
                        }
                    },
                    Some(r) => {
                        let len = r.end.saturating_sub(r.start);
                        if len == 0 {
                            return;
                        }

                        let accepted = r.start;
                        let base_len = *base_ring_length;
                        let w = *window_length;

                        let (new_offset, new_len) = if base_len < w {
                            let after = base_len.saturating_add(accepted);
                            if after <= w {
                                (*base_ring_offset, after)
                            } else {
                                let overflow = after - w;
                                ((base_ring_offset + overflow) % w, w)
                            }
                        } else {
                            ((base_ring_offset + accepted) % w, w)
                        };

                        *ring_offset = new_offset;
                        *ring_length = new_len;

                        let mut dst_keys = self.keys.borrow_mut();
                        let mut dst_values = self.values.borrow_mut();

                        for (i, &slot) in slots[r.clone()].iter().enumerate() {
                            let src_i = r.start + i;
                            self.prefix_token_positions[slot] =
                                positions[src_i];
                            dst_keys.copy_slice(
                                keys,
                                1,
                                src_i..src_i + 1,
                                slot,
                            );
                            dst_values.copy_slice(
                                values,
                                1,
                                src_i..src_i + 1,
                                slot,
                            );
                        }
                    },
                }
            },
            _ => {},
        }
    }
}
