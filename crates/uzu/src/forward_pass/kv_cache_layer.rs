use std::cell::RefCell;

use crate::{Array, DeviceContext};

pub type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBiasUpdate {
    pub key: Option<usize>,
    pub unmask_col: i32,
    pub mask_col: i32,
}

pub enum KVSlice<C: DeviceContext> {
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
        positions: Vec<usize>,  // per slot
        keys: C::DeviceArray,   // [num_groups, slots.len(), head_dim]
        values: C::DeviceArray, // [num_groups, slots.len(), head_dim]
    },
}

impl<C: DeviceContext> Clone for KVSlice<C>
where
    C::DeviceArray: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Full {
                base_prefix_len,
                base_positions_len,
                positions,
            } => Self::Full {
                base_prefix_len: *base_prefix_len,
                base_positions_len: *base_positions_len,
                positions: positions.clone(),
            },
            Self::Window {
                window_length,
                base_ring_offset,
                base_ring_length,
                slots,
                positions,
                keys,
                values,
            } => Self::Window {
                window_length: *window_length,
                base_ring_offset: *base_ring_offset,
                base_ring_length: *base_ring_length,
                slots: slots.clone(),
                positions: positions.clone(),
                keys: keys.clone(),
                values: values.clone(),
            },
        }
    }
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
pub struct KVCacheLayer<C: DeviceContext> {
    pub state: KVCacheLayerState,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: ArrayCell<C>,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: ArrayCell<C>,

    pub prefix_token_positions: Vec<usize>,
    pub max_suffix_length: usize,
}

impl<C: DeviceContext> KVCacheLayer<C> {
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
        dst: &mut C::DeviceArray,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        context: &C,
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
                let w = *window_length;
                for &pos in token_positions {
                    let slot = (*ring_offset + *ring_length) % w;
                    self.prefix_token_positions[slot] = pos;
                    if *ring_length < w {
                        *ring_length += 1;
                    } else {
                        *ring_offset = (*ring_offset + 1) % w;
                    }
                }
            },
        }
    }

    pub fn attention_bias_update_after_acceptance(
        &self,
        _accepted_len: usize,
    ) -> Option<AttentionBiasUpdate> {
        match &self.state {
            KVCacheLayerState::Full {
                ..
            } => None, // Full attention doesn't need rolling updates usually? Or handled differently.
            KVCacheLayerState::Windowed {
                ring_offset: _ring_offset,
                ring_length: _ring_length,
                window_length,
            } => {
                let _w = *window_length;
                // Calculate which column to unmask and which to mask
                // This logic depends on the specific ring buffer state transition
                // Simplified for now based on what likely happens:
                // We accepted `accepted_len` tokens.
                // The ring buffer rotated.
                // We need to update the mask for specific columns.
                // Returning None as placeholder if logic is complex/backend specific
                // but keeping signature for compatibility.
                None
            },
        }
    }

    pub fn slice(
        &self,
        context: &C,
        range: std::ops::Range<usize>,
    ) -> Option<KVSlice<C>> {
        match &self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => Some(KVSlice::Full {
                base_prefix_len: *prefix_len,
                base_positions_len: self.prefix_token_positions.len(),
                positions: self.prefix_token_positions.clone(),
            }),
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let w = *window_length;
                let rl = *ring_length;
                let ro = *ring_offset;

                let slots: Vec<usize> = (range.start..range.end)
                    .enumerate()
                    .map(|(offset, _)| {
                        let x = rl + offset;
                        if x < w {
                            (ro + x) % w
                        } else {
                            (ro + (x - w)) % w
                        }
                    })
                    .collect();

                let positions: Vec<usize> = slots
                    .iter()
                    .map(|&s| self.prefix_token_positions[s])
                    .collect();

                let keys_ref = self.keys.borrow();
                let values_ref = self.values.borrow();

                let shape = keys_ref.shape(); // [num_groups, total_len, head_dim]
                let num_groups = shape[0];
                let head_dim = shape[2];

                let slice_shape = [num_groups, slots.len(), head_dim];
                let mut slice_keys = unsafe {
                    context
                        .array_uninitialized(&slice_shape, keys_ref.data_type())
                };
                let mut slice_values = unsafe {
                    context.array_uninitialized(
                        &slice_shape,
                        values_ref.data_type(),
                    )
                };

                for (i, &slot) in slots.iter().enumerate() {
                    slice_keys.copy_slice(&keys_ref, 1, slot..slot + 1, i);
                    slice_values.copy_slice(&values_ref, 1, slot..slot + 1, i);
                }

                Some(KVSlice::Window {
                    window_length: w,
                    base_ring_offset: *ring_offset,
                    base_ring_length: *ring_length,
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
        slice: &KVSlice<C>,
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
                    window_length: _,
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
                    window_length,
                },
            ) => {
                let w = *window_length;

                let mut dst_keys = self.keys.borrow_mut();
                let mut dst_values = self.values.borrow_mut();

                match range {
                    None => {
                        *ring_offset = *base_ring_offset;
                        *ring_length = *base_ring_length;

                        // Restore all
                        for (i, &slot) in slots.iter().enumerate() {
                            self.prefix_token_positions[slot] = positions[i];
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

                        // Restore sub-range
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
