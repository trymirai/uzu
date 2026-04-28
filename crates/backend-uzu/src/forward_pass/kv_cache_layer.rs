use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{
        Backend, Encoder,
        kernel::kv_cache_update::{KVCacheUpdate, KVLayerData},
    },
};

#[derive(Clone)]
pub enum KVSlice<B: Backend> {
    Full {
        base_prefix_len: usize,
    },
    Window {
        window_length: usize,
        base_ring_offset: usize,
        base_ring_length: usize,
        slots: Vec<usize>,
        keys: Array<B>,   // [num_groups, slots.len(), head_dim]
        values: Array<B>, // [num_groups, slots.len(), head_dim]
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
        /// Maximum length of the window
        window_length: usize,
    },
}

pub const INVALID_POSITION: usize = i32::MAX as usize;

#[derive(Debug)]
pub struct KVCacheLayer<B: Backend> {
    pub state: KVCacheLayerState,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: Array<B>,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: Array<B>,
}

impl<B: Backend> KVCacheLayer<B> {
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

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                if accepted_suffix_indices.is_empty() {
                    return;
                }

                // Absolute positions of the *source* rows.
                let source_indices: Vec<usize> =
                    accepted_suffix_indices.iter().map(|i| i + suffix_start.unwrap_or(*prefix_len)).collect();

                // Absolute positions of the *destination* rows.
                let destination_indices: Vec<usize> =
                    (*prefix_len..*prefix_len + accepted_suffix_indices.len()).collect();

                self.scatter_if_required(&source_indices, &destination_indices, encoder, kv_cache_update);
            },

            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let suffix_indices: Vec<usize> = if accepted_suffix_indices.is_empty() {
                    vec![0]
                } else {
                    accepted_suffix_indices.to_vec()
                };

                let source_indices: Vec<usize> = suffix_indices.iter().map(|i| i + *window_length).collect();

                let mut destination_indices = Vec::with_capacity(suffix_indices.len());

                for i in 0..suffix_indices.len() {
                    destination_indices.push((*ring_length + *ring_offset + i) % *window_length);
                }

                self.scatter_if_required(&source_indices, &destination_indices, encoder, kv_cache_update);
            },
        }
    }

    fn scatter_if_required(
        &self,
        source_indices: &[usize],
        destination_indices: &[usize],
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        if source_indices == destination_indices {
            return;
        }

        let k_shape = self.keys.shape();
        let v_shape = self.values.shape();

        let layer_data = KVLayerData {
            key_buffer: self.keys.buffer(),
            key_shape: [k_shape[0], k_shape[1], k_shape[2]],
            value_buffer: self.values.buffer(),
            value_shape: [v_shape[0], v_shape[1], v_shape[2]],
        };

        let _ = kv_cache_update.encode(&[layer_data], source_indices, destination_indices, encoder);
    }

    pub fn register_accepted_tokens(
        &mut self,
        number_of_accepted_tokens: usize,
    ) {
        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                *prefix_len += number_of_accepted_tokens;
            },
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let ring_advance = number_of_accepted_tokens.saturating_sub(*window_length - *ring_length);
                *ring_offset = (*ring_offset + ring_advance) % *window_length;
                *ring_length = (*ring_length + number_of_accepted_tokens).min(*window_length);
            },
        }
    }

    pub fn slice(
        &self,
        context: &B::Context,
        range: std::ops::Range<usize>,
    ) -> Option<KVSlice<B>> {
        match self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => Some(KVSlice::Full {
                base_prefix_len: prefix_len,
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
                let shape = self.keys.shape();
                let num_groups = shape[1];
                let head_dim = shape[2];
                let dtype = self.keys.data_type();

                let slice_shape = [len, num_groups, head_dim];
                let mut slice_keys =
                    context.create_array_uninitialized(&slice_shape, dtype, "kv_cache_layer_slice_keys");
                let mut slice_values =
                    context.create_array_uninitialized(&slice_shape, dtype, "kv_cache_layer_slice_values");

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

                for (i, &slot) in slots.iter().enumerate() {
                    slice_keys.copy_slice(&self.keys, 0, slot..slot + 1, i);
                    slice_values.copy_slice(&self.values, 0, slot..slot + 1, i);
                }

                Some(KVSlice::Window {
                    window_length,
                    base_ring_offset: ring_offset,
                    base_ring_length: ring_length,
                    slots,
                    keys: slice_keys,
                    values: slice_values,
                })
            },
        }
    }

    pub fn apply_slice(
        &mut self,
        slice: &KVSlice<B>,
        range: Option<std::ops::Range<usize>>,
    ) {
        match (slice, &mut self.state) {
            (
                KVSlice::Full {
                    base_prefix_len,
                },
                KVCacheLayerState::Full {
                    prefix_len,
                },
            ) => match range {
                None => {
                    *prefix_len = *base_prefix_len;
                },
                Some(r) => {
                    *prefix_len = base_prefix_len.saturating_add(r.start);
                },
            },
            (
                KVSlice::Window {
                    window_length,
                    base_ring_offset,
                    base_ring_length,
                    slots,
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
                            self.keys.copy_slice(keys, 0, i..i + 1, slot);
                            self.values.copy_slice(values, 0, i..i + 1, slot);
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

                        for (i, &slot) in slots[r.clone()].iter().enumerate() {
                            let src_i = r.start + i;
                            self.keys.copy_slice(keys, 0, src_i..src_i + 1, slot);
                            self.values.copy_slice(values, 0, src_i..src_i + 1, slot);
                        }
                    },
                }
            },
            _ => {},
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/forward_pass/kv_cache_state_test.rs"]
mod tests;
