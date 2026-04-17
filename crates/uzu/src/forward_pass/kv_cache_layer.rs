use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    backends::common::{
        Allocation, Backend, Buffer, Encoder,
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

pub struct KVCacheLayer<B: Backend> {
    pub state: KVCacheLayerState,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: Allocation<B>,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: Allocation<B>,
    pub shape: [usize; 3],
    pub data_type: DataType,
}

impl<B: Backend> KVCacheLayer<B> {
    pub fn encode_copy_prefix_rows_to(
        &self,
        destination: &KVCacheLayer<B>,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) {
        if row_count == 0 {
            return;
        }

        let [num_groups, source_seq, head_dim] = self.shape;
        let [destination_groups, destination_seq, destination_head_dim] = destination.shape;
        assert_eq!(num_groups, destination_groups, "KV cache group count mismatch");
        assert_eq!(head_dim, destination_head_dim, "KV cache head dim mismatch");
        assert_eq!(self.data_type, destination.data_type, "KV cache dtype mismatch");
        assert!(row_count <= source_seq, "source KV cache copy exceeds source sequence");
        assert!(row_count <= destination_seq, "source KV cache copy exceeds destination sequence");

        let row_size = size_for_shape(&[1, 1, head_dim], self.data_type);
        let source_block_size = source_seq * row_size;
        let destination_block_size = destination_seq * row_size;
        let copy_size = row_count * row_size;

        let (source_keys_buffer, source_keys_range) = self.keys.as_buffer_range();
        let (source_values_buffer, source_values_range) = self.values.as_buffer_range();
        let (destination_keys_buffer, destination_keys_range) = destination.keys.as_buffer_range();
        let (destination_values_buffer, destination_values_range) = destination.values.as_buffer_range();

        for group in 0..num_groups {
            let source_key_offset = source_keys_range.start + group * source_block_size;
            let source_value_offset = source_values_range.start + group * source_block_size;
            let destination_key_offset = destination_keys_range.start + group * destination_block_size;
            let destination_value_offset = destination_values_range.start + group * destination_block_size;

            encoder.encode_copy(
                source_keys_buffer,
                source_key_offset..source_key_offset + copy_size,
                destination_keys_buffer,
                destination_key_offset..destination_key_offset + copy_size,
            );
            encoder.encode_copy(
                source_values_buffer,
                source_value_offset..source_value_offset + copy_size,
                destination_values_buffer,
                destination_value_offset..destination_value_offset + copy_size,
            );
        }
    }

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
        &mut self,
        source_indices: &[usize],
        destination_indices: &[usize],
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        if source_indices == destination_indices {
            return;
        }

        let mut layer_data = KVLayerData {
            key_allocation: &mut self.keys,
            key_shape: self.shape,
            value_allocation: &mut self.values,
            value_shape: self.shape,
        };

        let _ =
            kv_cache_update.encode(std::slice::from_mut(&mut layer_data), source_indices, destination_indices, encoder);
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
                let [num_groups, _, head_dim] = self.shape;
                let dtype = self.data_type;

                let slice_shape = [num_groups, len, head_dim];
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
                    copy_allocation_slot_to_array(&self.keys, self.shape, self.data_type, slot, &mut slice_keys, i);
                    copy_allocation_slot_to_array(&self.values, self.shape, self.data_type, slot, &mut slice_values, i);
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
                            copy_array_slot_to_allocation(keys, i, &mut self.keys, self.shape, self.data_type, slot);
                            copy_array_slot_to_allocation(
                                values,
                                i,
                                &mut self.values,
                                self.shape,
                                self.data_type,
                                slot,
                            );
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
                            copy_array_slot_to_allocation(
                                keys,
                                src_i,
                                &mut self.keys,
                                self.shape,
                                self.data_type,
                                slot,
                            );
                            copy_array_slot_to_allocation(
                                values,
                                src_i,
                                &mut self.values,
                                self.shape,
                                self.data_type,
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

fn copy_allocation_slot_to_array<B: Backend>(
    source: &Allocation<B>,
    source_shape: [usize; 3],
    data_type: DataType,
    source_slot: usize,
    destination: &mut Array<B>,
    destination_slot: usize,
) {
    let [num_groups, source_seq, head_dim] = source_shape;
    let row_size = size_for_shape(&[1, 1, head_dim], data_type);
    let block_size = source_seq * row_size;
    let destination_seq = destination.shape()[1];
    let destination_block_size = destination_seq * row_size;
    let (source_buffer, source_range) = source.as_buffer_range();
    let destination_bytes = destination.as_bytes_mut();

    for group in 0..num_groups {
        let source_offset = source_range.start + group * block_size + source_slot * row_size;
        let destination_offset = group * destination_block_size + destination_slot * row_size;
        unsafe {
            let source_bytes = std::slice::from_raw_parts(
                (source_buffer.cpu_ptr().as_ptr() as *const u8).add(source_offset),
                row_size,
            );
            destination_bytes[destination_offset..destination_offset + row_size].copy_from_slice(source_bytes);
        }
    }
}

fn copy_array_slot_to_allocation<B: Backend>(
    source: &Array<B>,
    source_slot: usize,
    destination: &mut Allocation<B>,
    destination_shape: [usize; 3],
    data_type: DataType,
    destination_slot: usize,
) {
    let [num_groups, destination_seq, head_dim] = destination_shape;
    let row_size = size_for_shape(&[1, 1, head_dim], data_type);
    let source_block_size = source.shape()[1] * row_size;
    let destination_block_size = destination_seq * row_size;
    let source_bytes = source.as_bytes();
    let (destination_buffer, destination_range) = destination.as_buffer_range();

    for group in 0..num_groups {
        let source_offset = group * source_block_size + source_slot * row_size;
        let destination_offset = destination_range.start + group * destination_block_size + destination_slot * row_size;
        unsafe {
            let destination_bytes = std::slice::from_raw_parts_mut(
                (destination_buffer.cpu_ptr().as_ptr() as *mut u8).add(destination_offset),
                row_size,
            );
            destination_bytes.copy_from_slice(&source_bytes[source_offset..source_offset + row_size]);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/forward_pass/kv_cache_state_test.rs"]
mod tests;
