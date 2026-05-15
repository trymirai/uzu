use std::ops::Range;

use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    backends::common::{
        Allocation, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, Encoder,
        kernel::kv_cache_update::{KVCacheUpdate, KVLayerData},
    },
};

pub enum KVSlice<B: Backend> {
    Full {
        base_prefix_len: usize,
    },
    Window {
        window_length: usize,
        base_ring_offset: usize,
        base_ring_length: usize,
        slots: Vec<usize>,
        keys: Array<B>,   // [slots.len(), num_groups, head_dim]
        values: Array<B>, // [slots.len(), num_groups, head_dim]
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
    /// [max_prefix_length + max_suffix_length, num_groups, head_dim]
    pub keys: Allocation<B>,
    /// [max_prefix_length + max_suffix_length, num_groups, head_dim]
    pub values: Allocation<B>,
    pub shape: [usize; 3],
    pub data_type: DataType,
}

impl<B: Backend> KVCacheLayer<B> {
    pub fn encode_copy_prefix_rows_to(
        &self,
        destination: &mut KVCacheLayer<B>,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) {
        if row_count == 0 {
            return;
        }

        let [source_seq, num_groups, head_dim] = self.shape;
        let [destination_seq, destination_groups, destination_head_dim] = destination.shape;
        assert_eq!(num_groups, destination_groups, "KV cache group count mismatch");
        assert_eq!(head_dim, destination_head_dim, "KV cache head dim mismatch");
        assert_eq!(self.data_type, destination.data_type, "KV cache dtype mismatch");
        assert!(row_count <= source_seq, "source KV cache copy exceeds source sequence");
        assert!(row_count <= destination_seq, "source KV cache copy exceeds destination sequence");

        let row_size = size_for_shape(&[1, num_groups, head_dim], self.data_type);
        let copy_size = row_count * row_size;

        encoder.encode_copy(&self.keys, 0..copy_size, &mut destination.keys, 0..copy_size);
        encoder.encode_copy(&self.values, 0..copy_size, &mut destination.values, 0..copy_size);
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
        encoder: &mut Encoder<B>,
        range: Range<usize>,
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
                let [_, num_groups, head_dim] = self.shape;
                let dtype = self.data_type;
                let row_size = size_for_shape(&[1, num_groups, head_dim], dtype);

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
                let row_pairs = slots.iter().enumerate().map(|(dst_row, &src_row)| (src_row, dst_row));

                let slice_shape = [len, num_groups, head_dim];
                let mut dst_keys = context.create_array_uninitialized(&slice_shape, dtype);
                let mut dst_values = context.create_array_uninitialized(&slice_shape, dtype);
                Self::copy_rows(
                    encoder,
                    &self.keys,
                    dst_keys.allocation_mut(),
                    &self.values,
                    dst_values.allocation_mut(),
                    row_size,
                    row_pairs,
                );

                Some(KVSlice::Window {
                    window_length,
                    base_ring_offset: ring_offset,
                    base_ring_length: ring_length,
                    slots,
                    keys: dst_keys,
                    values: dst_values,
                })
            },
        }
    }

    pub fn apply_slice(
        &mut self,
        encoder: &mut Encoder<B>,
        slice: &KVSlice<B>,
        range: Option<Range<usize>>,
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
                let [_, num_groups, head_dim] = self.shape;
                let row_size = size_for_shape(&[1, num_groups, head_dim], self.data_type);

                let row_pairs: Option<Box<dyn Iterator<Item = (usize, usize)>>> = match range {
                    None => {
                        *ring_offset = *base_ring_offset;
                        *ring_length = *base_ring_length;
                        let pairs = slots.iter().enumerate().map(|(src_row, &dst_row)| (src_row, dst_row));
                        Some(Box::new(pairs))
                    },
                    Some(r) => {
                        let len = r.end.saturating_sub(r.start);
                        if len == 0 {
                            None
                        } else {
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

                            let pairs =
                                slots[r.clone()].iter().enumerate().map(move |(i, &dst_row)| (r.start + i, dst_row));
                            Some(Box::new(pairs))
                        }
                    },
                };
                let Some(row_pairs) = row_pairs else {
                    return;
                };

                Self::copy_rows(
                    encoder,
                    keys.allocation(),
                    &mut self.keys,
                    values.allocation(),
                    &mut self.values,
                    row_size,
                    row_pairs,
                );
            },
            _ => {},
        }
    }

    fn copy_rows<SrcKeys, DstKeys, SrcValues, DstValues>(
        encoder: &mut Encoder<B>,
        src_keys: &SrcKeys,
        dst_keys: &mut DstKeys,
        src_values: &SrcValues,
        dst_values: &mut DstValues,
        row_size: usize,
        row_pairs: impl IntoIterator<Item = (usize, usize)>,
    ) where
        SrcKeys: AsBufferRangeRef,
        SrcKeys::Buffer: Buffer<Backend = B>,
        DstKeys: AsBufferRangeMut,
        DstKeys::Buffer: Buffer<Backend = B>,
        SrcValues: AsBufferRangeRef,
        SrcValues::Buffer: Buffer<Backend = B>,
        DstValues: AsBufferRangeMut,
        DstValues::Buffer: Buffer<Backend = B>,
    {
        for (src_row, dst_row) in row_pairs {
            let src_offset = src_row * row_size;
            let dst_offset = dst_row * row_size;
            encoder.encode_copy(
                src_keys,
                src_offset..src_offset + row_size,
                dst_keys,
                dst_offset..dst_offset + row_size,
            );
            encoder.encode_copy(
                src_values,
                src_offset..src_offset + row_size,
                dst_values,
                dst_offset..dst_offset + row_size,
            );
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/forward_pass/kv_cache_state_test.rs"]
mod tests;
