use std::{any::Any, ops::Range};

use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    backends::common::{
        AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, Context, Encoder, SparseBuffer,
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

pub trait KVCacheLayerTrait<B: Backend> {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn shape(&self) -> [usize; 3];

    fn data_type(&self) -> DataType;

    fn encode_zero(
        &mut self,
        encoder: &mut Encoder<'_, B>,
    );

    fn state(&self) -> KVCacheLayerState;

    fn set_state(
        &mut self,
        state: &KVCacheLayerState,
    );

    fn clear_state(&mut self);

    fn encode_copy_prefix_rows_to(
        &self,
        destination: &mut dyn KVCacheLayerTrait<B>,
        row_count: usize,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    );

    fn prefix_segment_length(&self) -> usize;

    fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    );

    fn register_accepted_tokens(
        &mut self,
        number_of_accepted_tokens: usize,
    );

    fn slice(
        &self,
        context: &B::Context,
        encoder: &mut Encoder<B>,
        range: Range<usize>,
    ) -> Option<KVSlice<B>>;

    fn apply_slice(
        &mut self,
        context: &B::Context,
        encoder: &mut Encoder<B>,
        slice: &KVSlice<B>,
        range: Option<Range<usize>>,
    );

    fn map_row_range(
        &mut self,
        context: &B::Context,
        range: Range<usize>,
    ) -> Result<(), B::Error>;
}

impl<B: Backend> dyn KVCacheLayerTrait<B> {
    pub fn new(
        context: &B::Context,
        state: &KVCacheLayerState,
        shape: [usize; 3],
        data_type: DataType,
    ) -> Result<Box<dyn KVCacheLayerTrait<B>>, B::Error> {
        let buffer_size = size_for_shape(&shape, data_type);
        if context.sparse_buffers_supported() {
            let layer = KVCacheLayer {
                state: state.clone(),
                keys: context.create_sparse_buffer(buffer_size)?,
                values: context.create_sparse_buffer(buffer_size)?,
                shape,
                data_type,
            };
            Ok(Box::new(layer))
        } else {
            let layer = KVCacheLayer {
                state: state.clone(),
                keys: context.create_buffer(buffer_size)?,
                values: context.create_buffer(buffer_size)?,
                shape,
                data_type,
            };
            Ok(Box::new(layer))
        }
    }
}

pub struct KVCacheLayer<B: Backend, Buf: Buffer<Backend = B>> {
    pub state: KVCacheLayerState,
    /// [max_prefix_length + max_suffix_length, num_groups, head_dim]
    pub keys: Buf,
    /// [max_prefix_length + max_suffix_length, num_groups, head_dim]
    pub values: Buf,
    pub shape: [usize; 3],
    pub data_type: DataType,
}

impl<B: Backend, Buf: Buffer<Backend = B>> KVCacheLayer<B, Buf> {
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
            key_buffer: &mut self.keys,
            key_shape: self.shape,
            value_buffer: &mut self.values,
            value_shape: self.shape,
        };

        let data = std::slice::from_mut(&mut layer_data);
        let _ = kv_cache_update.encode(data, source_indices, destination_indices, encoder);
    }

    fn copy_rows<
        SrcKeys: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
        DstKeys: AsBufferRangeMut<Buffer: Buffer<Backend = B>>,
        SrcValues: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
        DstValues: AsBufferRangeMut<Buffer: Buffer<Backend = B>>,
    >(
        encoder: &mut Encoder<B>,
        src_keys: &SrcKeys,
        dst_keys: &mut DstKeys,
        src_values: &SrcValues,
        dst_values: &mut DstValues,
        row_size: usize,
        row_pairs: impl IntoIterator<Item = (usize, usize)>,
    ) {
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

    fn page_range_for_row_range(
        shape: [usize; 3],
        data_type: DataType,
        page_size_bytes: usize,
        row_range: Range<usize>,
    ) -> Range<usize> {
        if row_range.is_empty() {
            return 0..0;
        }

        let [_, num_groups, head_dim] = shape;
        let row_size = size_for_shape(&[1, num_groups, head_dim], data_type);
        let byte_start = row_range.start * row_size;
        let byte_end = row_range.end * row_size;
        byte_start / page_size_bytes..byte_end.div_ceil(page_size_bytes)
    }

    fn map_sparse_row_range(
        &mut self,
        context: &B::Context,
        row_range: Range<usize>,
    ) -> Result<(), B::Error> {
        if row_range.is_empty() {
            return Ok(());
        }

        if let Some(layer) = self.as_any_mut().downcast_mut::<KVCacheLayer<B, B::SparseBuffer>>() {
            let key_pages = Self::page_range_for_row_range(
                layer.shape,
                layer.data_type,
                layer.keys.page_size_bytes(),
                row_range.clone(),
            );
            layer.keys.map(context, &key_pages)?;

            let value_pages =
                Self::page_range_for_row_range(layer.shape, layer.data_type, layer.values.page_size_bytes(), row_range);
            layer.values.map(context, &value_pages)?;
        }
        Ok(())
    }
}

impl<B: Backend, Buf: Buffer<Backend = B>> KVCacheLayerTrait<B> for KVCacheLayer<B, Buf> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn shape(&self) -> [usize; 3] {
        self.shape
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn encode_zero(
        &mut self,
        encoder: &mut Encoder<'_, B>,
    ) {
        if self.as_any().is::<KVCacheLayer<B, B::SparseBuffer>>() {
            return;
        }
        encoder.encode_fill(&mut self.keys, 0);
        encoder.encode_fill(&mut self.values, 0);
    }

    fn state(&self) -> KVCacheLayerState {
        self.state.clone()
    }

    fn set_state(
        &mut self,
        state: &KVCacheLayerState,
    ) {
        self.state = state.clone();
    }

    fn clear_state(&mut self) {
        match &mut self.state {
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
    }

    fn encode_copy_prefix_rows_to(
        &self,
        destination: &mut dyn KVCacheLayerTrait<B>,
        row_count: usize,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) {
        if row_count == 0 {
            return;
        }

        let [source_seq, num_groups, head_dim] = self.shape;
        let [destination_seq, destination_groups, destination_head_dim] = destination.shape();
        assert_eq!(num_groups, destination_groups, "KV cache group count mismatch");
        assert_eq!(head_dim, destination_head_dim, "KV cache head dim mismatch");
        assert_eq!(self.data_type, destination.data_type(), "KV cache dtype mismatch");
        assert!(row_count <= source_seq, "source KV cache copy exceeds source sequence");
        assert!(row_count <= destination_seq, "source KV cache copy exceeds destination sequence");
        destination.map_row_range(context, 0..row_count).expect("Failed to map destination KV cache rows");

        let row_size = size_for_shape(&[1, num_groups, head_dim], self.data_type);
        let copy_size = row_count * row_size;
        if let Some(dest) = destination.as_any_mut().downcast_mut::<KVCacheLayer<B, B::SparseBuffer>>() {
            encoder.encode_copy(&self.keys, 0..copy_size, &mut dest.keys, 0..copy_size);
            encoder.encode_copy(&self.values, 0..copy_size, &mut dest.values, 0..copy_size);
        } else if let Some(dest) = destination.as_any_mut().downcast_mut::<KVCacheLayer<B, B::DenseBuffer>>() {
            encoder.encode_copy(&self.keys, 0..copy_size, &mut dest.keys, 0..copy_size);
            encoder.encode_copy(&self.values, 0..copy_size, &mut dest.values, 0..copy_size);
        } else {
            panic!("Wrong destination type!")
        }
    }

    fn prefix_segment_length(&self) -> usize {
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

    fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        context: &B::Context,
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

                let prefix_len_value = *prefix_len;

                // Absolute positions of the *source* rows.
                let source_indices: Vec<usize> =
                    accepted_suffix_indices.iter().map(|i| i + suffix_start.unwrap_or(prefix_len_value)).collect();

                // Absolute positions of the *destination* rows.
                let destination_indices: Vec<usize> =
                    (prefix_len_value..prefix_len_value + accepted_suffix_indices.len()).collect();

                self.map_sparse_row_range(context, prefix_len_value..prefix_len_value + accepted_suffix_indices.len())
                    .expect("Failed to map accepted KV cache rows");
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

                for destination_index in destination_indices.iter().copied() {
                    self.map_sparse_row_range(context, destination_index..destination_index + 1)
                        .expect("Failed to map accepted KV cache row");
                }
                self.scatter_if_required(&source_indices, &destination_indices, encoder, kv_cache_update);
            },
        }
    }

    fn register_accepted_tokens(
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

    fn slice(
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

    fn apply_slice(
        &mut self,
        context: &B::Context,
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

                let row_pairs: Option<Vec<(usize, usize)>> = match &range {
                    None => {
                        *ring_offset = *base_ring_offset;
                        *ring_length = *base_ring_length;
                        let pairs = slots.iter().enumerate().map(|(src_row, &dst_row)| (src_row, dst_row)).collect();
                        Some(pairs)
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
                            Some(pairs.collect())
                        }
                    },
                };
                let Some(row_pairs) = row_pairs else {
                    return;
                };
                for (_, destination_row) in row_pairs.iter().copied() {
                    self.map_sparse_row_range(context, destination_row..destination_row + 1)
                        .expect("Failed to map sliced KV cache row");
                }

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

    fn map_row_range(
        &mut self,
        context: &B::Context,
        range: Range<usize>,
    ) -> Result<(), B::Error> {
        self.map_sparse_row_range(context, range)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/forward_pass/kv_cache_state_test.rs"]
mod tests;
