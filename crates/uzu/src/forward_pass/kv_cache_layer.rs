use crate::{
    DataType,
    array::{Array, ArrayCell, ArrayContextExt},
    backends::common::{
        Backend, Encoder,
        kernel::kv_cache_update::{KVCacheUpdate, KVLayerData},
    },
    utils::attention::fill_attention_bias,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCompressionMode {
    None,
    SparseValue,
    ShearKv,
    TurboQuant,
    ShapedCache,
    SpectralQuant,
    TriAttention,
}

pub struct SingleDecodeValueKernelBuffers<'a, B: Backend> {
    pub codes: &'a B::Buffer,
    pub scales: &'a B::Buffer,
    pub biases: &'a B::Buffer,
    pub bits: usize,
    pub row_bytes: usize,
}

pub trait KvCompressor<B: Backend> {
    fn compress(
        &mut self,
        keys: &Array<B>,
        values: &Array<B>,
    );
    fn decompress(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    );
    fn decompress_keys(
        &self,
        keys: &mut Array<B>,
    ) {
        let _ = keys;
        panic!("compressed key materialization is not implemented for this compressor");
    }
    fn decompress_values(
        &self,
        values: &mut Array<B>,
    ) {
        let _ = values;
        panic!("compressed value materialization is not implemented for this compressor");
    }
    fn update_rows_from_dense(
        &mut self,
        context: &B::Context,
        keys: &Array<B>,
        values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    );
    fn supports_prefix_attention_scores_for_single_decode(&self) -> bool {
        false
    }
    fn supports_value_row_decoding_for_single_decode(&self) -> bool {
        false
    }
    fn value_kernel_buffers_for_single_decode(&self) -> Option<SingleDecodeValueKernelBuffers<'_, B>> {
        None
    }
    fn fill_prefix_attention_scores_for_single_decode(
        &self,
        queries: &[f32],
        num_heads: usize,
        prefix_length: usize,
        scores_out: &mut [f32],
    ) -> bool {
        let _ = (queries, num_heads, prefix_length, scores_out);
        false
    }
    fn decode_value_row_for_single_decode(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        let _ = (row_index, scratch, output);
        panic!("compressed value row decoding is not implemented for this compressor");
    }
    fn decode_value_rows_for_single_decode(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        if row_count == 0 {
            assert!(output.is_empty(), "compressed value row decode output must be empty when row_count is zero");
            return;
        }
        let head_dim = output.len() / row_count;
        assert_eq!(
            output.len(),
            row_count * head_dim,
            "compressed value row decode output must contain a whole number of rows"
        );
        let mut scratch = vec![0.0; head_dim];
        for (offset, row) in output.chunks_exact_mut(head_dim).enumerate() {
            self.decode_value_row_for_single_decode(row_index + offset, &mut scratch, row);
        }
    }
    fn memory_usage_bytes(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriAttentionConfig {
    pub budget: usize,
    pub divide_length: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseValueConfig {
    pub recent_window: usize,
    pub keep_mass: f32,
    pub page_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseValueState {
    pub config: SparseValueConfig,
    pub shadow_keys: Box<[f32]>,
    pub hot_value_capacity: usize,
    pub pending_suffix_len: usize,
    pub pending_values_ready_on_cpu: bool,
    pub pending_keys: Box<[f32]>,
}

impl SparseValueState {
    pub fn new(
        config: SparseValueConfig,
        num_groups: usize,
        sequence_length: usize,
        head_dim: usize,
        max_suffix_length: usize,
    ) -> Self {
        let shadow_len = num_groups * sequence_length * head_dim;
        let pending_len = num_groups * max_suffix_length * head_dim;
        let hot_value_capacity = (config.recent_window + config.page_size).min(sequence_length);
        Self {
            config,
            shadow_keys: vec![0.0; shadow_len].into_boxed_slice(),
            hot_value_capacity,
            pending_suffix_len: 0,
            pending_values_ready_on_cpu: false,
            pending_keys: vec![0.0; pending_len].into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriAttentionCalibration {
    pub num_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub query_token_count: usize,
    pub query_sum: Box<[f32]>,
    pub query_norm_sum: Box<[f32]>,
}

impl TriAttentionCalibration {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        rope_dim: usize,
    ) -> Self {
        assert!(head_dim % 2 == 0, "TriAttention requires an even head dimension");
        assert!(rope_dim % 2 == 0, "TriAttention requires an even rope dimension");
        assert!(rope_dim <= head_dim, "TriAttention rope dimension must not exceed head dimension");
        let rope_pair_count = rope_dim / 2;
        Self {
            num_heads,
            head_dim,
            rope_dim,
            query_token_count: 0,
            query_sum: vec![0.0; num_heads * head_dim].into_boxed_slice(),
            query_norm_sum: vec![0.0; num_heads * rope_pair_count].into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriAttentionState {
    pub config: TriAttentionConfig,
    pub tokens_since_last_prune: usize,
    pub calibration: TriAttentionCalibration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBiasUpdate {
    pub key: Option<usize>,
    pub unmask_col: i32,
    pub mask_col: i32,
}

#[derive(Clone)]
pub enum KVSlice<B: Backend> {
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
        keys: Array<B>,        // [num_groups, slots.len(), head_dim]
        values: Array<B>,      // [num_groups, slots.len(), head_dim]
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

pub struct KVCacheLayer<B: Backend> {
    pub state: KVCacheLayerState,
    pub shape: [usize; 3],
    pub data_type: DataType,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub keys: Option<ArrayCell<B>>,
    /// [num_groups, max_prefix_length + max_suffix_length, head_dim]
    pub values: Option<ArrayCell<B>>,

    pub prefix_token_positions: Vec<usize>,
    pub next_token_position: usize,
    pub max_suffix_length: usize,
    pub compression_mode: KvCompressionMode,
    pub compressor: Option<Box<dyn KvCompressor<B>>>,
    pub sparse_value: Option<SparseValueState>,
    pub sparse_value_pending_values: Option<ArrayCell<B>>,
    pub sparse_value_recent_values: Option<ArrayCell<B>>,
    pub triattention: Option<TriAttentionState>,
}

impl<B: Backend> std::fmt::Debug for KVCacheLayer<B> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("KVCacheLayer")
            .field("state", &self.state)
            .field("shape", &self.shape)
            .field("data_type", &self.data_type)
            .field("compression_mode", &self.compression_mode)
            .field("sparse_value", &self.sparse_value.as_ref().map(|state| &state.config))
            .field(
                "sparse_value_pending_values",
                &self.sparse_value_pending_values.as_ref().map(|array| array.borrow().shape().to_vec()),
            )
            .field(
                "sparse_value_recent_values",
                &self.sparse_value_recent_values.as_ref().map(|array| array.borrow().shape().to_vec()),
            )
            .field("prefix_token_positions", &self.prefix_token_positions)
            .field("max_suffix_length", &self.max_suffix_length)
            .finish()
    }
}

impl<B: Backend> KVCacheLayer<B> {
    pub fn uses_compressed_storage(&self) -> bool {
        self.keys.is_none() || self.values.is_none()
    }

    pub fn dense_keys(&self) -> &ArrayCell<B> {
        self.keys.as_ref().expect("dense KV keys are required")
    }

    pub fn dense_values(&self) -> &ArrayCell<B> {
        self.values.as_ref().expect("dense KV values are required")
    }

    pub fn materialize_into(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    ) {
        self.materialize_keys_into(keys);
        self.materialize_values_into(values);
    }

    pub fn materialize_keys_into(
        &self,
        keys: &mut Array<B>,
    ) {
        if let Some(dense_keys) = &self.keys {
            keys.copy_from_array(&dense_keys.borrow());
            return;
        }
        self.compressor.as_ref().expect("compressed KV storage is required").decompress_keys(keys);
    }

    pub fn materialize_values_into(
        &self,
        values: &mut Array<B>,
    ) {
        if let Some(dense_values) = &self.values {
            values.copy_from_array(&dense_values.borrow());
            return;
        }
        self.compressor.as_ref().expect("compressed KV storage is required").decompress_values(values);
    }

    pub fn compress_from(
        &mut self,
        keys: &Array<B>,
        values: &Array<B>,
    ) {
        if let Some(dense_keys) = &self.keys {
            dense_keys.borrow_mut().copy_from_array(keys);
        }
        if let Some(dense_values) = &self.values {
            dense_values.borrow_mut().copy_from_array(values);
        }
        if self.uses_compressed_storage() {
            self.compressor.as_mut().expect("compressed KV storage is required").compress(keys, values);
        }
    }

    pub fn storage_bytes(&self) -> usize {
        let dense_keys = self.keys.as_ref().map(|keys| keys.borrow().size()).unwrap_or(0);
        let dense_values = self.values.as_ref().map(|values| values.borrow().size()).unwrap_or(0);
        let compressed = self
            .uses_compressed_storage()
            .then(|| self.compressor.as_ref().expect("compressed KV storage is required").memory_usage_bytes())
            .unwrap_or(0);
        dense_keys + dense_values + compressed
    }

    pub fn fill_compressed_prefix_attention_scores_for_single_decode(
        &self,
        queries: &[f32],
        num_heads: usize,
        prefix_length: usize,
        scores_out: &mut [f32],
    ) -> bool {
        let Some(compressor) = &self.compressor else {
            return false;
        };
        compressor.fill_prefix_attention_scores_for_single_decode(queries, num_heads, prefix_length, scores_out)
    }

    pub fn supports_compressed_prefix_attention_scores_for_single_decode(&self) -> bool {
        self.compressor
            .as_ref()
            .map(|compressor| compressor.supports_prefix_attention_scores_for_single_decode())
            .unwrap_or(false)
    }

    pub fn supports_value_row_decoding_for_single_decode(&self) -> bool {
        self.compressor
            .as_ref()
            .map(|compressor| compressor.supports_value_row_decoding_for_single_decode())
            .unwrap_or(false)
    }

    pub fn value_kernel_buffers_for_single_decode(&self) -> Option<SingleDecodeValueKernelBuffers<'_, B>> {
        self.compressor.as_ref().and_then(|compressor| compressor.value_kernel_buffers_for_single_decode())
    }

    pub fn decode_value_row_for_single_decode(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        self.compressor
            .as_ref()
            .expect("compressed KV storage is required")
            .decode_value_row_for_single_decode(row_index, scratch, output);
    }

    pub fn decode_value_rows_for_single_decode(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        self.compressor
            .as_ref()
            .expect("compressed KV storage is required")
            .decode_value_rows_for_single_decode(row_index, row_count, output);
    }

    pub fn uses_materialized_transformer_state(&self) -> bool {
        self.uses_compressed_storage() || self.sparse_value.is_some()
    }

    pub fn requires_synchronous_acceptance_update(&self) -> bool {
        self.sparse_value.as_ref().is_some_and(|state| !state.pending_values_ready_on_cpu)
    }

    pub fn allows_pre_encode_for_single_decode(&self) -> bool {
        if self.sparse_value.is_some() {
            return false;
        }
        if !self.uses_compressed_storage() {
            return true;
        }
        matches!(self.state, KVCacheLayerState::Full { .. })
            && self.supports_compressed_prefix_attention_scores_for_single_decode()
            && (self.values.is_some() || self.supports_value_row_decoding_for_single_decode())
    }

    fn create_materialized_storage(
        &self,
        context: &B::Context,
    ) -> (Array<B>, Array<B>) {
        let mut keys = context.create_array(&self.shape, self.data_type, "kv_cache_layer_materialized_keys");
        let mut values = context.create_array(&self.shape, self.data_type, "kv_cache_layer_materialized_values");
        self.materialize_into(&mut keys, &mut values);
        (keys, values)
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

    pub fn fill_attention_bias(
        &self,
        dst: &mut Array<B>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        let prefix_segment_length = self.prefix_segment_length();
        fill_attention_bias(dst, suffix_length, prefix_segment_length, |row_index, column_index| {
            if let Some(bias_fn) = external_bias_fn {
                bias_fn(row_index, column_index)
            } else {
                self.bias_should_be_neg_inf(row_index, column_index, suffix_token_positions)
            }
        });
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
        context: &B::Context,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        materialized_keys: Option<&Array<B>>,
        materialized_values: Option<&Array<B>>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        let (source_base, source_indices, destination_indices) = match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                if accepted_suffix_indices.is_empty() {
                    return;
                }
                let source_base = suffix_start.unwrap_or(*prefix_len);
                (
                    source_base,
                    accepted_suffix_indices.iter().map(|i| i + source_base).collect::<Vec<_>>(),
                    (*prefix_len..*prefix_len + accepted_suffix_indices.len()).collect(),
                )
            },
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let suffix_indices = if accepted_suffix_indices.is_empty() {
                    vec![0]
                } else {
                    accepted_suffix_indices.to_vec()
                };
                let source_base = *window_length;
                let source_indices = suffix_indices.iter().map(|i| i + *window_length).collect::<Vec<_>>();
                let mut destination_indices = Vec::with_capacity(suffix_indices.len());
                for index in 0..suffix_indices.len() {
                    destination_indices.push((*ring_length + *ring_offset + index) % *window_length);
                }
                (source_base, source_indices, destination_indices)
            },
        };
        self.scatter_if_required(
            context,
            &source_indices,
            &destination_indices,
            materialized_keys,
            materialized_values,
            encoder,
            kv_cache_update,
        );
        if self.sparse_value_pending_values.is_some() {
            if self.sparse_value.as_ref().is_some_and(|state| state.pending_values_ready_on_cpu) {
                self.copy_sparse_value_recent_values_from_pending(source_base, &source_indices, &destination_indices);
            } else {
                self.encode_sparse_value_recent_values_from_pending(
                    source_base,
                    &source_indices,
                    &destination_indices,
                    encoder,
                );
            }
        } else {
            let dense_values = self.values.as_ref().map(|values| values.borrow());
            let source_values = materialized_values
                .or(dense_values.as_deref())
                .expect("SparseValue recent-value updates require value rows");
            if materialized_values.is_some() {
                self.encode_sparse_value_recent_values_from_source(
                    source_values,
                    &source_indices,
                    &destination_indices,
                    encoder,
                );
            } else {
                self.copy_sparse_value_recent_values_from_source(source_values, &source_indices, &destination_indices);
            }
        }
        self.update_sparse_value_from_pending(source_base, &source_indices, &destination_indices);
    }

    fn scatter_if_required(
        &mut self,
        context: &B::Context,
        source_indices: &[usize],
        destination_indices: &[usize],
        materialized_keys: Option<&Array<B>>,
        materialized_values: Option<&Array<B>>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        if !self.uses_compressed_storage() {
            if source_indices == destination_indices {
                return;
            }

            let keys = self.dense_keys();
            let values = self.dense_values();
            let k_shape = keys.borrow().shape().to_vec();
            let v_shape = values.borrow().shape().to_vec();
            let layer_data = KVLayerData {
                key_buffer: keys.borrow().buffer(),
                key_shape: [k_shape[0], k_shape[1], k_shape[2]],
                value_buffer: values.borrow().buffer(),
                value_shape: [v_shape[0], v_shape[1], v_shape[2]],
            };
            let _ = kv_cache_update.encode(&[layer_data], source_indices, destination_indices, encoder);
            return;
        }

        if source_indices != destination_indices {
            if let Some(keys) = &self.keys {
                copy_dense_rows(keys, source_indices, destination_indices);
            }
            if let Some(values) = &self.values {
                copy_dense_rows(values, source_indices, destination_indices);
            }
        }

        let dense_keys = self.keys.as_ref().map(|keys| keys.borrow());
        let dense_values = self.values.as_ref().map(|values| values.borrow());
        let keys = materialized_keys.or(dense_keys.as_deref()).expect("compressed KV key update requires rows");
        let values = materialized_values.or(dense_values.as_deref()).expect("compressed KV value update requires rows");
        self.compressor.as_mut().expect("compressed KV storage is required").update_rows_from_dense(
            context,
            keys,
            values,
            source_indices,
            destination_indices,
        );
    }

    fn update_sparse_value_from_pending(
        &mut self,
        source_base: usize,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        let Some(sparse_value) = &mut self.sparse_value else {
            return;
        };
        if source_indices.is_empty() {
            return;
        }
        assert_eq!(
            source_indices.len(),
            destination_indices.len(),
            "SparseValue acceptance source/destination lengths must match"
        );
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let pending_stride = self.max_suffix_length * head_dim;
        let shadow_stride = sequence_length * head_dim;

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            let pending_index = source_index
                .checked_sub(source_base)
                .expect("SparseValue acceptance source index must be inside the staged suffix");
            assert!(
                pending_index < sparse_value.pending_suffix_len,
                "SparseValue acceptance source index exceeds the staged suffix length"
            );
            for group_index in 0..self.shape[0] {
                let pending_base = group_index * pending_stride + pending_index * head_dim;
                let shadow_base = group_index * shadow_stride + destination_index * head_dim;
                sparse_value.shadow_keys[shadow_base..shadow_base + head_dim]
                    .copy_from_slice(&sparse_value.pending_keys[pending_base..pending_base + head_dim]);
            }
        }
    }

    fn copy_sparse_value_recent_values_from_source(
        &self,
        source_values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        let Some(recent_values) = &self.sparse_value_recent_values else {
            return;
        };
        assert_eq!(
            source_indices.len(),
            destination_indices.len(),
            "SparseValue recent-value source/destination lengths must match"
        );
        if source_indices.is_empty() {
            return;
        }
        let recent_values = &mut *recent_values.borrow_mut();
        assert_eq!(
            source_values.data_type(),
            recent_values.data_type(),
            "SparseValue source/recent value dtype mismatch"
        );
        let head_dim = self.shape[2];
        let source_stride = source_values.shape()[1] * head_dim;
        let recent_stride = recent_values.shape()[1] * head_dim;

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            let recent_index = destination_index % recent_values.shape()[1];
            for group_index in 0..self.shape[0] {
                let source_start = group_index * source_stride + source_index * head_dim;
                let destination_start = group_index * recent_stride + recent_index * head_dim;
                match recent_values.data_type() {
                    DataType::BF16 => recent_values.as_slice_mut::<half::bf16>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(
                            &source_values.as_slice::<half::bf16>()[source_start..source_start + head_dim],
                        ),
                    DataType::F16 => recent_values.as_slice_mut::<half::f16>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(&source_values.as_slice::<half::f16>()[source_start..source_start + head_dim]),
                    DataType::F32 => recent_values.as_slice_mut::<f32>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(&source_values.as_slice::<f32>()[source_start..source_start + head_dim]),
                    dtype => panic!("SparseValue does not support recent-value dtype {dtype:?}"),
                }
            }
        }
    }

    fn encode_sparse_value_recent_values_from_source(
        &self,
        source_values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) {
        let Some(recent_values) = &self.sparse_value_recent_values else {
            return;
        };
        assert_eq!(
            source_indices.len(),
            destination_indices.len(),
            "SparseValue recent-value source/destination lengths must match"
        );
        if source_indices.is_empty() {
            return;
        }
        let recent_values = recent_values.borrow();
        assert_eq!(
            source_values.data_type(),
            recent_values.data_type(),
            "SparseValue source/recent value dtype mismatch"
        );
        let head_dim = self.shape[2];
        let row_bytes = head_dim * source_values.data_type().size_in_bytes();
        let source_group_stride = source_values.shape()[1] * row_bytes;
        let recent_group_stride = recent_values.shape()[1] * row_bytes;
        let recent_capacity = recent_values.shape()[1];
        let source_buffer = source_values.buffer();
        let source_buffer = source_buffer.borrow();
        let recent_buffer = recent_values.buffer();
        let mut recent_buffer = recent_buffer.borrow_mut();

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            let recent_index = destination_index % recent_capacity;
            for group_index in 0..self.shape[0] {
                let source_start =
                    source_values.offset() + group_index * source_group_stride + source_index * row_bytes;
                let destination_start =
                    recent_values.offset() + group_index * recent_group_stride + recent_index * row_bytes;
                encoder.encode_copy(
                    &*source_buffer,
                    source_start..source_start + row_bytes,
                    &mut *recent_buffer,
                    destination_start..destination_start + row_bytes,
                );
            }
        }
    }

    fn copy_sparse_value_recent_values_from_pending(
        &self,
        source_base: usize,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        let Some(recent_values) = &self.sparse_value_recent_values else {
            return;
        };
        let Some(pending_values) = &self.sparse_value_pending_values else {
            return;
        };
        assert_eq!(
            source_indices.len(),
            destination_indices.len(),
            "SparseValue recent-value source/destination lengths must match"
        );
        if source_indices.is_empty() {
            return;
        }
        let pending_values = pending_values.borrow();
        let recent_values = &mut *recent_values.borrow_mut();
        assert_eq!(
            pending_values.data_type(),
            recent_values.data_type(),
            "SparseValue pending/recent value dtype mismatch"
        );
        let head_dim = self.shape[2];
        let pending_group_stride = pending_values.shape()[1] * head_dim;
        let recent_group_stride = recent_values.shape()[1] * head_dim;
        let recent_capacity = recent_values.shape()[1];

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            let pending_index = source_index
                .checked_sub(source_base)
                .expect("SparseValue recent-value source index must be inside the pending suffix");
            assert!(
                pending_index < pending_values.shape()[1],
                "SparseValue recent-value source index exceeds the pending suffix length"
            );
            let recent_index = destination_index % recent_capacity;
            for group_index in 0..self.shape[0] {
                let source_start = group_index * pending_group_stride + pending_index * head_dim;
                let destination_start = group_index * recent_group_stride + recent_index * head_dim;
                match recent_values.data_type() {
                    DataType::BF16 => recent_values.as_slice_mut::<half::bf16>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(
                            &pending_values.as_slice::<half::bf16>()[source_start..source_start + head_dim],
                        ),
                    DataType::F16 => recent_values.as_slice_mut::<half::f16>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(
                            &pending_values.as_slice::<half::f16>()[source_start..source_start + head_dim],
                        ),
                    DataType::F32 => recent_values.as_slice_mut::<f32>()
                        [destination_start..destination_start + head_dim]
                        .copy_from_slice(&pending_values.as_slice::<f32>()[source_start..source_start + head_dim]),
                    dtype => panic!("SparseValue does not support recent-value dtype {dtype:?}"),
                }
            }
        }
    }

    fn encode_sparse_value_recent_values_from_pending(
        &self,
        source_base: usize,
        source_indices: &[usize],
        destination_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) {
        let Some(recent_values) = &self.sparse_value_recent_values else {
            return;
        };
        let Some(pending_values) = &self.sparse_value_pending_values else {
            return;
        };
        assert_eq!(
            source_indices.len(),
            destination_indices.len(),
            "SparseValue recent-value source/destination lengths must match"
        );
        if source_indices.is_empty() {
            return;
        }
        let pending_values = pending_values.borrow();
        let recent_values = recent_values.borrow();
        assert_eq!(
            pending_values.data_type(),
            recent_values.data_type(),
            "SparseValue pending/recent value dtype mismatch"
        );
        let head_dim = self.shape[2];
        let row_bytes = head_dim * pending_values.data_type().size_in_bytes();
        let pending_group_stride = pending_values.shape()[1] * row_bytes;
        let recent_group_stride = recent_values.shape()[1] * row_bytes;
        let recent_capacity = recent_values.shape()[1];
        let pending_buffer = pending_values.buffer();
        let pending_buffer = pending_buffer.borrow();
        let recent_buffer = recent_values.buffer();
        let mut recent_buffer = recent_buffer.borrow_mut();

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            let pending_index = source_index
                .checked_sub(source_base)
                .expect("SparseValue recent-value source index must be inside the pending suffix");
            assert!(
                pending_index < pending_values.shape()[1],
                "SparseValue recent-value source index exceeds the pending suffix length"
            );
            let recent_index = destination_index % recent_capacity;
            for group_index in 0..self.shape[0] {
                let source_start =
                    pending_values.offset() + group_index * pending_group_stride + pending_index * row_bytes;
                let destination_start =
                    recent_values.offset() + group_index * recent_group_stride + recent_index * row_bytes;
                encoder.encode_copy(
                    &*pending_buffer,
                    source_start..source_start + row_bytes,
                    &mut *recent_buffer,
                    destination_start..destination_start + row_bytes,
                );
            }
        }
    }

    pub fn encode_sparse_value_pending_values_from_source(
        &mut self,
        source_values: &Array<B>,
        prefix_length: usize,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) {
        let Some(pending_values) = &self.sparse_value_pending_values else {
            return;
        };
        if suffix_length == 0 {
            return;
        }
        let pending_values = pending_values.borrow();
        assert_eq!(
            source_values.data_type(),
            pending_values.data_type(),
            "SparseValue source/pending value dtype mismatch"
        );
        assert!(
            prefix_length + suffix_length <= source_values.shape()[1],
            "SparseValue source rows must fit inside the value source"
        );
        assert!(suffix_length <= pending_values.shape()[1], "SparseValue suffix length exceeds pending-value capacity");
        let head_dim = self.shape[2];
        let row_bytes = head_dim * source_values.data_type().size_in_bytes();
        let source_group_stride = source_values.shape()[1] * row_bytes;
        let pending_group_stride = pending_values.shape()[1] * row_bytes;
        let source_buffer = source_values.buffer();
        let source_buffer = source_buffer.borrow();
        let pending_buffer = pending_values.buffer();
        let mut pending_buffer = pending_buffer.borrow_mut();

        for group_index in 0..self.shape[0] {
            let source_start = source_values.offset() + group_index * source_group_stride + prefix_length * row_bytes;
            let destination_start = pending_values.offset() + group_index * pending_group_stride;
            let byte_len = suffix_length * row_bytes;
            encoder.encode_copy(
                &*source_buffer,
                source_start..source_start + byte_len,
                &mut *pending_buffer,
                destination_start..destination_start + byte_len,
            );
        }
        self.sparse_value.as_mut().expect("SparseValue state must exist").pending_values_ready_on_cpu = false;
    }

    pub fn register_accepted_tokens(
        &mut self,
        number_of_tokens: usize,
    ) {
        if number_of_tokens == 0 {
            return;
        }
        if let Some(triattention) = &mut self.triattention {
            triattention.tokens_since_last_prune += number_of_tokens;
        }
        match &mut self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => {
                let start = self.next_token_position;
                let end = start + number_of_tokens;
                self.prefix_token_positions.extend(start..end);
                self.next_token_position = end;
                *prefix_len = self.prefix_token_positions.len();
            },
            KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                for _ in 0..number_of_tokens {
                    let token_pos = self.next_token_position;
                    self.next_token_position = self.next_token_position.saturating_add(1);
                    if *ring_length < *window_length {
                        let dst = (*ring_offset + *ring_length) % *window_length;
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

    pub fn stage_sparse_value_suffix_rows(
        &mut self,
        rotated_keys: &Array<B>,
        qkv: &Array<B>,
        num_heads: usize,
    ) {
        let Some(sparse_value) = &mut self.sparse_value else {
            return;
        };
        let mut pending_values =
            self.sparse_value_pending_values.as_ref().expect("SparseValue pending values must exist").borrow_mut();
        let suffix_length = qkv.shape()[0];
        assert!(suffix_length <= self.max_suffix_length, "SparseValue suffix length exceeds layer capacity");
        assert_eq!(rotated_keys.shape()[0], self.shape[0], "SparseValue rotated key group count mismatch");
        assert_eq!(rotated_keys.shape()[1], suffix_length, "SparseValue rotated key suffix length mismatch");
        assert_eq!(rotated_keys.shape()[2], self.shape[2], "SparseValue rotated key head dim mismatch");
        assert_eq!(rotated_keys.data_type(), qkv.data_type(), "SparseValue suffix staging dtype mismatch");
        match qkv.data_type() {
            DataType::BF16 => stage_sparse_value_suffix_rows_typed::<B, half::bf16>(
                rotated_keys,
                qkv,
                self.shape[0],
                num_heads,
                self.shape[2],
                self.max_suffix_length,
                &mut sparse_value.pending_keys,
                &mut pending_values,
            ),
            DataType::F16 => stage_sparse_value_suffix_rows_typed::<B, half::f16>(
                rotated_keys,
                qkv,
                self.shape[0],
                num_heads,
                self.shape[2],
                self.max_suffix_length,
                &mut sparse_value.pending_keys,
                &mut pending_values,
            ),
            DataType::F32 => stage_sparse_value_suffix_rows_typed::<B, f32>(
                rotated_keys,
                qkv,
                self.shape[0],
                num_heads,
                self.shape[2],
                self.max_suffix_length,
                &mut sparse_value.pending_keys,
                &mut pending_values,
            ),
            dtype => panic!("SparseValue does not support suffix dtype {dtype:?}"),
        }
        sparse_value.pending_suffix_len = suffix_length;
        sparse_value.pending_values_ready_on_cpu = true;
    }

    pub fn update_triattention_query_stats(
        &mut self,
        qkv: &Array<B>,
        active_row_count: usize,
    ) {
        let Some(triattention) = &mut self.triattention else {
            return;
        };
        if active_row_count == 0 {
            return;
        }

        match qkv.data_type() {
            DataType::BF16 => {
                update_triattention_query_stats_typed::<B, half::bf16>(qkv, active_row_count, triattention)
            },
            DataType::F16 => update_triattention_query_stats_typed::<B, half::f16>(qkv, active_row_count, triattention),
            DataType::F32 => update_triattention_query_stats_typed::<B, f32>(qkv, active_row_count, triattention),
            dtype => panic!("TriAttention does not support query dtype {dtype:?}"),
        }
    }

    pub fn prune_triattention_if_needed(
        &mut self,
        rope_cosines: &Array<B>,
        rope_sines: &Array<B>,
    ) {
        let Some(triattention) = self.triattention.as_ref() else {
            return;
        };
        let prefix_len = match self.state {
            KVCacheLayerState::Full {
                prefix_len,
            } => prefix_len,
            KVCacheLayerState::Windowed {
                ..
            } => return,
        };
        if prefix_len <= triattention.config.budget
            || triattention.tokens_since_last_prune < triattention.config.divide_length
        {
            return;
        }

        let keep_indices = {
            let keys = self.dense_keys().borrow();
            triattention_keep_indices(
                &keys,
                rope_cosines,
                rope_sines,
                &self.prefix_token_positions,
                self.next_token_position,
                prefix_len,
                triattention.config.budget,
                &triattention.calibration,
            )
        };
        copy_dense_rows_to_front(self.dense_keys(), &keep_indices);
        copy_dense_rows_to_front(self.dense_values(), &keep_indices);
        compact_positions_to_front(&mut self.prefix_token_positions, &keep_indices);
        let KVCacheLayerState::Full {
            prefix_len,
        } = &mut self.state
        else {
            panic!("TriAttention only supports full KV cache layers");
        };
        *prefix_len = keep_indices.len();
        self.triattention.as_mut().expect("TriAttention state must exist").tokens_since_last_prune = 0;
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
                    .then_some((ring_offset + ring_length + window_length - 1) % window_length)
                    .unwrap_or(0);
                let unmask_col = (ring_length > 0).then_some(newest_slot as i32).unwrap_or(-1);
                let mask_col = (ring_length == window_length).then_some(ring_offset as i32).unwrap_or(-1);

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
        context: &B::Context,
        range: std::ops::Range<usize>,
    ) -> Option<KVSlice<B>> {
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

                let positions: Vec<usize> = slots.iter().map(|&s| self.prefix_token_positions[s]).collect();
                let slice_shape = [self.shape[0], len, self.shape[2]];
                let mut slice_keys = context.create_array(&slice_shape, self.data_type, "kv_cache_layer_slice_keys");
                let mut slice_values =
                    context.create_array(&slice_shape, self.data_type, "kv_cache_layer_slice_values");

                if !self.uses_compressed_storage() {
                    let keys = self.dense_keys().borrow();
                    let values = self.dense_values().borrow();
                    for (i, &slot) in slots.iter().enumerate() {
                        slice_keys.copy_slice(&keys, 1, slot..slot + 1, i);
                        slice_values.copy_slice(&values, 1, slot..slot + 1, i);
                    }
                } else {
                    let (keys, values) = self.create_materialized_storage(context);
                    for (i, &slot) in slots.iter().enumerate() {
                        slice_keys.copy_slice(&keys, 1, slot..slot + 1, i);
                        slice_values.copy_slice(&values, 1, slot..slot + 1, i);
                    }
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
        context: &B::Context,
        slice: &KVSlice<B>,
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
                    let keep_positions = base_positions_len.saturating_add(accepted);
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
                let copy_range = match range {
                    None => {
                        *ring_offset = *base_ring_offset;
                        *ring_length = *base_ring_length;
                        0..slots.len()
                    },
                    Some(r) => {
                        if r.end.saturating_sub(r.start) == 0 {
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
                        r
                    },
                };

                for (index, &slot) in slots[copy_range.clone()].iter().enumerate() {
                    let src_index = copy_range.start + index;
                    self.prefix_token_positions[slot] = positions[src_index];
                }

                if !self.uses_compressed_storage() {
                    let mut dst_keys = self.dense_keys().borrow_mut();
                    let mut dst_values = self.dense_values().borrow_mut();
                    for (index, &slot) in slots[copy_range.clone()].iter().enumerate() {
                        let src_index = copy_range.start + index;
                        dst_keys.copy_slice(keys, 1, src_index..src_index + 1, slot);
                        dst_values.copy_slice(values, 1, src_index..src_index + 1, slot);
                    }
                } else {
                    let (mut dst_keys, mut dst_values) = self.create_materialized_storage(context);
                    for (index, &slot) in slots[copy_range.clone()].iter().enumerate() {
                        let src_index = copy_range.start + index;
                        dst_keys.copy_slice(keys, 1, src_index..src_index + 1, slot);
                        dst_values.copy_slice(values, 1, src_index..src_index + 1, slot);
                    }
                    self.compress_from(&dst_keys, &dst_values);
                }
            },
            _ => {},
        }

        let max_position = self.prefix_token_positions.iter().copied().filter(|&pos| pos != INVALID_POSITION).max();
        self.next_token_position = max_position.map_or(0, |pos| pos.saturating_add(1));
    }
}

fn stage_sparse_value_suffix_rows_typed<B: Backend, T>(
    rotated_keys: &Array<B>,
    qkv: &Array<B>,
    num_groups: usize,
    num_heads: usize,
    head_dim: usize,
    max_suffix_length: usize,
    pending_keys: &mut [f32],
    pending_values: &mut Array<B>,
) where
    T: crate::ArrayElement + Copy,
    f32: From<T>,
{
    let suffix_length = qkv.shape()[0];
    let rotated_key_rows = rotated_keys.as_slice::<T>();
    let qkv_rows = qkv.as_slice::<T>();
    let qkv_stride = qkv.shape()[1];
    let value_base = (num_heads + num_groups) * head_dim;
    let pending_values = pending_values.as_slice_mut::<T>();

    for token_index in 0..suffix_length {
        for group_index in 0..num_groups {
            let pending_base = (group_index * max_suffix_length + token_index) * head_dim;
            let key_base = (group_index * suffix_length + token_index) * head_dim;
            let value_row_base = token_index * qkv_stride + value_base + group_index * head_dim;

            for (dst, &src) in pending_keys[pending_base..pending_base + head_dim]
                .iter_mut()
                .zip(rotated_key_rows[key_base..key_base + head_dim].iter())
            {
                *dst = f32::from(src);
            }
            pending_values[pending_base..pending_base + head_dim]
                .copy_from_slice(&qkv_rows[value_row_base..value_row_base + head_dim]);
        }
    }
}

fn copy_dense_rows<B: Backend>(
    rows: &ArrayCell<B>,
    source_indices: &[usize],
    destination_indices: &[usize],
) {
    if source_indices == destination_indices {
        return;
    }

    let source = rows.borrow().clone();
    let mut destination = rows.borrow_mut();
    for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
        destination.copy_slice(&source, 1, source_index..source_index + 1, destination_index);
    }
}

fn copy_dense_rows_to_front<B: Backend>(
    rows: &ArrayCell<B>,
    source_indices: &[usize],
) {
    if source_indices.iter().enumerate().all(|(destination_index, &source_index)| source_index == destination_index) {
        return;
    }

    let source = rows.borrow().clone();
    let mut destination = rows.borrow_mut();
    let mut run_start = 0;
    while run_start < source_indices.len() {
        let mut run_end = run_start + 1;
        while run_end < source_indices.len() {
            let destination_delta = run_end - run_start;
            let source_delta = source_indices[run_end] - source_indices[run_start];
            if source_delta != destination_delta {
                break;
            }
            run_end += 1;
        }
        destination.copy_slice(
            &source,
            1,
            source_indices[run_start]..source_indices[run_start] + (run_end - run_start),
            run_start,
        );
        run_start = run_end;
    }
}

fn compact_positions_to_front(
    positions: &mut Vec<usize>,
    source_indices: &[usize],
) {
    if source_indices.iter().enumerate().all(|(destination_index, &source_index)| source_index == destination_index) {
        positions.truncate(source_indices.len());
        return;
    }

    for (destination_index, &source_index) in source_indices.iter().enumerate() {
        positions[destination_index] = positions[source_index];
    }
    positions.truncate(source_indices.len());
}

fn triattention_keep_indices<B: Backend>(
    keys: &Array<B>,
    rope_cosines: &Array<B>,
    rope_sines: &Array<B>,
    prefix_positions: &[usize],
    next_token_position: usize,
    prefix_len: usize,
    budget: usize,
    calibration: &TriAttentionCalibration,
) -> Vec<usize> {
    let mut indices = (0..prefix_len).collect::<Vec<_>>();
    if prefix_len <= budget {
        return indices;
    }

    assert!(calibration.query_token_count > 0, "TriAttention calibration requires at least one query token");
    assert_eq!(prefix_positions.len(), prefix_len, "TriAttention prefix positions must match prefix length");

    let scores = match (keys.data_type(), rope_cosines.data_type()) {
        (DataType::BF16, DataType::BF16) => triattention_scores_typed::<B, half::bf16, half::bf16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::BF16, DataType::F16) => triattention_scores_typed::<B, half::bf16, half::f16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::BF16, DataType::F32) => triattention_scores_typed::<B, half::bf16, f32>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F16, DataType::BF16) => triattention_scores_typed::<B, half::f16, half::bf16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F16, DataType::F16) => triattention_scores_typed::<B, half::f16, half::f16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F16, DataType::F32) => triattention_scores_typed::<B, half::f16, f32>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F32, DataType::BF16) => triattention_scores_typed::<B, f32, half::bf16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F32, DataType::F16) => triattention_scores_typed::<B, f32, half::f16>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (DataType::F32, DataType::F32) => triattention_scores_typed::<B, f32, f32>(
            keys,
            rope_cosines,
            rope_sines,
            prefix_positions,
            next_token_position,
            prefix_len,
            calibration,
        ),
        (key_dtype, rope_dtype) => {
            panic!("TriAttention does not support KV dtype {key_dtype:?} with rope dtype {rope_dtype:?}")
        },
    };

    indices.sort_unstable_by(|&left, &right| {
        scores[right].partial_cmp(&scores[left]).expect("TriAttention scores must be finite")
    });
    indices.truncate(budget);
    indices.sort_unstable();
    indices
}

fn update_triattention_query_stats_typed<B: Backend, T>(
    qkv: &Array<B>,
    active_row_count: usize,
    triattention: &mut TriAttentionState,
) where
    T: crate::ArrayElement + Copy,
    f32: From<T>,
{
    let calibration = &mut triattention.calibration;
    let qkv_stride = qkv.shape()[1];
    let qkv = qkv.as_slice::<T>();
    let head_dim = calibration.head_dim;
    let rope_pair_count = calibration.rope_dim / 2;
    let half_rope_dim = rope_pair_count;

    for token_index in 0..active_row_count {
        let token_base = token_index * qkv_stride;
        for head_index in 0..calibration.num_heads {
            let query_base = token_base + head_index * head_dim;
            let query_sum_base = head_index * head_dim;
            let query_norm_base = head_index * rope_pair_count;

            for dim_index in 0..head_dim {
                calibration.query_sum[query_sum_base + dim_index] += f32::from(qkv[query_base + dim_index]);
            }

            for pair_index in 0..rope_pair_count {
                let real = f32::from(qkv[query_base + pair_index]);
                let imag = f32::from(qkv[query_base + pair_index + half_rope_dim]);
                calibration.query_norm_sum[query_norm_base + pair_index] += (real * real + imag * imag).sqrt();
            }
        }
    }

    calibration.query_token_count += active_row_count;
}

fn triattention_scores_typed<B: Backend, K, R>(
    keys: &Array<B>,
    rope_cosines: &Array<B>,
    rope_sines: &Array<B>,
    prefix_positions: &[usize],
    next_token_position: usize,
    prefix_len: usize,
    calibration: &TriAttentionCalibration,
) -> Vec<f32>
where
    K: crate::ArrayElement + Copy,
    R: crate::ArrayElement + Copy,
    f32: From<K> + From<R>,
{
    const FUTURE_OFFSETS: [usize; 17] =
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];

    let key_shape = keys.shape();
    let num_groups = key_shape[0];
    let total_len = key_shape[1];
    let head_dim = key_shape[2];
    let rope_dim = calibration.rope_dim;
    let half_rope_dim = rope_dim / 2;
    let gqa_factor = calibration.num_heads / num_groups;
    let rope_max_seq_len = rope_cosines.shape()[0];

    let keys = keys.as_slice::<K>();
    let rope_cosines = rope_cosines.as_slice::<R>();
    let rope_sines = rope_sines.as_slice::<R>();

    let future_positions = FUTURE_OFFSETS
        .into_iter()
        .map(|offset| next_token_position.saturating_add(offset))
        .filter(|&position| position < rope_max_seq_len)
        .collect::<Vec<_>>();
    assert!(!future_positions.is_empty(), "TriAttention requires at least one valid future position");

    let averaged_queries = averaged_future_queries::<R>(rope_cosines, rope_sines, &future_positions, calibration);
    let norm_weights = triattention_norm_weights(calibration);
    let mut final_scores = vec![f32::NEG_INFINITY; prefix_len];
    let rope_pair_count = rope_dim / 2;

    for group_index in 0..num_groups {
        for group_head_offset in 0..gqa_factor {
            let head_index = group_index * gqa_factor + group_head_offset;
            let query_base = head_index * head_dim;
            let norm_base = head_index * rope_pair_count;
            let averaged_query = &averaged_queries[query_base..query_base + head_dim];
            let head_norm_weights = &norm_weights[norm_base..norm_base + rope_pair_count];
            let raw_scores = (0..prefix_len)
                .map(|token_index| {
                    let key_base = (group_index * total_len + token_index) * head_dim;
                    let key_row = &keys[key_base..key_base + head_dim];
                    let trig_score = averaged_query
                        .iter()
                        .zip(key_row.iter())
                        .map(|(&query_value, &key_value)| query_value * f32::from(key_value))
                        .sum::<f32>();
                    let norm_score = (0..rope_pair_count)
                        .map(|pair_index| {
                            let real = f32::from(key_row[pair_index]);
                            let imag = f32::from(key_row[pair_index + half_rope_dim]);
                            head_norm_weights[pair_index] * (real * real + imag * imag).sqrt()
                        })
                        .sum::<f32>();
                    trig_score + norm_score
                })
                .collect::<Vec<_>>();

            let mean = raw_scores.iter().sum::<f32>() / raw_scores.len() as f32;
            let variance = raw_scores
                .iter()
                .map(|&score| {
                    let centered = score - mean;
                    centered * centered
                })
                .sum::<f32>()
                / raw_scores.len() as f32;
            let std = variance.sqrt();

            for token_index in 0..prefix_len {
                let centered = raw_scores[token_index] - mean;
                let normalized = if std > 0.0 {
                    centered / std
                } else {
                    centered
                };
                final_scores[token_index] = final_scores[token_index].max(normalized);
            }
        }
    }

    for (token_index, &position) in prefix_positions.iter().enumerate() {
        assert!(position != INVALID_POSITION, "TriAttention prefix positions must be valid");
        assert!(final_scores[token_index].is_finite(), "TriAttention scores must be finite");
    }

    final_scores
}

fn averaged_future_queries<R>(
    rope_cosines: &[R],
    rope_sines: &[R],
    future_positions: &[usize],
    calibration: &TriAttentionCalibration,
) -> Box<[f32]>
where
    R: crate::ArrayElement + Copy,
    f32: From<R>,
{
    let mut averaged = vec![0.0; calibration.num_heads * calibration.head_dim];
    let rope_dim = calibration.rope_dim;
    let half_rope_dim = rope_dim / 2;

    for head_index in 0..calibration.num_heads {
        let query_base = head_index * calibration.head_dim;
        let query_sum = &calibration.query_sum[query_base..query_base + calibration.head_dim];
        let dst = &mut averaged[query_base..query_base + calibration.head_dim];

        for &future_position in future_positions {
            let rope_base = future_position * rope_dim;
            for dim_index in 0..rope_dim {
                let q_value = query_sum[dim_index] / calibration.query_token_count as f32;
                let q_paired = if dim_index < half_rope_dim {
                    -(query_sum[dim_index + half_rope_dim] / calibration.query_token_count as f32)
                } else {
                    query_sum[dim_index - half_rope_dim] / calibration.query_token_count as f32
                };
                let cos = f32::from(rope_cosines[rope_base + dim_index]);
                let sin = f32::from(rope_sines[rope_base + dim_index]);
                dst[dim_index] += q_value * cos + q_paired * sin;
            }
            for dim_index in rope_dim..calibration.head_dim {
                dst[dim_index] += query_sum[dim_index] / calibration.query_token_count as f32;
            }
        }

        for value in dst.iter_mut() {
            *value /= future_positions.len() as f32;
        }
    }

    averaged.into_boxed_slice()
}

fn triattention_norm_weights(calibration: &TriAttentionCalibration) -> Box<[f32]> {
    let rope_pair_count = calibration.rope_dim / 2;
    let mut weights = vec![0.0; calibration.num_heads * rope_pair_count];
    for head_index in 0..calibration.num_heads {
        let query_base = head_index * calibration.head_dim;
        let norm_base = head_index * rope_pair_count;
        for pair_index in 0..rope_pair_count {
            let mean_real = calibration.query_sum[query_base + pair_index] / calibration.query_token_count as f32;
            let mean_imag =
                calibration.query_sum[query_base + pair_index + rope_pair_count] / calibration.query_token_count as f32;
            let mean_norm = (mean_real * mean_real + mean_imag * mean_imag).sqrt();
            let expected_norm =
                calibration.query_norm_sum[norm_base + pair_index] / calibration.query_token_count as f32;
            weights[norm_base + pair_index] = (expected_norm - mean_norm).max(0.0);
        }
    }
    weights.into_boxed_slice()
}

#[cfg(test)]
mod sparse_value_tests {
    use std::{
        cell::RefCell,
        ops::{Deref, DerefMut},
    };

    use crate::{
        ArrayContextExt, DataType,
        backends::{
            common::{
                Backend, Context, Encoder, Kernels,
                kernel::{AttentionUpdateKVCacheKernel, kv_cache_update::KVCacheUpdate},
            },
            cpu::Cpu,
            metal::Metal,
        },
        forward_pass::kv_cache_layer::{KVCacheLayerState, KvCompressionMode},
    };

    use super::{KVCacheLayer, SparseValueConfig, SparseValueState, stage_sparse_value_suffix_rows_typed};

    #[test]
    fn sparse_value_stage_suffix_rows_captures_group_keys() {
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let rotated_keys_data: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        let mut qkv_data = vec![0.0f32; 2 * 16];
        qkv_data[12..16].copy_from_slice(&[100.0, 101.0, 200.0, 201.0]);
        qkv_data[28..32].copy_from_slice(&[110.0, 111.0, 210.0, 211.0]);
        let rotated_keys = context.create_array_from(&[2, 2, 2], &rotated_keys_data, "rotated_keys");
        let qkv = context.create_array_from(&[2, 16], &qkv_data, "qkv");
        let mut pending_keys = vec![0.0; 2 * 4 * 2];
        let mut pending_values = context.create_array_zeros(&[2, 4, 2], DataType::F32, "pending_values");

        stage_sparse_value_suffix_rows_typed::<Cpu, f32>(
            &rotated_keys,
            &qkv,
            2,
            4,
            2,
            4,
            &mut pending_keys,
            &mut pending_values,
        );

        assert_eq!(&pending_keys[0..4], &[10.0, 11.0, 20.0, 21.0]);
        assert_eq!(&pending_keys[8..12], &[30.0, 31.0, 40.0, 41.0]);
        assert_eq!(
            pending_values.as_slice::<f32>(),
            &[100.0, 101.0, 110.0, 111.0, 0.0, 0.0, 0.0, 0.0, 200.0, 201.0, 210.0, 211.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn sparse_value_acceptance_updates_shadow_keys() {
        let sparse_value = SparseValueState::new(
            SparseValueConfig {
                recent_window: 4,
                keep_mass: 1.0,
                page_size: 2,
            },
            1,
            8,
            2,
            4,
        );
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let mut layer = KVCacheLayer::<Cpu> {
            state: KVCacheLayerState::Full {
                prefix_len: 5,
            },
            shape: [1, 8, 2],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(&[1, 8, 2], DataType::F32, "keys"))),
            values: Some(RefCell::new(context.create_array_zeros(&[1, 8, 2], DataType::F32, "values"))),
            prefix_token_positions: vec![0, 1, 2, 3, 4],
            next_token_position: 5,
            max_suffix_length: 4,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(sparse_value),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 4, 2],
                DataType::F32,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 6, 2],
                DataType::F32,
                "recent_values",
            ))),
            triattention: None,
        };
        let state = layer.sparse_value.as_mut().expect("sparse state");
        state.pending_suffix_len = 2;
        state.pending_keys[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        layer.update_sparse_value_from_pending(5, &[5, 6], &[5, 6]);

        let state = layer.sparse_value.as_ref().expect("sparse state");
        assert_eq!(&state.shadow_keys[10..14], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sparse_value_recent_values_track_sequential_prefix_rows() {
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let values =
            context.create_array_from(&[1, 80, 2], &(0..160).map(|index| index as f32).collect::<Vec<_>>(), "values");
        let layer = KVCacheLayer::<Cpu> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [1, 80, 2],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(&[1, 80, 2], DataType::F32, "keys"))),
            values: Some(RefCell::new(values)),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length: 16,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: 64,
                    keep_mass: 1.0,
                    page_size: 2,
                },
                1,
                80,
                2,
                16,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 16, 2],
                DataType::F32,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 64, 2],
                DataType::F32,
                "recent_values",
            ))),
            triattention: None,
        };
        let mut source_values = layer.dense_values().borrow().clone();
        for token_index in 0..80 {
            source_values.as_slice_mut::<f32>()[token_index * 2..token_index * 2 + 2]
                .copy_from_slice(&[token_index as f32 * 2.0, token_index as f32 * 2.0 + 1.0]);
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.encode_sparse_value_recent_values_from_source(
                &source_values,
                &[token_index],
                &[token_index],
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
        }

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<f32>();
        let recent_stride = 64 * 2;
        for token_index in 16..80 {
            let slot = token_index % 64;
            let base = slot * 2;
            let expected = [token_index as f32 * 2.0, token_index as f32 * 2.0 + 1.0];
            assert_eq!(&recent_values[base..base + 2], &expected);
        }
        assert_eq!(&recent_values[recent_stride - 2..recent_stride], &[126.0, 127.0]);
    }

    #[test]
    fn sparse_value_recent_values_follow_pending_suffix_rows() {
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let mut layer = KVCacheLayer::<Cpu> {
            state: KVCacheLayerState::Full {
                prefix_len: 5,
            },
            shape: [1, 8, 2],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(&[1, 8, 2], DataType::F32, "keys"))),
            values: Some(RefCell::new(context.create_array_zeros(&[1, 8, 2], DataType::F32, "values"))),
            prefix_token_positions: vec![0, 1, 2, 3, 4],
            next_token_position: 5,
            max_suffix_length: 1,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: 8,
                    keep_mass: 1.0,
                    page_size: 2,
                },
                1,
                8,
                2,
                1,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 1, 2],
                DataType::F32,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[1, 8, 2],
                DataType::F32,
                "recent_values",
            ))),
            triattention: None,
        };
        let pending_source = context.create_array_from(
            &[1, 8, 2],
            &[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.0, 102.0, 0.0, 0.0, 0.0, 0.0],
            "pending_source",
        );
        let materialized_values = context.create_array_from(
            &[1, 8, 2],
            &[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            "materialized_values",
        );
        let kv_cache_update = KVCacheUpdate::new(context.as_ref(), DataType::F32, 8).expect("kv cache update");
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");

        layer.sparse_value.as_mut().expect("sparse state").pending_suffix_len = 1;
        layer.encode_sparse_value_pending_values_from_source(&pending_source, 5, 1, &mut encoder);
        layer.update_after_acceptance(
            &context,
            &[0],
            None,
            None,
            Some(&materialized_values),
            &mut encoder,
            &kv_cache_update,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        assert_eq!(&recent_values.as_slice::<f32>()[10..12], &[101.0, 102.0]);
    }

    #[test]
    fn sparse_value_recent_values_track_prefill_rows_on_metal() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let suffix_length = 64;
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let rotated_keys =
            context.create_array_zeros(&[num_groups, suffix_length, head_dim], DataType::F32, "rotated_keys");
        let qkv = context.create_array_from(
            &[suffix_length, qkv_stride],
            &(0..suffix_length)
                .flat_map(|token_index| {
                    let token_base = token_index as f32 * 100.0;
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        token_base + 10.0,
                        token_base + 11.0,
                        token_base + 20.0,
                        token_base + 21.0,
                    ]
                })
                .collect::<Vec<_>>(),
            "qkv",
        );
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, 256, head_dim],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(&[num_groups, 256, head_dim], DataType::F32, "keys"))),
            values: None,
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length: suffix_length,
            compression_mode: KvCompressionMode::SpectralQuant,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: 256,
                    keep_mass: 1.0,
                    page_size: 32,
                },
                num_groups,
                256,
                head_dim,
                suffix_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, suffix_length, head_dim],
                DataType::F32,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, 256, head_dim],
                DataType::F32,
                "recent_values",
            ))),
            triattention: None,
        };

        let mut source_values =
            context.create_array_zeros(&[num_groups, 256, head_dim], DataType::F32, "source_values");
        for token_index in 0..suffix_length {
            let token_base = token_index * head_dim;
            source_values.as_slice_mut::<f32>()[token_base..token_base + head_dim]
                .copy_from_slice(&[token_index as f32 * 100.0 + 10.0, token_index as f32 * 100.0 + 11.0]);
            let group_base = 256 * head_dim + token_base;
            source_values.as_slice_mut::<f32>()[group_base..group_base + head_dim]
                .copy_from_slice(&[token_index as f32 * 100.0 + 20.0, token_index as f32 * 100.0 + 21.0]);
        }
        layer.stage_sparse_value_suffix_rows(&rotated_keys, &qkv, num_heads);
        let accepted_indices = (0..suffix_length).collect::<Vec<_>>();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        layer.encode_sparse_value_recent_values_from_source(
            &source_values,
            &accepted_indices,
            &accepted_indices,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<f32>();
        for token_index in 0..suffix_length {
            let token_base = token_index * head_dim;
            assert_eq!(
                &recent_values[token_base..token_base + head_dim],
                &[token_index as f32 * 100.0 + 10.0, token_index as f32 * 100.0 + 11.0]
            );
            let group_base = 256 * head_dim + token_base;
            assert_eq!(
                &recent_values[group_base..group_base + head_dim],
                &[token_index as f32 * 100.0 + 20.0, token_index as f32 * 100.0 + 21.0]
            );
        }
    }

    #[test]
    fn sparse_value_recent_values_track_prefill_then_single_token_updates_on_metal() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let hot_value_capacity = 512;
        let prompt_length = 154;
        let generated_length = 64;
        let max_suffix_length = prompt_length;
        let total_length = hot_value_capacity + max_suffix_length;
        let kv_cache_update =
            KVCacheUpdate::new(context.as_ref(), DataType::F32, total_length).expect("kv cache update");
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let make_qkv = |token_start: usize, token_count: usize| {
            context.create_array_from(
                &[token_count, qkv_stride],
                &(0..token_count)
                    .flat_map(|token_offset| {
                        let token_index = token_start + token_offset;
                        let token_base = token_index as f32 * 100.0;
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            token_base + 10.0,
                            token_base + 11.0,
                            token_base + 20.0,
                            token_base + 21.0,
                        ]
                    })
                    .collect::<Vec<_>>(),
                "qkv",
            )
        };
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, total_length, head_dim],
            data_type: DataType::F32,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::F32,
                "keys",
            ))),
            values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::F32,
                "values",
            ))),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: hot_value_capacity,
                    keep_mass: 1.0,
                    page_size: 32,
                },
                num_groups,
                total_length,
                head_dim,
                max_suffix_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, max_suffix_length, head_dim],
                DataType::F32,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::F32,
                "recent_values",
            ))),
            triattention: None,
        };
        let rotated_keys =
            context.create_array_zeros(&[num_groups, max_suffix_length, head_dim], DataType::F32, "rotated_keys");

        let prompt_qkv = make_qkv(0, prompt_length);
        layer.stage_sparse_value_suffix_rows(
            &rotated_keys.view(&[num_groups, prompt_length, head_dim]),
            &prompt_qkv,
            num_heads,
        );
        {
            let mut values = layer.dense_values().borrow_mut();
            for token_index in 0..prompt_length {
                let token_base = token_index * head_dim;
                values.as_slice_mut::<f32>()[token_base..token_base + head_dim]
                    .copy_from_slice(&[token_index as f32 * 100.0 + 10.0, token_index as f32 * 100.0 + 11.0]);
                let group_base = total_length * head_dim + token_base;
                values.as_slice_mut::<f32>()[group_base..group_base + head_dim]
                    .copy_from_slice(&[token_index as f32 * 100.0 + 20.0, token_index as f32 * 100.0 + 21.0]);
            }
        }
        let prompt_indices = (0..prompt_length).collect::<Vec<_>>();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        layer.update_after_acceptance(&context, &prompt_indices, Some(0), None, None, &mut encoder, &kv_cache_update);
        encoder.end_encoding().submit().wait_until_completed().expect("complete");
        layer.register_accepted_tokens(prompt_length);

        for token_index in prompt_length..prompt_length + generated_length {
            let qkv = make_qkv(token_index, 1);
            layer.stage_sparse_value_suffix_rows(&rotated_keys.view(&[num_groups, 1, head_dim]), &qkv, num_heads);
            {
                let mut values = layer.dense_values().borrow_mut();
                let token_base = token_index * head_dim;
                values.as_slice_mut::<f32>()[token_base..token_base + head_dim]
                    .copy_from_slice(&[token_index as f32 * 100.0 + 10.0, token_index as f32 * 100.0 + 11.0]);
                let group_base = total_length * head_dim + token_base;
                values.as_slice_mut::<f32>()[group_base..group_base + head_dim]
                    .copy_from_slice(&[token_index as f32 * 100.0 + 20.0, token_index as f32 * 100.0 + 21.0]);
            }
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.update_after_acceptance(&context, &[0], None, None, None, &mut encoder, &kv_cache_update);
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
            layer.register_accepted_tokens(1);
        }

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<f32>();
        for token_index in 0..prompt_length + generated_length {
            let token_base = token_index * head_dim;
            assert_eq!(
                &recent_values[token_base..token_base + head_dim],
                &[token_index as f32 * 100.0 + 10.0, token_index as f32 * 100.0 + 11.0]
            );
            let group_base = hot_value_capacity * head_dim + token_base;
            assert_eq!(
                &recent_values[group_base..group_base + head_dim],
                &[token_index as f32 * 100.0 + 20.0, token_index as f32 * 100.0 + 21.0]
            );
        }
    }

    #[test]
    fn sparse_value_recent_values_track_prefill_then_single_token_updates_on_metal_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let hot_value_capacity = 256;
        let prompt_length = 40;
        let generated_length = 64;
        let max_suffix_length = prompt_length;
        let total_length = hot_value_capacity + max_suffix_length;
        let kv_cache_update =
            KVCacheUpdate::new(context.as_ref(), DataType::BF16, total_length).expect("kv cache update");
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let make_qkv = |token_start: usize, token_count: usize| {
            context.create_array_from(
                &[token_count, qkv_stride],
                &(0..token_count)
                    .flat_map(|token_offset| {
                        let token_index = token_start + token_offset;
                        let token_base = token_index as f32 * 8.0;
                        [
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(token_base + 8.0),
                            half::bf16::from_f32(token_base + 12.0),
                            half::bf16::from_f32(token_base + 16.0),
                            half::bf16::from_f32(token_base + 20.0),
                        ]
                    })
                    .collect::<Vec<_>>(),
                "qkv",
            )
        };
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, total_length, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "keys",
            ))),
            values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "values",
            ))),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: hot_value_capacity,
                    keep_mass: 1.0,
                    page_size: 32,
                },
                num_groups,
                total_length,
                head_dim,
                max_suffix_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, max_suffix_length, head_dim],
                DataType::BF16,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::BF16,
                "recent_values",
            ))),
            triattention: None,
        };
        let rotated_keys =
            context.create_array_zeros(&[num_groups, max_suffix_length, head_dim], DataType::BF16, "rotated_keys");

        let prompt_qkv = make_qkv(0, prompt_length);
        layer.stage_sparse_value_suffix_rows(
            &rotated_keys.view(&[num_groups, prompt_length, head_dim]),
            &prompt_qkv,
            num_heads,
        );
        {
            let mut values = layer.dense_values().borrow_mut();
            for token_index in 0..prompt_length {
                let token_base = token_index * head_dim;
                let expected_base = token_index as f32 * 8.0;
                values.as_slice_mut::<half::bf16>()[token_base..token_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 8.0),
                    half::bf16::from_f32(expected_base + 12.0),
                ]);
                let group_base = total_length * head_dim + token_base;
                values.as_slice_mut::<half::bf16>()[group_base..group_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 16.0),
                    half::bf16::from_f32(expected_base + 20.0),
                ]);
            }
        }
        let prompt_indices = (0..prompt_length).collect::<Vec<_>>();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        layer.update_after_acceptance(&context, &prompt_indices, Some(0), None, None, &mut encoder, &kv_cache_update);
        encoder.end_encoding().submit().wait_until_completed().expect("complete");
        layer.register_accepted_tokens(prompt_length);

        for token_index in prompt_length..prompt_length + generated_length {
            let qkv = make_qkv(token_index, 1);
            layer.stage_sparse_value_suffix_rows(&rotated_keys.view(&[num_groups, 1, head_dim]), &qkv, num_heads);
            {
                let mut values = layer.dense_values().borrow_mut();
                let token_base = token_index * head_dim;
                let expected_base = token_index as f32 * 8.0;
                values.as_slice_mut::<half::bf16>()[token_base..token_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 8.0),
                    half::bf16::from_f32(expected_base + 12.0),
                ]);
                let group_base = total_length * head_dim + token_base;
                values.as_slice_mut::<half::bf16>()[group_base..group_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 16.0),
                    half::bf16::from_f32(expected_base + 20.0),
                ]);
            }
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.update_after_acceptance(&context, &[0], None, None, None, &mut encoder, &kv_cache_update);
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
            layer.register_accepted_tokens(1);
        }

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<half::bf16>();
        for token_index in 0..prompt_length + generated_length {
            let token_base = token_index * head_dim;
            let expected_base = token_index as f32 * 8.0;
            assert_eq!(recent_values[token_base].to_f32(), expected_base + 8.0);
            assert_eq!(recent_values[token_base + 1].to_f32(), expected_base + 12.0);
            let group_base = hot_value_capacity * head_dim + token_base;
            assert_eq!(recent_values[group_base].to_f32(), expected_base + 16.0);
            assert_eq!(recent_values[group_base + 1].to_f32(), expected_base + 20.0);
        }
    }

    #[test]
    fn sparse_value_recent_values_follow_gpu_written_prefill_rows_on_metal_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let suffix_length = 4;
        let hot_value_capacity = 16;
        let total_length = hot_value_capacity + suffix_length;
        let kv_cache_update =
            KVCacheUpdate::new(context.as_ref(), DataType::BF16, total_length).expect("kv cache update");
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let qkv = context.create_array_from(
            &[suffix_length, qkv_stride],
            &(0..suffix_length)
                .flat_map(|token_index| {
                    let token_base = token_index as f32 * 8.0;
                    [
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(0.0),
                        half::bf16::from_f32(token_base + 8.0),
                        half::bf16::from_f32(token_base + 12.0),
                        half::bf16::from_f32(token_base + 16.0),
                        half::bf16::from_f32(token_base + 20.0),
                    ]
                })
                .collect::<Vec<_>>(),
            "qkv",
        );
        let rotated_keys =
            context.create_array_zeros(&[num_groups, suffix_length, head_dim], DataType::BF16, "rotated_keys");
        let materialized_keys =
            context.create_array_zeros(&[num_groups, total_length, head_dim], DataType::BF16, "materialized_keys");
        let materialized_values =
            context.create_array_zeros(&[num_groups, total_length, head_dim], DataType::BF16, "materialized_values");
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, total_length, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "keys",
            ))),
            values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "values",
            ))),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length: suffix_length,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: hot_value_capacity,
                    keep_mass: 1.0,
                    page_size: 2,
                },
                num_groups,
                total_length,
                head_dim,
                suffix_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, suffix_length, head_dim],
                DataType::BF16,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::BF16,
                "recent_values",
            ))),
            triattention: None,
        };
        layer.stage_sparse_value_suffix_rows(&rotated_keys, &qkv, num_heads);

        let update_kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(
            &context,
            DataType::BF16,
            false,
        )
        .expect("attention update kernel");
        let rotated_keys_buffer = rotated_keys.buffer();
        let rotated_keys_buffer = rotated_keys_buffer.borrow();
        let qkv_buffer = qkv.buffer();
        let qkv_buffer = qkv_buffer.borrow();
        let materialized_keys_buffer = materialized_keys.buffer();
        let mut materialized_keys_buffer = materialized_keys_buffer.borrow_mut();
        let materialized_values_buffer = materialized_values.buffer();
        let mut materialized_values_buffer = materialized_values_buffer.borrow_mut();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        update_kernel.encode(
            Some(rotated_keys_buffer.deref()),
            qkv_buffer.deref(),
            materialized_keys_buffer.deref_mut(),
            materialized_values_buffer.deref_mut(),
            num_groups as u32,
            num_heads as u32,
            head_dim as u32,
            suffix_length as u32,
            0,
            total_length as u32,
            &mut encoder,
        );
        drop(materialized_values_buffer);
        drop(materialized_keys_buffer);
        drop(qkv_buffer);
        drop(rotated_keys_buffer);

        let indices = (0..suffix_length).collect::<Vec<_>>();
        layer.update_after_acceptance(
            &context,
            &indices,
            Some(0),
            None,
            Some(&materialized_values),
            &mut encoder,
            &kv_cache_update,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<half::bf16>();
        for token_index in 0..suffix_length {
            let expected_base = token_index as f32 * 8.0;
            let group0_base = token_index * head_dim;
            assert_eq!(recent_values[group0_base].to_f32(), half::bf16::from_f32(expected_base + 8.0).to_f32());
            assert_eq!(recent_values[group0_base + 1].to_f32(), half::bf16::from_f32(expected_base + 12.0).to_f32());
            let group1_base = hot_value_capacity * head_dim + group0_base;
            assert_eq!(recent_values[group1_base].to_f32(), half::bf16::from_f32(expected_base + 16.0).to_f32());
            assert_eq!(recent_values[group1_base + 1].to_f32(), half::bf16::from_f32(expected_base + 20.0).to_f32());
        }
    }

    #[test]
    fn sparse_value_recent_values_track_external_materialized_rows_on_metal_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let hot_value_capacity = 256;
        let prompt_length = 40;
        let generated_length = 64;
        let total_length = hot_value_capacity + prompt_length;
        let kv_cache_update =
            KVCacheUpdate::new(context.as_ref(), DataType::BF16, total_length).expect("kv cache update");
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let make_qkv = |token_start: usize, token_count: usize| {
            context.create_array_from(
                &[token_count, qkv_stride],
                &(0..token_count)
                    .flat_map(|token_offset| {
                        let token_index = token_start + token_offset;
                        let token_base = token_index as f32 * 8.0;
                        [
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(token_base + 8.0),
                            half::bf16::from_f32(token_base + 12.0),
                            half::bf16::from_f32(token_base + 16.0),
                            half::bf16::from_f32(token_base + 20.0),
                        ]
                    })
                    .collect::<Vec<_>>(),
                "qkv",
            )
        };
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, total_length, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "keys",
            ))),
            values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "values",
            ))),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length: prompt_length,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: hot_value_capacity,
                    keep_mass: 1.0,
                    page_size: 32,
                },
                num_groups,
                total_length,
                head_dim,
                prompt_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, prompt_length, head_dim],
                DataType::BF16,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::BF16,
                "recent_values",
            ))),
            triattention: None,
        };
        let rotated_keys =
            context.create_array_zeros(&[num_groups, prompt_length, head_dim], DataType::BF16, "rotated_keys");
        let mut materialized_values =
            context.create_array_zeros(&[num_groups, total_length, head_dim], DataType::BF16, "materialized_values");

        {
            let rows = materialized_values.as_slice_mut::<half::bf16>();
            for token_index in 0..prompt_length {
                let expected_base = token_index as f32 * 8.0;
                let group0_base = token_index * head_dim;
                rows[group0_base..group0_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 8.0),
                    half::bf16::from_f32(expected_base + 12.0),
                ]);
                let group1_base = total_length * head_dim + group0_base;
                rows[group1_base..group1_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 16.0),
                    half::bf16::from_f32(expected_base + 20.0),
                ]);
            }
        }

        let prompt_qkv = make_qkv(0, prompt_length);
        layer.stage_sparse_value_suffix_rows(
            &rotated_keys.view(&[num_groups, prompt_length, head_dim]),
            &prompt_qkv,
            num_heads,
        );
        let prompt_indices = (0..prompt_length).collect::<Vec<_>>();
        let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
        layer.update_after_acceptance(
            &context,
            &prompt_indices,
            Some(0),
            None,
            Some(&materialized_values),
            &mut encoder,
            &kv_cache_update,
        );
        encoder.end_encoding().submit().wait_until_completed().expect("complete");
        layer.register_accepted_tokens(prompt_length);

        for token_index in prompt_length..prompt_length + generated_length {
            let qkv = make_qkv(token_index, 1);
            layer.stage_sparse_value_suffix_rows(&rotated_keys.view(&[num_groups, 1, head_dim]), &qkv, num_heads);
            let expected_base = token_index as f32 * 8.0;
            let rows = materialized_values.as_slice_mut::<half::bf16>();
            let group0_base = token_index * head_dim;
            rows[group0_base..group0_base + head_dim].copy_from_slice(&[
                half::bf16::from_f32(expected_base + 8.0),
                half::bf16::from_f32(expected_base + 12.0),
            ]);
            let group1_base = total_length * head_dim + group0_base;
            rows[group1_base..group1_base + head_dim].copy_from_slice(&[
                half::bf16::from_f32(expected_base + 16.0),
                half::bf16::from_f32(expected_base + 20.0),
            ]);

            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.update_after_acceptance(
                &context,
                &[0],
                None,
                None,
                Some(&materialized_values),
                &mut encoder,
                &kv_cache_update,
            );
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
            layer.register_accepted_tokens(1);
        }

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<half::bf16>();
        for token_index in 0..prompt_length + generated_length {
            let expected_base = token_index as f32 * 8.0;
            let group0_base = token_index * head_dim;
            assert_eq!(recent_values[group0_base].to_f32(), expected_base + 8.0);
            assert_eq!(recent_values[group0_base + 1].to_f32(), expected_base + 12.0);
            let group1_base = hot_value_capacity * head_dim + group0_base;
            assert_eq!(recent_values[group1_base].to_f32(), expected_base + 16.0);
            assert_eq!(recent_values[group1_base + 1].to_f32(), expected_base + 20.0);
        }
    }

    #[test]
    fn sparse_value_recent_values_track_chunked_prefill_materialized_rows_on_metal_bf16() {
        let Some(context) = <Metal as Backend>::Context::new().ok() else {
            return;
        };
        let num_groups = 2;
        let num_heads = 4;
        let head_dim = 2;
        let hot_value_capacity = 512;
        let prompt_length = 154;
        let prefill_step = 128;
        let generated_length = 64;
        let total_length = hot_value_capacity + prompt_length;
        let kv_cache_update =
            KVCacheUpdate::new(context.as_ref(), DataType::BF16, total_length).expect("kv cache update");
        let qkv_stride = (num_heads + num_groups + num_groups) * head_dim;
        let make_qkv = |token_start: usize, token_count: usize| {
            context.create_array_from(
                &[token_count, qkv_stride],
                &(0..token_count)
                    .flat_map(|token_offset| {
                        let token_index = token_start + token_offset;
                        let token_base = token_index as f32 * 8.0;
                        [
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(0.0),
                            half::bf16::from_f32(token_base + 8.0),
                            half::bf16::from_f32(token_base + 12.0),
                            half::bf16::from_f32(token_base + 16.0),
                            half::bf16::from_f32(token_base + 20.0),
                        ]
                    })
                    .collect::<Vec<_>>(),
                "qkv",
            )
        };
        let mut layer = KVCacheLayer::<Metal> {
            state: KVCacheLayerState::Full {
                prefix_len: 0,
            },
            shape: [num_groups, total_length, head_dim],
            data_type: DataType::BF16,
            keys: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "keys",
            ))),
            values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, total_length, head_dim],
                DataType::BF16,
                "values",
            ))),
            prefix_token_positions: Vec::new(),
            next_token_position: 0,
            max_suffix_length: prompt_length,
            compression_mode: KvCompressionMode::SparseValue,
            compressor: None,
            sparse_value: Some(SparseValueState::new(
                SparseValueConfig {
                    recent_window: hot_value_capacity,
                    keep_mass: 1.0,
                    page_size: 32,
                },
                num_groups,
                total_length,
                head_dim,
                prompt_length,
            )),
            sparse_value_pending_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, prompt_length, head_dim],
                DataType::BF16,
                "pending_values",
            ))),
            sparse_value_recent_values: Some(RefCell::new(context.create_array_zeros(
                &[num_groups, hot_value_capacity, head_dim],
                DataType::BF16,
                "recent_values",
            ))),
            triattention: None,
        };
        let rotated_keys =
            context.create_array_zeros(&[num_groups, prompt_length, head_dim], DataType::BF16, "rotated_keys");
        let mut materialized_values =
            context.create_array_zeros(&[num_groups, total_length, head_dim], DataType::BF16, "materialized_values");

        {
            let rows = materialized_values.as_slice_mut::<half::bf16>();
            for token_index in 0..prompt_length {
                let expected_base = token_index as f32 * 8.0;
                let group0_base = token_index * head_dim;
                rows[group0_base..group0_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 8.0),
                    half::bf16::from_f32(expected_base + 12.0),
                ]);
                let group1_base = total_length * head_dim + group0_base;
                rows[group1_base..group1_base + head_dim].copy_from_slice(&[
                    half::bf16::from_f32(expected_base + 16.0),
                    half::bf16::from_f32(expected_base + 20.0),
                ]);
            }
        }

        for chunk_start in (0..prompt_length).step_by(prefill_step) {
            let chunk_len = (prompt_length - chunk_start).min(prefill_step);
            let qkv = make_qkv(chunk_start, chunk_len);
            layer.stage_sparse_value_suffix_rows(
                &rotated_keys.view(&[num_groups, chunk_len, head_dim]),
                &qkv,
                num_heads,
            );
            let indices = (0..chunk_len).collect::<Vec<_>>();
            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.update_after_acceptance(
                &context,
                &indices,
                Some(chunk_start),
                None,
                Some(&materialized_values),
                &mut encoder,
                &kv_cache_update,
            );
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
            layer.register_accepted_tokens(chunk_len);
        }

        for token_index in prompt_length..prompt_length + generated_length {
            let qkv = make_qkv(token_index, 1);
            layer.stage_sparse_value_suffix_rows(&rotated_keys.view(&[num_groups, 1, head_dim]), &qkv, num_heads);
            let expected_base = token_index as f32 * 8.0;
            let rows = materialized_values.as_slice_mut::<half::bf16>();
            let group0_base = token_index * head_dim;
            rows[group0_base..group0_base + head_dim].copy_from_slice(&[
                half::bf16::from_f32(expected_base + 8.0),
                half::bf16::from_f32(expected_base + 12.0),
            ]);
            let group1_base = total_length * head_dim + group0_base;
            rows[group1_base..group1_base + head_dim].copy_from_slice(&[
                half::bf16::from_f32(expected_base + 16.0),
                half::bf16::from_f32(expected_base + 20.0),
            ]);

            let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
            layer.update_after_acceptance(
                &context,
                &[0],
                None,
                None,
                Some(&materialized_values),
                &mut encoder,
                &kv_cache_update,
            );
            encoder.end_encoding().submit().wait_until_completed().expect("complete");
            layer.register_accepted_tokens(1);
        }

        let recent_values = layer.sparse_value_recent_values.as_ref().expect("recent values").borrow();
        let recent_values = recent_values.as_slice::<half::bf16>();
        for token_index in 0..prompt_length + generated_length {
            let expected_base = token_index as f32 * 8.0;
            let group0_base = token_index * head_dim;
            assert_eq!(
                recent_values[group0_base].to_f32(),
                half::bf16::from_f32(expected_base + 8.0).to_f32(),
                "token {token_index} group0 value0"
            );
            assert_eq!(
                recent_values[group0_base + 1].to_f32(),
                half::bf16::from_f32(expected_base + 12.0).to_f32(),
                "token {token_index} group0 value1"
            );
            let group1_base = hot_value_capacity * head_dim + group0_base;
            assert_eq!(
                recent_values[group1_base].to_f32(),
                half::bf16::from_f32(expected_base + 16.0).to_f32(),
                "token {token_index} group1 value0"
            );
            assert_eq!(
                recent_values[group1_base + 1].to_f32(),
                half::bf16::from_f32(expected_base + 20.0).to_f32(),
                "token {token_index} group1 value1"
            );
        }
    }
}
