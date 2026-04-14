use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    sync::Arc,
};

use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{Backend, Encoder, kernel::kv_cache_update::KVCacheUpdate},
    config::DecoderLayerType,
    forward_pass::{
        delta_net_layer::DeltaNetLayer,
        kv_cache_layer::{
            AttentionBiasUpdate, INVALID_POSITION, KVCacheLayer, KVCacheLayerState, KVSlice, KvCompressionMode,
            KvCompressor, SparseValueConfig, SparseValueState, TriAttentionConfig, TriAttentionState,
        },
        model_shape::ModelShape,
        short_conv_layer::ShortConvLayer,
        ssm_layer::SSMLayer,
    },
};

pub type KvCompressorFactory<B> =
    dyn Fn(&<B as Backend>::Context, usize, [usize; 3]) -> Box<dyn KvCompressor<B>> + Send + Sync;

#[derive(Clone)]
pub struct KvCompressionConfig<B: Backend> {
    pub mode: KvCompressionMode,
    pub compress_keys: bool,
    pub compress_values: bool,
    pub factory: Option<Arc<KvCompressorFactory<B>>>,
    pub sparse_value: Option<SparseValueConfig>,
    pub triattention: Option<TriAttentionConfig>,
}

impl<B: Backend> KvCompressionConfig<B> {
    pub fn disabled() -> Self {
        Self {
            mode: KvCompressionMode::None,
            compress_keys: false,
            compress_values: false,
            factory: None,
            sparse_value: None,
            triattention: None,
        }
    }
}

#[derive(Debug)]
pub enum CacheLayer<B: Backend> {
    Transformer(KVCacheLayer<B>),
    StateSpace(SSMLayer<B>),
    ShortConv(ShortConvLayer<B>),
    DeltaNet(DeltaNetLayer<B>),
}

#[derive(Clone)]
pub enum CacheLayerSlice<B: Backend> {
    Transformer(KVSlice<B>),
    StateSpace,
    ShortConv,
    DeltaNet,
}

const ARRAY_TRANSFORMER_KEYS_LABEL: &str = "cache_layers_transformer_keys";
const ARRAY_TRANSFORMER_VALUES_LABEL: &str = "cache_layers_transformer_values";
const ARRAY_STATE_SPACE_CONV_STATE_LABEL: &str = "cache_layers_state_space_conv_state";
const ARRAY_STATE_SPACE_SSM_STATE_LABEL: &str = "cache_layers_state_space_ssm_state";
const ARRAY_SHORT_CONV_CONV_STATE_LABEL: &str = "cache_layers_short_conv_conv_state";
const ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL: &str = "cache_layers_short_conv_suffix_state";
const ARRAY_DELTA_NET_CONV_STATE_LABEL: &str = "cache_layers_delta_net_conv_state";
const ARRAY_DELTA_NET_SSM_STATE_LABEL: &str = "cache_layers_delta_net_ssm_state";

impl<B: Backend> CacheLayer<B> {
    pub fn as_transformer(&self) -> Option<&KVCacheLayer<B>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_transformer_mut(&mut self) -> Option<&mut KVCacheLayer<B>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space(&self) -> Option<&SSMLayer<B>> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space_mut(&mut self) -> Option<&mut SSMLayer<B>> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv(&self) -> Option<&ShortConvLayer<B>> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv_mut(&mut self) -> Option<&mut ShortConvLayer<B>> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_delta_net(&self) -> Option<&DeltaNetLayer<B>> {
        match self {
            CacheLayer::DeltaNet(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_delta_net_mut(&mut self) -> Option<&mut DeltaNetLayer<B>> {
        match self {
            CacheLayer::DeltaNet(layer) => Some(layer),
            _ => None,
        }
    }
}

pub struct CacheLayers<B: Backend> {
    max_suffix_length: usize,
    max_prefix_length: usize,
    pub data: Box<[CacheLayer<B>]>,
}

#[derive(Clone)]
pub struct CacheLayersSlice<B: Backend> {
    pub layers: Vec<CacheLayerSlice<B>>,
}

impl<B: Backend> CacheLayers<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
        kv_compression: KvCompressionConfig<B>,
    ) -> Self {
        let total_context_length = max_prefix_length.max(max_suffix_length);
        let kv_shapes: Vec<[usize; 3]> =
            model_shape.kv_cache_layer_shapes(max_prefix_length, max_suffix_length).collect();
        let compression_mode = kv_compression.mode;
        let compress_keys = kv_compression.compress_keys;
        let compress_values = kv_compression.compress_values;
        let compression_factory = kv_compression.factory.clone();
        let sparse_value_config = kv_compression.sparse_value.clone();
        let triattention_config = kv_compression.triattention.clone();

        let data: Box<[CacheLayer<B>]> = model_shape
            .layer_types()
            .iter()
            .enumerate()
            .map(|(layer_index, layer_type)| match layer_type {
                DecoderLayerType::Transformer => {
                    let base_shape = kv_shapes[layer_index];
                    let window_length = model_shape.sliding_window_length_per_layer[layer_index]
                        .filter(|&window_size| window_size < total_context_length);
                    let use_dense_windowed_sparse_value = sparse_value_config.is_some() && window_length.is_some();
                    let layer_compression_mode = if use_dense_windowed_sparse_value {
                        KvCompressionMode::None
                    } else {
                        compression_mode
                    };
                    let layer_compress_keys = compress_keys && !use_dense_windowed_sparse_value;
                    let layer_compress_values = compress_values && !use_dense_windowed_sparse_value;
                    let triattention = (triattention_config.is_some() && window_length.is_none()).then(|| {
                        let config = triattention_config
                            .clone()
                            .expect("TriAttention config is required when TriAttention is enabled");
                        TriAttentionState {
                            config,
                            tokens_since_last_prune: 0,
                            calibration: crate::forward_pass::kv_cache_layer::TriAttentionCalibration::new(
                                model_shape.num_heads(),
                                model_shape.head_dim(),
                                model_shape.rope_dim(),
                            ),
                        }
                    });
                    let sparse_value = (sparse_value_config.is_some() && window_length.is_none()).then(|| {
                        SparseValueState::new(
                            sparse_value_config
                                .clone()
                                .expect("SparseValue config is required when SparseValue is enabled"),
                            base_shape[0],
                            base_shape[1],
                            base_shape[2],
                            max_suffix_length,
                        )
                    });
                    let sparse_value_recent_values = sparse_value.as_ref().map(|state| {
                        RefCell::new(context.create_array(
                            &[base_shape[0], state.hot_value_capacity, base_shape[2]],
                            model_shape.kv_cache_data_type(),
                            &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}_sparse_recent"),
                        ))
                    });
                    let sparse_value_pending_values = sparse_value.as_ref().map(|_| {
                        RefCell::new(context.create_array(
                            &[base_shape[0], max_suffix_length, base_shape[2]],
                            model_shape.kv_cache_data_type(),
                            &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}_sparse_pending"),
                        ))
                    });
                    let shape = if let Some(triattention) = &triattention {
                        [base_shape[0], triattention.config.budget + max_suffix_length, base_shape[2]]
                    } else {
                        base_shape
                    };

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

                    CacheLayer::Transformer(KVCacheLayer {
                        state: state.clone(),
                        shape,
                        data_type: model_shape.kv_cache_data_type(),
                        keys: (!layer_compress_keys).then(|| {
                            RefCell::new(context.create_array(
                                &shape,
                                model_shape.kv_cache_data_type(),
                                &format!("{ARRAY_TRANSFORMER_KEYS_LABEL}_{layer_index}"),
                            ))
                        }),
                        values: (!layer_compress_values).then(|| {
                            RefCell::new(context.create_array(
                                &shape,
                                model_shape.kv_cache_data_type(),
                                &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}"),
                            ))
                        }),
                        prefix_token_positions: match &state {
                            KVCacheLayerState::Full {
                                ..
                            } => Vec::with_capacity(max_prefix_length),
                            KVCacheLayerState::Windowed {
                                window_length,
                                ..
                            } => (0..*window_length).map(|_| INVALID_POSITION).collect(),
                        },
                        next_token_position: 0,
                        max_suffix_length,
                        compression_mode: layer_compression_mode,
                        compressor: (!use_dense_windowed_sparse_value)
                            .then(|| compression_factory.as_ref().map(|factory| factory(context, layer_index, shape)))
                            .flatten(),
                        sparse_value,
                        sparse_value_pending_values,
                        sparse_value_recent_values,
                        triattention,
                    })
                },
                DecoderLayerType::StateSpace {
                    conv_dim,
                    kernel_size,
                    state_dim,
                    num_heads,
                    head_dim,
                    ..
                } => {
                    let conv_shape = [*conv_dim, kernel_size.saturating_sub(1)];
                    let ssm_shape = [*num_heads, *head_dim, *state_dim];
                    let dtype = model_shape.activation_data_type();

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: context.create_array_zeros(
                            &conv_shape,
                            dtype,
                            &format!("{ARRAY_STATE_SPACE_CONV_STATE_LABEL}_{layer_index}"),
                        ),
                        ssm_state: context.create_array_zeros(
                            &ssm_shape,
                            dtype,
                            &format!("{ARRAY_STATE_SPACE_SSM_STATE_LABEL}_{layer_index}"),
                        ),
                    })
                },
                DecoderLayerType::ShortConv {
                    kernel_size,
                } => {
                    let conv_shape = [model_shape.model_dim(), kernel_size.saturating_sub(1)];
                    let suffix_state_shape =
                        [max_suffix_length, model_shape.model_dim(), kernel_size.saturating_sub(1)];
                    let dtype = model_shape.activation_data_type();

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: context.create_array_zeros(
                            &conv_shape,
                            dtype,
                            &format!("{ARRAY_SHORT_CONV_CONV_STATE_LABEL}_{layer_index}"),
                        ),
                        suffix_state: context.create_array_zeros(
                            &suffix_state_shape,
                            dtype,
                            &format!("{ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL}_{layer_index}"),
                        ),
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
                    })
                },
                DecoderLayerType::DeltaNet {
                    conv_dim,
                    kernel_size,
                    num_heads,
                    head_dim,
                    value_head_dim,
                    ..
                } => {
                    let conv_shape = [*conv_dim, kernel_size.saturating_sub(1)];
                    let ssm_shape = [*num_heads, *value_head_dim, *head_dim];
                    let dtype = model_shape.activation_data_type();

                    CacheLayer::DeltaNet(DeltaNetLayer {
                        conv_state: context.create_array_zeros(
                            &conv_shape,
                            dtype,
                            &format!("{ARRAY_DELTA_NET_CONV_STATE_LABEL}_{layer_index}"),
                        ),
                        ssm_state: context.create_array_zeros(
                            &ssm_shape,
                            dtype,
                            &format!("{ARRAY_DELTA_NET_SSM_STATE_LABEL}_{layer_index}"),
                        ),
                    })
                },
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
            match layer {
                CacheLayer::Transformer(layer) => {
                    match &mut layer.state {
                        KVCacheLayerState::Full {
                            prefix_len,
                        } => {
                            *prefix_len = 0;
                            layer.prefix_token_positions.clear();
                        },
                        KVCacheLayerState::Windowed {
                            ring_offset,
                            ring_length,
                            ..
                        } => {
                            *ring_offset = 0;
                            *ring_length = 0;
                            layer.prefix_token_positions.fill(INVALID_POSITION);
                        },
                    }
                    layer.next_token_position = 0;
                },
                CacheLayer::StateSpace(layer) => layer.zero(),
                CacheLayer::ShortConv(layer) => layer.zero(),
                CacheLayer::DeltaNet(layer) => layer.zero(),
            }
            if let Some(layer) = layer.as_transformer_mut() {
                if let Some(triattention) = &mut layer.triattention {
                    triattention.tokens_since_last_prune = 0;
                    triattention.calibration.query_token_count = 0;
                    triattention.calibration.query_sum.fill(0.0);
                    triattention.calibration.query_norm_sum.fill(0.0);
                }
            }
        }
    }

    pub fn uses_compressed_transformer_layers(&self) -> bool {
        self.data.iter().filter_map(|layer| layer.as_transformer()).any(|layer| layer.uses_compressed_storage())
    }

    pub fn uses_materialized_transformer_state(&self) -> bool {
        self.data
            .iter()
            .filter_map(|layer| layer.as_transformer())
            .any(|layer| layer.uses_materialized_transformer_state())
    }

    pub fn blocks_pre_encode_for_single_decode(&self) -> bool {
        self.data
            .iter()
            .filter_map(|layer| layer.as_transformer())
            .any(|layer| !layer.allows_pre_encode_for_single_decode())
    }

    pub fn requires_synchronous_acceptance_update(&self) -> bool {
        self.data
            .iter()
            .filter_map(|layer| layer.as_transformer())
            .any(|layer| layer.requires_synchronous_acceptance_update())
    }

    pub fn kv_storage_bytes(&self) -> usize {
        self.data.iter().filter_map(|layer| layer.as_transformer()).map(|layer| layer.storage_bytes()).sum()
    }

    pub fn max_suffix_length(&self) -> usize {
        self.max_suffix_length
    }

    pub fn max_prefix_length(&self) -> usize {
        self.max_prefix_length
    }

    pub fn fill_attention_bias(
        &self,
        dst: &mut HashMap<Option<usize>, Array<B>>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        for layer in self.data.iter() {
            if let CacheLayer::Transformer(layer) = layer {
                if let Some(array) = dst.get_mut(&layer.window_length()) {
                    layer.fill_attention_bias(array, suffix_token_positions, suffix_length, external_bias_fn);
                }
            }
        }
    }

    pub fn fill_attention_bias_scratch(
        &self,
        dst: &HashMap<Option<usize>, RefCell<Array<B>>>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        _context: &B::Context,
    ) {
        for layer in self.data.iter() {
            if let CacheLayer::Transformer(layer) = layer {
                if let Some(cell) = dst.get(&layer.window_length()) {
                    layer.fill_attention_bias(&mut cell.borrow_mut(), suffix_token_positions, suffix_length, None);
                }
            }
        }
    }

    pub fn update_after_acceptance(
        &mut self,
        context: &B::Context,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        let short_conv_commit_index = accepted_suffix_indices.last().copied().unwrap_or(0);
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(
                    context,
                    accepted_suffix_indices,
                    suffix_start,
                    None,
                    None,
                    encoder,
                    kv_cache_update,
                );
            } else if let Some(layer) = layer.as_short_conv_mut() {
                layer.commit_from_suffix_state_if_valid(short_conv_commit_index);
            }
        }
    }

    pub fn register_accepted_tokens(
        &mut self,
        number_of_tokens: usize,
    ) {
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.register_accepted_tokens(number_of_tokens);
            }
        }
    }

    pub fn reset_triattention_prune_counters(&mut self) {
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                if let Some(triattention) = &mut layer.triattention {
                    triattention.tokens_since_last_prune = 0;
                }
            }
        }
    }

    pub fn prune_triattention_if_needed(
        &mut self,
        rope_cosines: &Array<B>,
        rope_sines: &Array<B>,
    ) {
        for layer in self.data.iter_mut() {
            if let CacheLayer::Transformer(layer) = layer {
                layer.prune_triattention_if_needed(rope_cosines, rope_sines);
            }
        }
    }

    pub fn attention_bias_updates_after_acceptance(
        &self,
        accepted_len: usize,
    ) -> Vec<AttentionBiasUpdate> {
        self.data
            .iter()
            .filter_map(|layer| match layer {
                CacheLayer::Transformer(kv) => kv.attention_bias_update_after_acceptance(accepted_len),
                _ => None,
            })
            .collect()
    }

    pub fn slice(
        &self,
        context: &B::Context,
        range: std::ops::Range<usize>,
    ) -> Option<CacheLayersSlice<B>> {
        let mut layers = Vec::with_capacity(self.data.len());
        for layer in self.data.iter() {
            match layer {
                CacheLayer::Transformer(kv) => {
                    let Some(slice) = kv.slice(context, range.clone()) else {
                        return None;
                    };
                    layers.push(CacheLayerSlice::Transformer(slice));
                },
                CacheLayer::StateSpace(_) => layers.push(CacheLayerSlice::StateSpace),
                CacheLayer::ShortConv(_) => layers.push(CacheLayerSlice::ShortConv),
                CacheLayer::DeltaNet(_) => layers.push(CacheLayerSlice::DeltaNet),
            }
        }

        Some(CacheLayersSlice {
            layers,
        })
    }

    pub fn apply_slice(
        &mut self,
        context: &B::Context,
        slice: &CacheLayersSlice<B>,
        range: Option<std::ops::Range<usize>>,
    ) {
        for (layer, snapshot) in self.data.iter_mut().zip(slice.layers.iter()) {
            match (layer, snapshot) {
                (CacheLayer::Transformer(kv), CacheLayerSlice::Transformer(s)) => {
                    kv.apply_slice(context, &s, range.clone());
                },
                (CacheLayer::StateSpace(_), CacheLayerSlice::StateSpace) => {},
                (CacheLayer::ShortConv(_), CacheLayerSlice::ShortConv) => {},
                _ => {},
            }
        }
    }

    pub fn clone(
        &self,
        context: &B::Context,
    ) -> Self {
        let mut max_prefix_capacity_across_layers = 0usize;
        let data: Box<[CacheLayer<B>]> = self
            .data
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| match layer {
                CacheLayer::Transformer(layer) => {
                    let num_groups = layer.shape[0];
                    let head_dim = layer.shape[2];
                    let dtype = layer.data_type;
                    let copy_rows = layer.prefix_segment_length();

                    let new_total_len = layer.shape[1];
                    if copy_rows > max_prefix_capacity_across_layers {
                        max_prefix_capacity_across_layers = copy_rows;
                    }

                    let new_shape = [num_groups, new_total_len, head_dim];
                    let mut new_keys = context.create_array(
                        &new_shape,
                        dtype,
                        &format!("{ARRAY_TRANSFORMER_KEYS_LABEL}_{layer_index}"),
                    );
                    let mut new_values = context.create_array(
                        &new_shape,
                        dtype,
                        &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}"),
                    );

                    layer.materialize_into(&mut new_keys, &mut new_values);

                    CacheLayer::Transformer(KVCacheLayer {
                        state: layer.state.clone(),
                        shape: [num_groups, new_total_len, head_dim],
                        data_type: dtype,
                        keys: Some(RefCell::new(new_keys)),
                        values: Some(RefCell::new(new_values)),
                        prefix_token_positions: layer.prefix_token_positions.clone(),
                        next_token_position: layer.next_token_position,
                        max_suffix_length: layer.max_suffix_length,
                        compression_mode: if layer.sparse_value.is_some() {
                            KvCompressionMode::SparseValue
                        } else {
                            KvCompressionMode::None
                        },
                        compressor: None,
                        sparse_value: layer.sparse_value.clone(),
                        sparse_value_pending_values: layer.sparse_value_pending_values.as_ref().map(|pending_values| {
                            let pending_values = pending_values.borrow();
                            let mut new_pending_values = context.create_array(
                                pending_values.shape(),
                                pending_values.data_type(),
                                &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}_sparse_pending"),
                            );
                            new_pending_values.copy_from_array(&pending_values);
                            RefCell::new(new_pending_values)
                        }),
                        sparse_value_recent_values: layer.sparse_value_recent_values.as_ref().map(|recent_values| {
                            let recent_values = recent_values.borrow();
                            let mut new_recent_values = context.create_array(
                                recent_values.shape(),
                                recent_values.data_type(),
                                &format!("{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}_sparse_recent"),
                            );
                            new_recent_values.copy_from_array(&recent_values);
                            RefCell::new(new_recent_values)
                        }),
                        triattention: None,
                    })
                },
                CacheLayer::StateSpace(layer) => {
                    let conv_shape = layer.conv_state.shape().to_vec();
                    let conv_dtype = layer.conv_state.data_type();
                    let mut new_conv = context.create_array(
                        &conv_shape,
                        conv_dtype,
                        &format!("{ARRAY_STATE_SPACE_CONV_STATE_LABEL}_{layer_index}"),
                    );
                    new_conv.copy_from_array(&layer.conv_state);

                    let ssm_shape = layer.ssm_state.shape().to_vec();
                    let ssm_dtype = layer.ssm_state.data_type();
                    let mut new_ssm = context.create_array(
                        &ssm_shape,
                        ssm_dtype,
                        &format!("{ARRAY_STATE_SPACE_SSM_STATE_LABEL}_{layer_index}"),
                    );
                    new_ssm.copy_from_array(&layer.ssm_state);

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: new_conv,
                        ssm_state: new_ssm,
                    })
                },
                CacheLayer::ShortConv(layer) => {
                    let conv_shape = layer.conv_state.shape().to_vec();
                    let conv_dtype = layer.conv_state.data_type();
                    let mut new_conv = context.create_array(
                        &conv_shape,
                        conv_dtype,
                        &format!("{ARRAY_SHORT_CONV_CONV_STATE_LABEL}_{layer_index}"),
                    );
                    new_conv.copy_from_array(&layer.conv_state);

                    let suffix_shape = layer.suffix_state.shape().to_vec();
                    let suffix_dtype = layer.suffix_state.data_type();
                    let mut new_suffix = context.create_array(
                        &suffix_shape,
                        suffix_dtype,
                        &format!("{ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL}_{layer_index}"),
                    );
                    new_suffix.as_bytes_mut().fill(0);

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: new_conv,
                        suffix_state: new_suffix,
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
                    })
                },
                CacheLayer::DeltaNet(layer) => {
                    let conv_shape = layer.conv_state.shape().to_vec();
                    let conv_dtype = layer.conv_state.data_type();
                    let mut new_conv = context.create_array(
                        &conv_shape,
                        conv_dtype,
                        &format!("{ARRAY_DELTA_NET_CONV_STATE_LABEL}_{layer_index}"),
                    );
                    new_conv.copy_from_array(&layer.conv_state);

                    let ssm_shape = layer.ssm_state.shape().to_vec();
                    let ssm_dtype = layer.ssm_state.data_type();
                    let mut new_ssm = context.create_array(
                        &ssm_shape,
                        ssm_dtype,
                        &format!("{ARRAY_DELTA_NET_SSM_STATE_LABEL}_{layer_index}"),
                    );
                    new_ssm.copy_from_array(&layer.ssm_state);

                    CacheLayer::DeltaNet(DeltaNetLayer {
                        conv_state: new_conv,
                        ssm_state: new_ssm,
                    })
                },
            })
            .collect();

        Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: max_prefix_capacity_across_layers,
            data,
        }
    }
}
