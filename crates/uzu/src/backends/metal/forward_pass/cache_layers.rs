use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

use super::{
    super::{MTLContext, MetalArray},
    kv_cache_layer::{
        AttentionBiasUpdate, INVALID_POSITION, KVCacheLayer, KVCacheLayerState,
        KVSlice,
    },
    short_conv_layer::ShortConvLayer,
    ssm_layer::SSMLayer,
};
use crate::{
    array::ArrayContextExt,
    backends::metal::{
        MTLCommandBuffer, ProtocolObject, Retained, kernel::KVCacheUpdate,
    },
    config::DecoderLayerType,
    forward_pass::model_shape::ModelShape,
};

#[derive(Debug)]
pub enum CacheLayer {
    Transformer(KVCacheLayer),
    StateSpace(SSMLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Clone)]
pub enum CacheLayerSlice {
    Transformer(KVSlice),
    StateSpace,
    ShortConv,
}

const ARRAY_TRANSFORMER_KEYS_LABEL: &str = "cache_layers_transformer_keys";
const ARRAY_TRANSFORMER_VALUES_LABEL: &str = "cache_layers_transformer_values";
const ARRAY_STATE_SPACE_CONV_STATE_LABEL: &str =
    "cache_layers_state_space_conv_state";
const ARRAY_STATE_SPACE_SSM_STATE_LABEL: &str =
    "cache_layers_state_space_ssm_state";
const ARRAY_SHORT_CONV_CONV_STATE_LABEL: &str =
    "cache_layers_short_conv_conv_state";
const ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL: &str =
    "cache_layers_short_conv_suffix_state";

impl CacheLayer {
    pub fn as_transformer(&self) -> Option<&KVCacheLayer> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_transformer_mut(&mut self) -> Option<&mut KVCacheLayer> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space(&self) -> Option<&SSMLayer> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space_mut(&mut self) -> Option<&mut SSMLayer> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv(&self) -> Option<&ShortConvLayer> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv_mut(&mut self) -> Option<&mut ShortConvLayer> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }
}

pub struct CacheLayers {
    max_suffix_length: usize,
    max_prefix_length: usize,
    pub data: Box<[CacheLayer]>,
}

#[derive(Clone)]
pub struct CacheLayersSlice {
    pub layers: Vec<CacheLayerSlice>,
}

impl CacheLayers {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Self {
        let total_context_length = max_prefix_length.max(max_suffix_length);
        let kv_shapes: Vec<[usize; 3]> = model_shape
            .kv_cache_layer_shapes(max_prefix_length, max_suffix_length)
            .collect();

        let data: Box<[CacheLayer]> = model_shape
            .layer_types()
            .iter()
            .enumerate()
            .map(|(layer_index, layer_type)| match layer_type {
                DecoderLayerType::Transformer => {
                    let shape = kv_shapes[layer_index];
                    let window_length = model_shape
                        .sliding_window_length_per_layer[layer_index]
                        .filter(|&window_size| {
                            window_size < total_context_length
                        });

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
                        keys: RefCell::new(context.create_array(
                            &shape,
                            model_shape.kv_cache_data_type(),
                            &format!(
                                "{ARRAY_TRANSFORMER_KEYS_LABEL}_{layer_index}"
                            ),
                        )),
                        values: RefCell::new(context.create_array(
                            &shape,
                            model_shape.kv_cache_data_type(),
                            &format!(
                                "{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}"
                            ),
                        )),
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
                        max_suffix_length,
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
                        conv_state: RefCell::new(context.create_array(
                            &conv_shape,
                            dtype,
                            &format!(
                                "{ARRAY_STATE_SPACE_CONV_STATE_LABEL}_{layer_index}"
                            ),
                        )),
                        ssm_state: RefCell::new(context.create_array(
                            &ssm_shape,
                            dtype,
                            &format!(
                                "{ARRAY_STATE_SPACE_SSM_STATE_LABEL}_{layer_index}"
                            ),
                        )),
                    })
                },
                DecoderLayerType::ShortConv {
                    kernel_size,
                } => {
                    let conv_shape = [
                        model_shape.model_dim(),
                        kernel_size.saturating_sub(1),
                    ];
                    let suffix_state_shape = [
                        max_suffix_length,
                        model_shape.model_dim(),
                        kernel_size.saturating_sub(1),
                    ];
                    let dtype = model_shape.activation_data_type();

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: RefCell::new(context.create_array(
                            &conv_shape,
                            dtype,
                            &format!(
                                "{ARRAY_SHORT_CONV_CONV_STATE_LABEL}_{layer_index}"
                            ),
                        )),
                        suffix_state: RefCell::new(context.create_array(
                            &suffix_state_shape,
                            dtype,
                            &format!(
                                "{ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL}_{layer_index}"
                            ),
                        )),
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
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
                CacheLayer::Transformer(layer) => match &mut layer.state {
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
                },
                CacheLayer::StateSpace(layer) => layer.zero(),
                CacheLayer::ShortConv(layer) => layer.zero(),
            }
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
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        for layer in self.data.iter() {
            if let CacheLayer::Transformer(layer) = layer {
                if let Some(array) = dst.get_mut(&layer.window_length()) {
                    layer.fill_attention_bias(
                        array,
                        suffix_token_positions,
                        suffix_length,
                        external_bias_fn,
                    );
                }
            }
        }
    }

    pub fn fill_attention_bias_scratch(
        &self,
        dst: &HashMap<Option<usize>, RefCell<MetalArray>>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        _context: &MTLContext,
    ) {
        for layer in self.data.iter() {
            if let CacheLayer::Transformer(layer) = layer {
                if let Some(cell) = dst.get(&layer.window_length()) {
                    layer.fill_attention_bias(
                        &mut cell.borrow_mut(),
                        suffix_token_positions,
                        suffix_length,
                        None,
                    );
                }
            }
        }
    }

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        kv_cache_update: &KVCacheUpdate,
    ) {
        let short_conv_commit_index =
            accepted_suffix_indices.last().copied().unwrap_or(0);
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(
                    accepted_suffix_indices,
                    suffix_start,
                    command_buffer,
                    kv_cache_update,
                );
            } else if let Some(layer) = layer.as_short_conv() {
                layer
                    .commit_from_suffix_state_if_valid(short_conv_commit_index);
            }
        }
    }

    pub fn register_accepted_tokens(
        &mut self,
        token_positions: &[usize],
    ) {
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.register_accepted_tokens(token_positions);
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
                CacheLayer::Transformer(kv) => {
                    kv.attention_bias_update_after_acceptance(accepted_len)
                },
                _ => None,
            })
            .collect()
    }

    pub fn slice(
        &self,
        context: &MTLContext,
        range: std::ops::Range<usize>,
    ) -> Option<CacheLayersSlice> {
        let mut layers = Vec::with_capacity(self.data.len());
        for layer in self.data.iter() {
            match layer {
                CacheLayer::Transformer(kv) => {
                    let Some(slice) = kv.slice(context, range.clone()) else {
                        return None;
                    };
                    layers.push(CacheLayerSlice::Transformer(slice));
                },
                CacheLayer::StateSpace(_) => {
                    layers.push(CacheLayerSlice::StateSpace)
                },
                CacheLayer::ShortConv(_) => {
                    layers.push(CacheLayerSlice::ShortConv)
                },
            }
        }

        Some(CacheLayersSlice {
            layers,
        })
    }

    pub fn apply_slice(
        &mut self,
        slice: &CacheLayersSlice,
        range: Option<std::ops::Range<usize>>,
    ) {
        for (layer, snapshot) in self.data.iter_mut().zip(slice.layers.iter()) {
            match (layer, snapshot) {
                (
                    CacheLayer::Transformer(kv),
                    CacheLayerSlice::Transformer(s),
                ) => {
                    kv.apply_slice(s, range.clone());
                },
                (CacheLayer::StateSpace(_), CacheLayerSlice::StateSpace) => {},
                (CacheLayer::ShortConv(_), CacheLayerSlice::ShortConv) => {},
                _ => {},
            }
        }
    }

    pub fn clone(
        &self,
        context: &MTLContext,
    ) -> Self {
        let mut max_prefix_capacity_across_layers = 0usize;
        let data: Box<[CacheLayer]> = self
            .data
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| match layer {
                CacheLayer::Transformer(layer) => {
                    let shape = layer.keys.borrow().shape().to_vec();
                    let num_groups = shape[0];
                    let head_dim = shape[2];
                    let dtype = layer.keys.borrow().data_type();
                    let copy_rows = layer.prefix_segment_length();

                    let new_total_len = copy_rows + self.max_suffix_length;
                    if copy_rows > max_prefix_capacity_across_layers {
                        max_prefix_capacity_across_layers = copy_rows;
                    }

                    let new_shape = [num_groups, new_total_len, head_dim];
                    let mut new_keys = context.create_array(
                        &new_shape,
                        dtype,
                        &format!(
                            "{ARRAY_TRANSFORMER_KEYS_LABEL}_{layer_index}"
                        ),
                    );
                    let mut new_values = context.create_array(
                        &new_shape,
                        dtype,
                        &format!(
                            "{ARRAY_TRANSFORMER_VALUES_LABEL}_{layer_index}"
                        ),
                    );

                    if copy_rows > 0 {
                        {
                            let keys = layer.keys.borrow();
                            new_keys.copy_slice(&keys, 1, 0..copy_rows, 0);
                        }
                        {
                            let values = layer.values.borrow();
                            new_values.copy_slice(&values, 1, 0..copy_rows, 0);
                        }
                    }

                    CacheLayer::Transformer(KVCacheLayer {
                        state: layer.state.clone(),
                        keys: RefCell::new(new_keys),
                        values: RefCell::new(new_values),
                        prefix_token_positions: layer
                            .prefix_token_positions
                            .clone(),
                        max_suffix_length: layer.max_suffix_length,
                    })
                },
                CacheLayer::StateSpace(layer) => {
                    let conv_shape = layer.conv_state.borrow().shape().to_vec();
                    let conv_dtype = layer.conv_state.borrow().data_type();
                    let mut new_conv = context.create_array(
                        &conv_shape,
                        conv_dtype,
                        &format!(
                            "{ARRAY_STATE_SPACE_CONV_STATE_LABEL}_{layer_index}"
                        ),
                    );
                    {
                        let conv_src = layer.conv_state.borrow();
                        new_conv.copy_from_array(&conv_src);
                    }

                    let ssm_shape = layer.ssm_state.borrow().shape().to_vec();
                    let ssm_dtype = layer.ssm_state.borrow().data_type();
                    let mut new_ssm = context.create_array(
                        &ssm_shape,
                        ssm_dtype,
                        &format!(
                            "{ARRAY_STATE_SPACE_SSM_STATE_LABEL}_{layer_index}"
                        ),
                    );
                    {
                        let ssm_src = layer.ssm_state.borrow();
                        new_ssm.copy_from_array(&ssm_src);
                    }

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: RefCell::new(new_conv),
                        ssm_state: RefCell::new(new_ssm),
                    })
                },
                CacheLayer::ShortConv(layer) => {
                    let conv_shape = layer.conv_state.borrow().shape().to_vec();
                    let conv_dtype = layer.conv_state.borrow().data_type();
                    let mut new_conv = context.create_array(
                        &conv_shape,
                        conv_dtype,
                        &format!(
                            "{ARRAY_SHORT_CONV_CONV_STATE_LABEL}_{layer_index}"
                        ),
                    );
                    {
                        let conv_src = layer.conv_state.borrow();
                        new_conv.copy_from_array(&conv_src);
                    }

                    let suffix_shape =
                        layer.suffix_state.borrow().shape().to_vec();
                    let suffix_dtype = layer.suffix_state.borrow().data_type();
                    let mut new_suffix = context.create_array(
                        &suffix_shape,
                        suffix_dtype,
                        &format!(
                            "{ARRAY_SHORT_CONV_SUFFIX_STATE_LABEL}_{layer_index}"
                        ),
                    );
                    {
                        new_suffix.as_bytes_mut().fill(0);
                    }

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: RefCell::new(new_conv),
                        suffix_state: RefCell::new(new_suffix),
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
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
