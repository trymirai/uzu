use std::cell::Cell;

use crate::{
    backends::common::{Backend, Encoder, allocation_helpers, kernel::kv_cache_update::KVCacheUpdate},
    config::DecoderLayerType,
    forward_pass::{
        delta_net_layer::DeltaNetLayer,
        kv_cache_layer::{KVCacheLayer, KVCacheLayerState, KVSlice},
        model_shape::ModelShape,
        short_conv_layer::ShortConvLayer,
        ssm_layer::SSMLayer,
    },
};

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
    ) -> Self {
        let total_context_length = max_prefix_length.max(max_suffix_length);
        let kv_shapes: Vec<[usize; 3]> =
            model_shape.kv_cache_layer_shapes(max_prefix_length, max_suffix_length).collect();

        let data: Box<[CacheLayer<B>]> = model_shape
            .layer_types()
            .iter()
            .enumerate()
            .map(|(layer_index, layer_type)| match layer_type {
                DecoderLayerType::Transformer => {
                    let shape = kv_shapes[layer_index];
                    let window_length = model_shape.sliding_window_length_per_layer[layer_index]
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

                    CacheLayer::Transformer(KVCacheLayer {
                        state: state.clone(),
                        keys: allocation_helpers::create_zeroed_allocation(
                            context,
                            &shape,
                            model_shape.kv_cache_data_type(),
                        ),
                        values: allocation_helpers::create_zeroed_allocation(
                            context,
                            &shape,
                            model_shape.kv_cache_data_type(),
                        ),
                        shape,
                        data_type: model_shape.kv_cache_data_type(),
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
                        conv_state: allocation_helpers::create_zeroed_allocation(context, &conv_shape, dtype),
                        conv_shape,
                        ssm_state: allocation_helpers::create_zeroed_allocation(context, &ssm_shape, dtype),
                        ssm_shape,
                        data_type: dtype,
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
                        conv_state: allocation_helpers::create_zeroed_allocation(context, &conv_shape, dtype),
                        conv_shape,
                        suffix_state: allocation_helpers::create_zeroed_allocation(context, &suffix_state_shape, dtype),
                        suffix_shape: suffix_state_shape,
                        data_type: dtype,
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
                        conv_state: allocation_helpers::create_zeroed_allocation(context, &conv_shape, dtype),
                        conv_shape,
                        ssm_state: allocation_helpers::create_zeroed_allocation(context, &ssm_shape, dtype),
                        ssm_shape,
                        data_type: dtype,
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
                        allocation_helpers::fill_allocation(&mut layer.keys, 0);
                        allocation_helpers::fill_allocation(&mut layer.values, 0);
                        *prefix_len = 0;
                    },
                    KVCacheLayerState::Windowed {
                        ring_offset,
                        ring_length,
                        ..
                    } => {
                        allocation_helpers::fill_allocation(&mut layer.keys, 0);
                        allocation_helpers::fill_allocation(&mut layer.values, 0);
                        *ring_offset = 0;
                        *ring_length = 0;
                    },
                },
                CacheLayer::StateSpace(layer) => layer.zero(),
                CacheLayer::ShortConv(layer) => layer.zero(),
                CacheLayer::DeltaNet(layer) => layer.zero(),
            }
        }
    }

    pub fn max_suffix_length(&self) -> usize {
        self.max_suffix_length
    }

    pub fn max_prefix_length(&self) -> usize {
        self.max_prefix_length
    }

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        let short_conv_commit_index = accepted_suffix_indices.last().copied().unwrap_or(0);
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(accepted_suffix_indices, suffix_start, encoder, kv_cache_update);
            } else if let Some(layer) = layer.as_short_conv_mut() {
                layer.commit_from_suffix_state_if_valid(short_conv_commit_index);
            }
        }
    }

    pub fn register_accepted_tokens(
        &mut self,
        number_of_accepted_tokens: usize,
    ) {
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.register_accepted_tokens(number_of_accepted_tokens);
            }
        }
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
        slice: &CacheLayersSlice<B>,
        range: Option<std::ops::Range<usize>>,
    ) {
        for (layer, snapshot) in self.data.iter_mut().zip(slice.layers.iter()) {
            match (layer, snapshot) {
                (CacheLayer::Transformer(kv), CacheLayerSlice::Transformer(s)) => {
                    kv.apply_slice(&s, range.clone());
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
            .map(|(_layer_index, layer)| match layer {
                CacheLayer::Transformer(layer) => {
                    let shape = layer.shape;
                    let [num_groups, _, head_dim] = shape;
                    let dtype = layer.data_type;
                    let copy_rows = layer.prefix_segment_length();

                    let new_total_len = copy_rows + self.max_suffix_length;
                    if copy_rows > max_prefix_capacity_across_layers {
                        max_prefix_capacity_across_layers = copy_rows;
                    }

                    let new_shape = [num_groups, new_total_len, head_dim];
                    let mut new_keys = allocation_helpers::create_zeroed_allocation(context, &new_shape, dtype);
                    let mut new_values = allocation_helpers::create_zeroed_allocation(context, &new_shape, dtype);

                    if copy_rows > 0 {
                        let slice = layer.slice(context, 0..copy_rows).expect("Failed to slice KV cache layer");
                        let mut new_layer = KVCacheLayer {
                            state: layer.state.clone(),
                            keys: new_keys,
                            values: new_values,
                            shape: new_shape,
                            data_type: dtype,
                        };
                        new_layer.apply_slice(&slice, None);
                        new_keys = new_layer.keys;
                        new_values = new_layer.values;
                    }

                    CacheLayer::Transformer(KVCacheLayer {
                        state: layer.state.clone(),
                        keys: new_keys,
                        values: new_values,
                        shape: new_shape,
                        data_type: dtype,
                    })
                },
                CacheLayer::StateSpace(layer) => {
                    let mut new_conv =
                        allocation_helpers::create_allocation(context, &layer.conv_shape, layer.data_type);
                    allocation_helpers::copy_allocation_to_allocation(&mut new_conv, &layer.conv_state);
                    let mut new_ssm = allocation_helpers::create_allocation(context, &layer.ssm_shape, layer.data_type);
                    allocation_helpers::copy_allocation_to_allocation(&mut new_ssm, &layer.ssm_state);

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: new_conv,
                        conv_shape: layer.conv_shape,
                        ssm_state: new_ssm,
                        ssm_shape: layer.ssm_shape,
                        data_type: layer.data_type,
                    })
                },
                CacheLayer::ShortConv(layer) => {
                    let mut new_conv =
                        allocation_helpers::create_allocation(context, &layer.conv_shape, layer.data_type);
                    allocation_helpers::copy_allocation_to_allocation(&mut new_conv, &layer.conv_state);
                    let new_suffix =
                        allocation_helpers::create_zeroed_allocation(context, &layer.suffix_shape, layer.data_type);

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: new_conv,
                        conv_shape: layer.conv_shape,
                        suffix_state: new_suffix,
                        suffix_shape: layer.suffix_shape,
                        data_type: layer.data_type,
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
                    })
                },
                CacheLayer::DeltaNet(layer) => {
                    let mut new_conv =
                        allocation_helpers::create_allocation(context, &layer.conv_shape, layer.data_type);
                    allocation_helpers::copy_allocation_to_allocation(&mut new_conv, &layer.conv_state);
                    let mut new_ssm = allocation_helpers::create_allocation(context, &layer.ssm_shape, layer.data_type);
                    allocation_helpers::copy_allocation_to_allocation(&mut new_ssm, &layer.ssm_state);

                    CacheLayer::DeltaNet(DeltaNetLayer {
                        conv_state: new_conv,
                        conv_shape: layer.conv_shape,
                        ssm_state: new_ssm,
                        ssm_shape: layer.ssm_shape,
                        data_type: layer.data_type,
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
