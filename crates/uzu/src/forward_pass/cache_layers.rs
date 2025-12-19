use std::{cell::RefCell, collections::HashMap};

use super::{
    kv_cache_layer::{
        AttentionBiasUpdate, INVALID_POSITION, KVCacheLayer, KVCacheLayerState,
        KVSlice,
    },
    model_shape::ModelShape,
    short_conv_layer::ShortConvLayer,
    ssm_layer::SSMLayer,
};
use crate::{Array, DeviceContext, config::DecoderLayerType};

#[derive(Debug)]
pub enum CacheLayer<C: DeviceContext> {
    Transformer(KVCacheLayer<C>),
    StateSpace(SSMLayer<C>),
    ShortConv(ShortConvLayer<C>),
}

#[derive(Clone)]
pub enum CacheLayerSlice<C: DeviceContext> {
    Transformer(KVSlice<C>),
    StateSpace,
    ShortConv,
}

impl<C: DeviceContext> CacheLayer<C> {
    pub fn as_transformer(&self) -> Option<&KVCacheLayer<C>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_transformer_mut(&mut self) -> Option<&mut KVCacheLayer<C>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space(&self) -> Option<&SSMLayer<C>> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_state_space_mut(&mut self) -> Option<&mut SSMLayer<C>> {
        match self {
            CacheLayer::StateSpace(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv(&self) -> Option<&ShortConvLayer<C>> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn as_short_conv_mut(&mut self) -> Option<&mut ShortConvLayer<C>> {
        match self {
            CacheLayer::ShortConv(layer) => Some(layer),
            _ => None,
        }
    }
}

pub struct CacheLayers<C: DeviceContext> {
    max_suffix_length: usize,
    max_prefix_length: usize,
    pub data: Box<[CacheLayer<C>]>,
}

#[derive(Clone)]
pub struct CacheLayersSlice<C: DeviceContext> {
    pub layers: Vec<CacheLayerSlice<C>>,
}

impl<C: DeviceContext> CacheLayers<C> {
    pub fn new(
        context: &C,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Self {
        let total_context_length = max_prefix_length + max_suffix_length;
        let kv_shapes: Vec<[usize; 3]> = model_shape
            .kv_cache_layer_shapes(max_prefix_length, max_suffix_length)
            .collect();

        let data: Box<[CacheLayer<C>]> =
            model_shape
                .layer_types()
                .iter()
                .enumerate()
                .map(|(layer_idx, layer_type)| match layer_type {
                    DecoderLayerType::Transformer => {
                        let shape = kv_shapes[layer_idx];
                        let window_length = model_shape
                            .sliding_window_length_per_layer[layer_idx]
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
                            keys: RefCell::new(context.array(
                                &shape,
                                model_shape.kv_cache_data_type(),
                            )),
                            values: RefCell::new(context.array(
                                &shape,
                                model_shape.kv_cache_data_type(),
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
                        let conv_shape =
                            [*conv_dim, kernel_size.saturating_sub(1)];
                        let ssm_shape = [*num_heads, *head_dim, *state_dim];
                        let dtype = model_shape.activation_data_type();

                        CacheLayer::StateSpace(SSMLayer {
                            conv_state: RefCell::new(
                                context.array(&conv_shape, dtype),
                            ),
                            ssm_state: RefCell::new(
                                context.array(&ssm_shape, dtype),
                            ),
                        })
                    },
                    DecoderLayerType::ShortConv {
                        kernel_size,
                    } => {
                        let conv_shape = [
                            model_shape.model_dim(),
                            kernel_size.saturating_sub(1),
                        ];
                        let dtype = model_shape.activation_data_type();

                        CacheLayer::ShortConv(ShortConvLayer {
                            conv_state: RefCell::new(
                                context.array(&conv_shape, dtype),
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
        dst: &mut HashMap<Option<usize>, C::DeviceArray>,
        suffix_token_positions: &[usize],
        suffix_length: usize,
        context: &C,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) {
        for layer in self.data.iter() {
            if let CacheLayer::Transformer(layer) = layer {
                let key: Option<usize> = match &layer.state {
                    KVCacheLayerState::Full {
                        ..
                    } => None,
                    KVCacheLayerState::Windowed {
                        window_length,
                        ..
                    } => Some(*window_length),
                };

                if let Some(array) = dst.get_mut(&key) {
                    layer.fill_attention_bias(
                        array,
                        suffix_token_positions,
                        suffix_length,
                        context,
                        external_bias_fn,
                    );
                }
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
        context: &C,
        range: std::ops::Range<usize>,
    ) -> Option<CacheLayersSlice<C>> {
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
        slice: &CacheLayersSlice<C>,
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
        context: &C,
    ) -> Self {
        let mut max_prefix_capacity_across_layers = 0usize;
        let data: Box<[CacheLayer<C>]> = self
            .data
            .iter()
            .map(|layer| match layer {
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
                    let mut new_keys = context.array(&new_shape, dtype);
                    let mut new_values = context.array(&new_shape, dtype);

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
                        max_suffix_length: self.max_suffix_length,
                    })
                },
                CacheLayer::StateSpace(layer) => {
                    let conv = layer.conv_state.borrow();
                    let mut new_conv =
                        context.array(conv.shape(), conv.data_type());
                    new_conv.copy_from(&conv);

                    let ssm = layer.ssm_state.borrow();
                    let mut new_ssm =
                        context.array(ssm.shape(), ssm.data_type());
                    new_ssm.copy_from(&ssm);

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: RefCell::new(new_conv),
                        ssm_state: RefCell::new(new_ssm),
                    })
                },
                CacheLayer::ShortConv(layer) => {
                    let conv = layer.conv_state.borrow();
                    let mut new_conv =
                        context.array(conv.shape(), conv.data_type());
                    new_conv.copy_from(&conv);

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: RefCell::new(new_conv),
                    })
                },
            })
            .collect();

        if max_prefix_capacity_across_layers > self.max_prefix_length {
            panic!(
                "Cached items count {} exceeds max_prefix_length {}",
                max_prefix_capacity_across_layers, self.max_prefix_length
            );
        }

        Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: self.max_prefix_length,
            data,
        }
    }
}
