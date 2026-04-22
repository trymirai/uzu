use std::cell::Cell;

use crate::{
    array::size_for_shape,
    backends::common::{AllocationType, Backend, Context, Encoder, kernel::kv_cache_update::KVCacheUpdate},
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

        let mut data: Box<[CacheLayer<B>]> = model_shape
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
                    let kv_bytes = size_for_shape(&shape, model_shape.kv_cache_data_type());

                    CacheLayer::Transformer(KVCacheLayer {
                        state: state.clone(),
                        keys: context
                            .create_allocation(kv_bytes, AllocationType::Global)
                            .expect("Failed to create kv keys allocation"),
                        values: context
                            .create_allocation(kv_bytes, AllocationType::Global)
                            .expect("Failed to create kv values allocation"),
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
                    let conv_bytes = size_for_shape(&conv_shape, dtype);
                    let ssm_bytes = size_for_shape(&ssm_shape, dtype);

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: (conv_bytes > 0).then(|| {
                            context
                                .create_allocation(conv_bytes, AllocationType::Global)
                                .expect("Failed to create ssm conv allocation")
                        }),
                        conv_shape,
                        ssm_state: context
                            .create_allocation(ssm_bytes, AllocationType::Global)
                            .expect("Failed to create ssm state allocation"),
                        ssm_shape,
                        data_type: dtype,
                    })
                },
                DecoderLayerType::ShortConv {
                    kernel_size,
                } => {
                    assert!(*kernel_size >= 2, "ShortConv kernel_size must be >= 2, got {}", kernel_size);
                    let conv_shape = [model_shape.model_dim(), kernel_size - 1];
                    let suffix_state_shape = [max_suffix_length, model_shape.model_dim(), kernel_size - 1];
                    let dtype = model_shape.activation_data_type();
                    let conv_bytes = size_for_shape(&conv_shape, dtype);
                    let suffix_bytes = size_for_shape(&suffix_state_shape, dtype);

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: context
                            .create_allocation(conv_bytes, AllocationType::Global)
                            .expect("Failed to create short conv allocation"),
                        conv_shape,
                        suffix_state: context
                            .create_allocation(suffix_bytes, AllocationType::Global)
                            .expect("Failed to create short conv suffix allocation"),
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
                    let conv_bytes = size_for_shape(&conv_shape, dtype);
                    let ssm_bytes = size_for_shape(&ssm_shape, dtype);

                    CacheLayer::DeltaNet(DeltaNetLayer {
                        conv_state: context
                            .create_allocation(conv_bytes, AllocationType::Global)
                            .expect("Failed to create delta net conv allocation"),
                        conv_shape,
                        ssm_state: context
                            .create_allocation(ssm_bytes, AllocationType::Global)
                            .expect("Failed to create delta net ssm allocation"),
                        ssm_shape,
                        data_type: dtype,
                    })
                },
            })
            .collect();

        let mut encoder: Encoder<B> = Encoder::new(context).expect("Failed to create cache initialization encoder");
        for layer in data.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => {
                    encoder.encode_fill(&mut layer.keys, 0);
                    encoder.encode_fill(&mut layer.values, 0);
                },
                CacheLayer::StateSpace(layer) => {
                    if let Some(conv_state) = layer.conv_state.as_mut() {
                        encoder.encode_fill(conv_state, 0);
                    }
                    encoder.encode_fill(&mut layer.ssm_state, 0);
                },
                CacheLayer::ShortConv(layer) => {
                    encoder.encode_fill(&mut layer.conv_state, 0);
                    encoder.encode_fill(&mut layer.suffix_state, 0);
                },
                CacheLayer::DeltaNet(layer) => {
                    encoder.encode_fill(&mut layer.conv_state, 0);
                    encoder.encode_fill(&mut layer.ssm_state, 0);
                },
            }
        }
        encoder.end_encoding().submit().wait_until_completed().expect("Failed to initialize cache allocations");

        Self {
            max_suffix_length,
            max_prefix_length,
            data,
        }
    }

    pub fn clear(
        &mut self,
        context: &B::Context,
    ) {
        let mut encoder: Option<Encoder<B>> = None;
        for layer in self.data.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => match &mut layer.state {
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
                },
                CacheLayer::StateSpace(layer) => {
                    let encoder = encoder
                        .get_or_insert_with(|| Encoder::new(context).expect("Failed to create cache clear encoder"));
                    if let Some(conv_state) = layer.conv_state.as_mut() {
                        encoder.encode_fill(conv_state, 0);
                    }
                    encoder.encode_fill(&mut layer.ssm_state, 0);
                },
                CacheLayer::ShortConv(layer) => {
                    let encoder = encoder
                        .get_or_insert_with(|| Encoder::new(context).expect("Failed to create cache clear encoder"));
                    encoder.encode_fill(&mut layer.conv_state, 0);
                    encoder.encode_fill(&mut layer.suffix_state, 0);
                    layer.clear_suffix_state_valid_range();
                },
                CacheLayer::DeltaNet(layer) => {
                    let encoder = encoder
                        .get_or_insert_with(|| Encoder::new(context).expect("Failed to create cache clear encoder"));
                    encoder.encode_fill(&mut layer.conv_state, 0);
                    encoder.encode_fill(&mut layer.ssm_state, 0);
                },
            }
        }
        if let Some(encoder) = encoder {
            encoder.end_encoding().submit().wait_until_completed().expect("Failed to clear cache layers");
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
                layer.commit_from_suffix_state_if_valid(short_conv_commit_index, encoder);
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
        let mut data: Box<[CacheLayer<B>]> = self
            .data
            .iter()
            .map(|layer| match layer {
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
                    let new_bytes = size_for_shape(&new_shape, dtype);
                    let new_keys = context
                        .create_allocation(new_bytes, AllocationType::Global)
                        .expect("Failed to create kv keys clone allocation");
                    let new_values = context
                        .create_allocation(new_bytes, AllocationType::Global)
                        .expect("Failed to create kv values clone allocation");

                    CacheLayer::Transformer(KVCacheLayer {
                        state: layer.state.clone(),
                        keys: new_keys,
                        values: new_values,
                        shape: new_shape,
                        data_type: dtype,
                    })
                },
                CacheLayer::StateSpace(layer) => {
                    let conv_bytes = size_for_shape(&layer.conv_shape, layer.data_type);
                    let ssm_bytes = size_for_shape(&layer.ssm_shape, layer.data_type);
                    let new_conv = (conv_bytes > 0).then(|| {
                        context
                            .create_allocation(conv_bytes, AllocationType::Global)
                            .expect("Failed to create ssm conv clone allocation")
                    });
                    let new_ssm = context
                        .create_allocation(ssm_bytes, AllocationType::Global)
                        .expect("Failed to create ssm state clone allocation");

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: new_conv,
                        conv_shape: layer.conv_shape,
                        ssm_state: new_ssm,
                        ssm_shape: layer.ssm_shape,
                        data_type: layer.data_type,
                    })
                },
                CacheLayer::ShortConv(layer) => {
                    let conv_bytes = size_for_shape(&layer.conv_shape, layer.data_type);
                    let suffix_bytes = size_for_shape(&layer.suffix_shape, layer.data_type);
                    let new_conv = context
                        .create_allocation(conv_bytes, AllocationType::Global)
                        .expect("Failed to create short conv clone allocation");
                    let new_suffix = context
                        .create_allocation(suffix_bytes, AllocationType::Global)
                        .expect("Failed to create short conv suffix clone allocation");

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
                    let conv_bytes = size_for_shape(&layer.conv_shape, layer.data_type);
                    let ssm_bytes = size_for_shape(&layer.ssm_shape, layer.data_type);
                    let new_conv = context
                        .create_allocation(conv_bytes, AllocationType::Global)
                        .expect("Failed to create delta net conv clone allocation");
                    let new_ssm = context
                        .create_allocation(ssm_bytes, AllocationType::Global)
                        .expect("Failed to create delta net ssm clone allocation");

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

        let mut zero_encoder: Encoder<B> = Encoder::new(context).expect("Failed to create cache clone zero encoder");
        for layer in data.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => {
                    zero_encoder.encode_fill(&mut layer.keys, 0);
                    zero_encoder.encode_fill(&mut layer.values, 0);
                },
                CacheLayer::ShortConv(layer) => {
                    zero_encoder.encode_fill(&mut layer.suffix_state, 0);
                },
                _ => {},
            }
        }
        zero_encoder.end_encoding().submit().wait_until_completed().expect("Failed to zero cloned cache layers");

        for (source_layer, destination_layer) in self.data.iter().zip(data.iter_mut()) {
            if let (CacheLayer::Transformer(source), CacheLayer::Transformer(destination)) =
                (source_layer, destination_layer)
            {
                let copy_rows = source.prefix_segment_length();
                if copy_rows > 0 && matches!(source.state, KVCacheLayerState::Windowed { .. }) {
                    let slice = source.slice(context, 0..copy_rows).expect("Failed to slice KV cache layer");
                    destination.apply_slice(&slice, None);
                }
            }
        }

        let mut encoder = Encoder::new(context).expect("Failed to create cache clone encoder");
        for (source_layer, destination_layer) in self.data.iter().zip(data.iter_mut()) {
            match (source_layer, destination_layer) {
                (CacheLayer::Transformer(source), CacheLayer::Transformer(destination)) => {
                    if matches!(source.state, KVCacheLayerState::Full { .. }) {
                        source.encode_copy_prefix_rows_to(destination, source.prefix_segment_length(), &mut encoder);
                    }
                },
                (CacheLayer::StateSpace(source), CacheLayer::StateSpace(destination)) => {
                    match (source.conv_state.as_ref(), destination.conv_state.as_mut()) {
                        (Some(source_conv_state), Some(destination_conv_state)) => {
                            encoder.encode_copy(source_conv_state, .., destination_conv_state, ..);
                        },
                        (None, None) => {},
                        _ => panic!("state-space conv_state presence mismatch"),
                    }
                    encoder.encode_copy(&source.ssm_state, .., &mut destination.ssm_state, ..);
                },
                (CacheLayer::ShortConv(source), CacheLayer::ShortConv(destination)) => {
                    encoder.encode_copy(&source.conv_state, .., &mut destination.conv_state, ..);
                },
                (CacheLayer::DeltaNet(source), CacheLayer::DeltaNet(destination)) => {
                    encoder.encode_copy(&source.conv_state, .., &mut destination.conv_state, ..);
                    encoder.encode_copy(&source.ssm_state, .., &mut destination.ssm_state, ..);
                },
                _ => {},
            }
        }
        encoder.end_encoding().submit().wait_until_completed().expect("Failed to clone cache layers");

        Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: max_prefix_capacity_across_layers,
            data,
        }
    }
}
