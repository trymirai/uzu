use std::cell::Cell;

use crate::{
    array::size_for_shape,
    backends::common::{AllocationType, Backend, Context, Encoder},
    config::token_mixer::AnyTokenMixerConfig,
    data_type::DataType,
    encodable_block::KVCacheUpdate,
    forward_pass::{
        delta_net_layer::DeltaNetLayer,
        kv_cache_layer::{KVCacheLayerState, KVCacheLayerTrait, KVSlice},
        model_shape::ModelShape,
        short_conv_layer::ShortConvLayer,
        ssm_layer::SSMLayer,
    },
};

pub enum CacheLayer<B: Backend> {
    Transformer(Box<dyn KVCacheLayerTrait<B>>),
    StateSpace(SSMLayer<B>),
    ShortConv(ShortConvLayer<B>),
    DeltaNet(DeltaNetLayer<B>),
}

pub enum CacheLayerSlice<B: Backend> {
    Transformer(KVSlice<B>),
    StateSpace,
    ShortConv,
    DeltaNet,
}

impl<B: Backend> CacheLayer<B> {
    pub fn as_transformer(&self) -> Option<&dyn KVCacheLayerTrait<B>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer.as_ref()),
            _ => None,
        }
    }

    pub fn as_transformer_mut(&mut self) -> Option<&mut dyn KVCacheLayerTrait<B>> {
        match self {
            CacheLayer::Transformer(layer) => Some(layer.as_mut()),
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
    entries: Box<[CacheLayer<B>]>,
    bindings: Box<[LayerCacheBinding]>,
}

pub struct CacheLayersSlice<B: Backend> {
    pub layers: Vec<CacheLayerSlice<B>>,
}

#[derive(Clone, Copy)]
struct CacheEntryIndex {
    index: usize,
}

#[derive(Clone, Copy)]
enum LayerCacheBinding {
    Owned {
        entry: CacheEntryIndex,
    },
}

pub enum LayerCacheAccess<'a, B: Backend> {
    Owned {
        entry: &'a mut CacheLayer<B>,
    },
}

impl<B: Backend> CacheLayers<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Self {
        let total_context_length = max_prefix_length.max(max_suffix_length);

        let mut entries: Vec<CacheLayer<B>> = Vec::with_capacity(model_shape.layer_mixers().len());
        let mut bindings: Vec<LayerCacheBinding> = Vec::with_capacity(model_shape.layer_mixers().len());
        for mixer in model_shape.layer_mixers().iter() {
            let layer = match mixer {
                AnyTokenMixerConfig::AttentionConfig(attn) => {
                    let sliding_window = attn.sliding_window_size;
                    let length = sliding_window.unwrap_or(max_prefix_length);
                    let shape = [length + max_suffix_length, attn.num_groups, attn.head_dim];
                    let window_length = sliding_window.filter(|&window_size| window_size < total_context_length);

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
                    let kv_layer = <dyn KVCacheLayerTrait<B>>::new(context, &state, shape, model_shape.data_type)
                        .expect("Failed to create KVCacheLayer");
                    CacheLayer::Transformer(kv_layer)
                },
                AnyTokenMixerConfig::Mamba2Config(c) => {
                    let data_type = DataType::F32;
                    let conv_shape = [c.conv_dim(), c.kernel_size.saturating_sub(1)];
                    let ssm_shape = [c.num_heads, c.head_dim, c.state_dim];
                    let conv_bytes = size_for_shape(&conv_shape, data_type);
                    let ssm_bytes = size_for_shape(&ssm_shape, data_type);

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
                        data_type,
                    })
                },
                AnyTokenMixerConfig::ShortConvConfig(c) => {
                    assert!(c.kernel_size >= 2, "ShortConv kernel_size must be >= 2, got {}", c.kernel_size);
                    let conv_shape = [model_shape.model_dim(), c.kernel_size - 1];
                    let suffix_state_shape = [max_suffix_length, model_shape.model_dim(), c.kernel_size - 1];
                    let conv_bytes = size_for_shape(&conv_shape, model_shape.data_type);
                    let suffix_bytes = size_for_shape(&suffix_state_shape, model_shape.data_type);

                    CacheLayer::ShortConv(ShortConvLayer {
                        conv_state: context
                            .create_allocation(conv_bytes, AllocationType::Global)
                            .expect("Failed to create short conv allocation"),
                        conv_shape,
                        suffix_state: context
                            .create_allocation(suffix_bytes, AllocationType::Global)
                            .expect("Failed to create short conv suffix allocation"),
                        suffix_shape: suffix_state_shape,
                        data_type: model_shape.data_type,
                        suffix_state_valid_start: Cell::new(0),
                        suffix_state_valid_len: Cell::new(0),
                    })
                },
                AnyTokenMixerConfig::DeltaNetConfig(c) => {
                    let conv_data_type = DataType::F32;
                    let ssm_data_type = DataType::F32;
                    let conv_shape = [c.conv_dim(), c.kernel_size.saturating_sub(1)];
                    let ssm_shape = [c.num_heads, c.value_head_dim, c.head_dim];
                    let conv_bytes = size_for_shape(&conv_shape, conv_data_type);
                    let ssm_bytes = size_for_shape(&ssm_shape, ssm_data_type);

                    CacheLayer::DeltaNet(DeltaNetLayer {
                        conv_state: context
                            .create_allocation(conv_bytes, AllocationType::Global)
                            .expect("Failed to create delta net conv allocation"),
                        conv_shape,
                        ssm_state: context
                            .create_allocation(ssm_bytes, AllocationType::Global)
                            .expect("Failed to create delta net ssm allocation"),
                        ssm_shape,
                        data_type: ssm_data_type,
                    })
                },
            };
            bindings.push(LayerCacheBinding::Owned {
                entry: CacheEntryIndex {
                    index: entries.len(),
                },
            });
            entries.push(layer);
        }
        let mut entries = entries.into_boxed_slice();
        let bindings = bindings.into_boxed_slice();

        let mut encoder: Encoder<B> = Encoder::new(context).expect("Failed to create cache initialization encoder");
        for layer in entries.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => {
                    layer.encode_zero(&mut encoder);
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
            entries,
            bindings,
        }
    }

    pub fn cache_for_layer(
        &mut self,
        layer_index: usize,
    ) -> LayerCacheAccess<'_, B> {
        match self.bindings[layer_index] {
            LayerCacheBinding::Owned {
                entry,
            } => LayerCacheAccess::Owned {
                entry: &mut self.entries[entry.index],
            },
        }
    }

    pub fn iter_layers(&self) -> impl Iterator<Item = (usize, &CacheLayer<B>)> {
        self.bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| match binding {
                LayerCacheBinding::Owned {
                    entry,
                } => Some((index, &self.entries[entry.index])),
            })
            .flatten()
    }

    pub fn clear(
        &mut self,
        context: &B::Context,
    ) {
        let mut encoder: Option<Encoder<B>> = None;
        for layer in self.entries.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => layer.clear_state(),
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
        context: &B::Context,
        encoder: &mut Encoder<B>,
        kv_cache_update: &KVCacheUpdate<B>,
    ) {
        let short_conv_commit_index = accepted_suffix_indices.last().copied().unwrap_or(0);
        for layer in self.entries.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(accepted_suffix_indices, suffix_start, context, encoder, kv_cache_update);
            } else if let Some(layer) = layer.as_short_conv_mut() {
                layer.commit_from_suffix_state_if_valid(short_conv_commit_index, encoder);
            }
        }
    }

    pub fn prepare_for_forward_pass(
        &mut self,
        context: &B::Context,
        active_row_count: usize,
    ) {
        if active_row_count == 0 {
            return;
        }

        for layer in self.entries.iter_mut() {
            let Some(layer) = layer.as_transformer_mut() else {
                continue;
            };
            let row_range = match layer.state() {
                KVCacheLayerState::Full {
                    prefix_len,
                } => prefix_len..prefix_len + active_row_count,
                KVCacheLayerState::Windowed {
                    window_length,
                    ..
                } => 0..window_length + active_row_count,
            };
            layer.map_row_range(context, row_range).expect("Failed to map KV cache rows for forward pass");
        }
    }

    pub fn register_accepted_tokens(
        &mut self,
        number_of_accepted_tokens: usize,
    ) {
        for layer in self.entries.iter_mut() {
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
        let mut layers = Vec::with_capacity(self.entries.len());
        let mut encoder: Option<Encoder<B>> = None;

        for layer in self.entries.iter() {
            match layer {
                CacheLayer::Transformer(kv) => {
                    let encoder =
                        encoder.get_or_insert_with(|| Encoder::new(context).expect("Failed to create Encoder"));
                    let slice = kv.slice(context, encoder, range.clone())?;
                    layers.push(CacheLayerSlice::Transformer(slice));
                },
                CacheLayer::StateSpace(_) => layers.push(CacheLayerSlice::StateSpace),
                CacheLayer::ShortConv(_) => layers.push(CacheLayerSlice::ShortConv),
                CacheLayer::DeltaNet(_) => layers.push(CacheLayerSlice::DeltaNet),
            }
        }

        if let Some(encoder) = encoder {
            encoder.end_encoding().submit().wait_until_completed().expect("Failed to end and wait encoder");
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
        let mut encoder: Option<Encoder<B>> = None;
        for (layer, snapshot) in self.entries.iter_mut().zip(slice.layers.iter()) {
            match (layer, snapshot) {
                (CacheLayer::Transformer(kv), CacheLayerSlice::Transformer(s)) => {
                    let encoder =
                        encoder.get_or_insert_with(|| Encoder::new(context).expect("Failed to create Encoder"));
                    kv.apply_slice(context, encoder, s, range.clone());
                },
                (CacheLayer::StateSpace(_), CacheLayerSlice::StateSpace) => {},
                (CacheLayer::ShortConv(_), CacheLayerSlice::ShortConv) => {},
                _ => {},
            }
        }

        if let Some(encoder) = encoder {
            encoder.end_encoding().submit().wait_until_completed().expect("Failed to end and wait encoder");
        }
    }

    pub fn copy_from(
        &mut self,
        source: &Self,
        context: &B::Context,
    ) {
        self.copy_metadata_and_windowed_data_from(source, context);

        let mut encoder = Encoder::new(context).expect("Failed to create cache layer copy encoder");
        self.encode_copy_data_from(source, context, &mut encoder);
        encoder.end_encoding().submit().wait_until_completed().expect("Failed to copy cache layers");
    }

    fn copy_metadata_and_windowed_data_from(
        &mut self,
        source: &Self,
        context: &B::Context,
    ) {
        assert_eq!(source.entries.len(), self.entries.len(), "cache layer count mismatch");

        // Key/value arrays stay alive until after encoder.submit().wait_until_completed() returns, then explicitly drops them.
        // This prevents the slice buffers from being freed while the GPU is still reading from them during windowed cache cloning.
        let mut pending_slices: Vec<KVSlice<B>> = Vec::new();
        let mut encoder: Option<Encoder<B>> = None;

        for (source_layer, destination_layer) in source.entries.iter().zip(self.entries.iter_mut()) {
            match (source_layer, destination_layer) {
                (CacheLayer::Transformer(source), CacheLayer::Transformer(destination)) => {
                    let copy_rows = source.prefix_segment_length();
                    if copy_rows > 0 && matches!(source.state(), KVCacheLayerState::Windowed { .. }) {
                        let encoder =
                            encoder.get_or_insert_with(|| Encoder::new(context).expect("Failed to create Encoder"));
                        let slice =
                            source.slice(context, encoder, 0..copy_rows).expect("Failed to slice KV cache layer");
                        destination.apply_slice(context, encoder, &slice, None);
                        pending_slices.push(slice);
                    }
                    destination.set_state(&source.state());
                },
                (CacheLayer::StateSpace(_), CacheLayer::StateSpace(_)) => {},
                (CacheLayer::ShortConv(_), CacheLayer::ShortConv(destination)) => {
                    destination.clear_suffix_state_valid_range();
                },
                (CacheLayer::DeltaNet(_), CacheLayer::DeltaNet(_)) => {},
                _ => panic!("cache layer type mismatch while copying cache layers"),
            }
        }

        if let Some(encoder) = encoder {
            encoder.end_encoding().submit().wait_until_completed().expect("Failed to end and wait encoder");
        }
        drop(pending_slices);
    }

    fn encode_copy_data_from(
        &mut self,
        source: &Self,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) {
        assert_eq!(source.entries.len(), self.entries.len(), "cache layer count mismatch");

        for (source_layer, destination_layer) in source.entries.iter().zip(self.entries.iter_mut()) {
            match (source_layer, destination_layer) {
                (CacheLayer::Transformer(source), CacheLayer::Transformer(destination)) => {
                    if matches!(source.state(), KVCacheLayerState::Full { .. }) {
                        source.encode_copy_prefix_rows_to(
                            destination.as_mut(),
                            source.prefix_segment_length(),
                            context,
                            encoder,
                        );
                    }
                },
                (CacheLayer::StateSpace(source), CacheLayer::StateSpace(destination)) => {
                    match (source.conv_state.as_ref(), destination.conv_state.as_mut()) {
                        (Some(source_conv_state), Some(destination_conv_state)) => {
                            encoder.encode_copy(source_conv_state, .., destination_conv_state, ..);
                        },
                        (None, None) => {},
                        _ => panic!("state-space conv_state presence mismatch while copying cache layers"),
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
                _ => panic!("cache layer type mismatch while copying cache layers"),
            }
        }
    }

    pub fn clone(
        &self,
        context: &B::Context,
    ) -> Self {
        let mut max_prefix_capacity_across_layers = 0usize;
        let mut entries: Box<[CacheLayer<B>]> = self
            .entries
            .iter()
            .map(|layer| match layer {
                CacheLayer::Transformer(layer) => {
                    let shape = layer.shape();
                    let [_, num_groups, head_dim] = shape;
                    let dtype = layer.data_type();
                    let copy_rows = layer.prefix_segment_length();

                    let new_total_len = copy_rows + self.max_suffix_length;
                    if copy_rows > max_prefix_capacity_across_layers {
                        max_prefix_capacity_across_layers = copy_rows;
                    }

                    let new_shape = [new_total_len, num_groups, head_dim];
                    let kv_layer = <dyn KVCacheLayerTrait<B>>::new(context, &layer.state(), new_shape, dtype)
                        .expect("Failed to create KVCacheLayer");
                    CacheLayer::Transformer(kv_layer)
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
        for layer in entries.iter_mut() {
            match layer {
                CacheLayer::Transformer(layer) => {
                    layer.encode_zero(&mut zero_encoder);
                },
                CacheLayer::ShortConv(layer) => {
                    zero_encoder.encode_fill(&mut layer.suffix_state, 0);
                },
                _ => {},
            }
        }
        zero_encoder.end_encoding().submit().wait_until_completed().expect("Failed to zero cloned cache layers");

        let mut cloned = Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: max_prefix_capacity_across_layers,
            entries,
            bindings: self.bindings.clone(),
        };
        cloned.copy_from(self, context);
        cloned
    }
}

#[cfg(test)]
#[path = "../../unit/forward_pass/cache_layers_test.rs"]
mod tests;
