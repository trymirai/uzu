use std::{cell::RefCell, collections::HashMap};

use metal::CommandBuffer as MTLCommandBuffer;

use super::{
    super::{MTLContext, MetalArray},
    kv_cache_layer::{INVALID_POSITION, KVCacheLayer, KVCacheLayerState},
    model_shape::ModelShape,
    ssm_layer::SSMLayer,
};
use crate::{
    DeviceContext, array::Array, backends::metal::kernel::KVCacheUpdate,
    config::DecoderLayerType,
};

#[derive(Debug)]
pub enum CacheLayer {
    Transformer(KVCacheLayer),
    StateSpace(SSMLayer),
}

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
}

pub struct CacheLayers {
    max_suffix_length: usize,
    max_prefix_length: usize,
    pub data: Box<[CacheLayer]>,
}

impl CacheLayers {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Self {
        let total_context_length = max_prefix_length + max_suffix_length;
        let kv_shapes: Vec<[usize; 3]> = model_shape
            .kv_cache_layer_shapes(max_prefix_length, max_suffix_length)
            .collect();

        let data: Box<[CacheLayer]> = model_shape
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
                        keys: RefCell::new(
                            context.array(
                                &shape,
                                model_shape.kv_cache_data_type(),
                            ),
                        ),
                        values: RefCell::new(
                            context.array(
                                &shape,
                                model_shape.kv_cache_data_type(),
                            ),
                        ),
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
                } => {
                    let batch_size = max_suffix_length;
                    let conv_shape =
                        [batch_size, *conv_dim, kernel_size.saturating_sub(1)];
                    let ssm_shape = [
                        batch_size,
                        model_shape.num_heads(),
                        model_shape.head_dim(),
                        *state_dim,
                    ];
                    let dtype = model_shape.activation_data_type();

                    let packed_shape = [batch_size, *conv_dim];
                    let x_shape = [
                        batch_size,
                        model_shape.num_heads(),
                        model_shape.head_dim(),
                    ];
                    let z_shape = x_shape;
                    let dt_shape = [batch_size, model_shape.num_heads()];
                    let decay_shape = dt_shape;
                    let bc_shape =
                        [batch_size, model_shape.num_groups(), *state_dim];

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: RefCell::new(
                            context.array(&conv_shape, dtype),
                        ),
                        ssm_state: RefCell::new(
                            context.array(&ssm_shape, dtype),
                        ),
                        packed: RefCell::new(
                            context.array(&packed_shape, dtype),
                        ),
                        x: RefCell::new(context.array(&x_shape, dtype)),
                        b: RefCell::new(context.array(&bc_shape, dtype)),
                        c: RefCell::new(context.array(&bc_shape, dtype)),
                        dt: RefCell::new(context.array(&dt_shape, dtype)),
                        decay: RefCell::new(context.array(&decay_shape, dtype)),
                        z: RefCell::new(context.array(&z_shape, dtype)),
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
        context: &MTLContext,
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

    pub fn update_after_acceptance(
        &mut self,
        accepted_suffix_indices: &[usize],
        suffix_start: Option<usize>,
        command_buffer: &MTLCommandBuffer,
        kv_cache_update: &KVCacheUpdate,
    ) {
        for layer in self.data.iter_mut() {
            if let Some(layer) = layer.as_transformer_mut() {
                layer.update_after_acceptance(
                    accepted_suffix_indices,
                    suffix_start,
                    command_buffer,
                    kv_cache_update,
                );
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

    pub fn clone_with_prefix_len(
        &self,
        context: &MTLContext,
        prefix_len: usize,
    ) -> Self {
        let new_total_len = prefix_len + self.max_suffix_length;
        let data: Box<[CacheLayer]> = self
            .data
            .iter()
            .map(|layer| match layer {
                CacheLayer::Transformer(layer) => {
                    let shape = layer.keys.borrow().shape().to_vec();
                    let num_groups = shape[0];
                    let head_dim = shape[2];
                    let dtype = layer.keys.borrow().data_type();
                    let new_shape = [num_groups, new_total_len, head_dim];

                    let mut new_keys = context.array(&new_shape, dtype);
                    let mut new_values = context.array(&new_shape, dtype);

                    let mut copy_rows = layer.prefix_segment_length();
                    if let Some(window_length) = layer.window_length() {
                        copy_rows = std::cmp::min(copy_rows, window_length);
                    }

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
                    let mut new_conv = context.array(&conv_shape, conv_dtype);
                    {
                        let conv_src = layer.conv_state.borrow();
                        new_conv.copy_from_array(&conv_src);
                    }

                    let ssm_shape = layer.ssm_state.borrow().shape().to_vec();
                    let ssm_dtype = layer.ssm_state.borrow().data_type();
                    let mut new_ssm = context.array(&ssm_shape, ssm_dtype);
                    {
                        let ssm_src = layer.ssm_state.borrow();
                        new_ssm.copy_from_array(&ssm_src);
                    }

                    let packed_shape = layer.packed.borrow().shape().to_vec();
                    let packed_dtype = layer.packed.borrow().data_type();
                    let mut new_packed =
                        context.array(&packed_shape, packed_dtype);
                    {
                        let src = layer.packed.borrow();
                        new_packed.copy_from_array(&src);
                    }

                    let x_shape = layer.x.borrow().shape().to_vec();
                    let x_dtype = layer.x.borrow().data_type();
                    let mut new_x = context.array(&x_shape, x_dtype);
                    {
                        let src = layer.x.borrow();
                        new_x.copy_from_array(&src);
                    }

                    let b_shape = layer.b.borrow().shape().to_vec();
                    let b_dtype = layer.b.borrow().data_type();
                    let mut new_b = context.array(&b_shape, b_dtype);
                    {
                        let src = layer.b.borrow();
                        new_b.copy_from_array(&src);
                    }

                    let c_shape = layer.c.borrow().shape().to_vec();
                    let c_dtype = layer.c.borrow().data_type();
                    let mut new_c = context.array(&c_shape, c_dtype);
                    {
                        let src = layer.c.borrow();
                        new_c.copy_from_array(&src);
                    }

                    let dt_shape = layer.dt.borrow().shape().to_vec();
                    let dt_dtype = layer.dt.borrow().data_type();
                    let mut new_dt = context.array(&dt_shape, dt_dtype);
                    {
                        let src = layer.dt.borrow();
                        new_dt.copy_from_array(&src);
                    }

                    let decay_shape = layer.decay.borrow().shape().to_vec();
                    let decay_dtype = layer.decay.borrow().data_type();
                    let mut new_decay =
                        context.array(&decay_shape, decay_dtype);
                    {
                        let src = layer.decay.borrow();
                        new_decay.copy_from_array(&src);
                    }

                    let z_shape = layer.z.borrow().shape().to_vec();
                    let z_dtype = layer.z.borrow().data_type();
                    let mut new_z = context.array(&z_shape, z_dtype);
                    {
                        let src = layer.z.borrow();
                        new_z.copy_from_array(&src);
                    }

                    CacheLayer::StateSpace(SSMLayer {
                        conv_state: RefCell::new(new_conv),
                        ssm_state: RefCell::new(new_ssm),
                        packed: RefCell::new(new_packed),
                        x: RefCell::new(new_x),
                        b: RefCell::new(new_b),
                        c: RefCell::new(new_c),
                        dt: RefCell::new(new_dt),
                        decay: RefCell::new(new_decay),
                        z: RefCell::new(new_z),
                    })
                },
            })
            .collect();

        Self {
            max_suffix_length: self.max_suffix_length,
            max_prefix_length: prefix_len,
            data,
        }
    }
}
