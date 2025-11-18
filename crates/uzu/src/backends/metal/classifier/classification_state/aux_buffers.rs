use std::cell::RefCell;

use super::types::ArrayCell;
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MetalArray,
        forward_pass::{ForwardPassBuffers, ModelShape},
    },
    config::DecoderConfig,
};

pub(super) struct AuxBuffers {
    pub(super) suffix_length: usize,
    pub(super) main: ArrayCell,
    pub(super) shortcut: ArrayCell,
    pub(super) qkv: ArrayCell,
    pub(super) attention_output: ArrayCell,
    pub(super) mlp_fused_up: ArrayCell,
    pub(super) mlp_hidden: ArrayCell,
    pub(super) rotated_queries: ArrayCell,
    pub(super) rotated_keys: ArrayCell,
    pub(super) rotated_values: ArrayCell,
    pub(super) attention_partials: ArrayCell,
    pub(super) attention_sums: ArrayCell,
    pub(super) attention_maxs: ArrayCell,
    pub(super) pooling: ArrayCell,
    pub(super) dense: ArrayCell,
    pub(super) norm: ArrayCell,
    pub(super) logits: ArrayCell,
}

impl AuxBuffers {
    pub(super) fn new(
        scratch: &ForwardPassBuffers,
        _decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
        context: &MTLContext,
        num_labels: usize,
    ) -> Self {
        Self {
            suffix_length,
            main: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.main.clone(),
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            shortcut: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.shortcut.clone(),
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            qkv: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.qkv.clone(),
                    &model_shape.qkv_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_output: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_output.clone(),
                    &model_shape.attention_output_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            mlp_fused_up: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.mlp_fused_up.clone(),
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            mlp_hidden: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.mlp_hidden.clone(),
                    &model_shape.mlp_hidden_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            rotated_queries: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.rotated_queries.clone(),
                    &model_shape.rotated_queries_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            rotated_keys: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.rotated_keys.clone(),
                    &model_shape.rotated_keys_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            rotated_values: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.rotated_values.clone(),
                    &model_shape.rotated_values_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_partials: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_partials.clone(),
                    &model_shape.attention_partials_shape(suffix_length),
                    model_shape.activation_data_type(),
                )
            }),
            attention_sums: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_sums.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    DataType::F32,
                )
            }),
            attention_maxs: RefCell::new(unsafe {
                MetalArray::new(
                    scratch.attention_maxs.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    DataType::F32,
                )
            }),
            pooling: {
                let batch_size = 1;
                let model_dim = model_shape.main_shape(1)[1];
                let data_type = model_shape.activation_data_type();
                let buffer_size =
                    (batch_size * model_dim * data_type.size_in_bytes()) as u64;
                let buffer = context.device.new_buffer(
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                RefCell::new(unsafe {
                    MetalArray::new(buffer, &[batch_size, model_dim], data_type)
                })
            },
            dense: {
                let batch_size = 1;
                let model_dim = model_shape.main_shape(1)[1];
                let data_type = model_shape.activation_data_type();
                let buffer_size =
                    (batch_size * model_dim * data_type.size_in_bytes()) as u64;
                let buffer = context.device.new_buffer(
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                RefCell::new(unsafe {
                    MetalArray::new(buffer, &[batch_size, model_dim], data_type)
                })
            },
            norm: {
                let batch_size = 1;
                let model_dim = model_shape.main_shape(1)[1];
                let data_type = model_shape.activation_data_type();
                let buffer_size =
                    (batch_size * model_dim * data_type.size_in_bytes()) as u64;
                let buffer = context.device.new_buffer(
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                RefCell::new(unsafe {
                    MetalArray::new(buffer, &[batch_size, model_dim], data_type)
                })
            },
            logits: {
                let batch_size = 1;
                let data_type = model_shape.activation_data_type();
                let buffer_size =
                    (batch_size * num_labels * data_type.size_in_bytes())
                        as u64;
                let buffer = context.device.new_buffer(
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                RefCell::new(unsafe {
                    MetalArray::new(
                        buffer,
                        &[batch_size, num_labels],
                        data_type,
                    )
                })
            },
        }
    }

    pub(super) fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}
