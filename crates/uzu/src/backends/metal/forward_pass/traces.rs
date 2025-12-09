//! Unified activation tracing for both LLM and classifier models.
//!
//! This module provides a single tracing implementation that works for any model type,
//! with optional fields for model-specific traces (e.g., embedding_norm for classifiers).

use std::{cell::RefCell, rc::Rc};

use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray, ModelShape},
};

type ArrayCell = RefCell<MetalArray>;

/// Activation trace for a single transformer layer.
///
/// This struct captures intermediate activations at each stage of a transformer layer,
/// useful for debugging and validation against reference implementations.
pub struct LayerActivationTrace {
    pub inputs: ArrayCell,
    pub pre_attention_norm: ArrayCell,
    pub attention: ArrayCell,
    pub post_attention_norm: ArrayCell,
    pub mlp_inputs: ArrayCell,
    pub pre_mlp_norm: ArrayCell,
    pub mlp: ArrayCell,
    pub post_mlp_norm: ArrayCell,
    pub outputs: ArrayCell,
}

impl LayerActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        unsafe {
            Self {
                inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                attention: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp_inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                outputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}

/// Unified activation trace for any model type.
///
/// This struct captures all intermediate activations during a forward pass.
/// Some fields are optional and only populated for specific model types:
/// - `embedding_norm`: Used by classifiers after embedding layer
/// - `output_pooling`: Used by classifiers for pooled output
pub struct ActivationTrace {
    /// Post-embedding normalization (classifier-specific).
    pub embedding_norm: Option<ArrayCell>,
    /// Per-layer activation traces.
    pub layer_results: Vec<Rc<RefCell<LayerActivationTrace>>>,
    /// Output normalization activations.
    pub output_norm: ArrayCell,
    /// Pooled output (classifier-specific).
    pub output_pooling: Option<ArrayCell>,
    /// Final logits.
    pub logits: ArrayCell,
}

impl ActivationTrace {
    /// Create a new activation trace for LLM (decoder) models.
    pub fn new_llm(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(LayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        unsafe {
            Self {
                embedding_norm: None,
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                output_pooling: None,
                logits: RefCell::new(context.array_uninitialized(
                    &model_shape.logits_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }

    /// Create a new activation trace for classifier models.
    pub fn new_classifier(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(LayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        let model_dim = model_shape.main_shape(1)[1];
        unsafe {
            Self {
                embedding_norm: Some(RefCell::new(
                    context.array_uninitialized(
                        &model_shape.main_shape(suffix_length),
                        model_shape.activation_data_type(),
                    ),
                )),
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                output_pooling: Some(RefCell::new(
                    context.array_uninitialized(
                        &[1, model_dim],
                        model_shape.activation_data_type(),
                    ),
                )),
                logits: RefCell::new(context.array_uninitialized(
                    &[1, num_labels],
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}

impl ActivationTrace {
    /// Get embedding_norm trace (classifier-specific).
    /// Panics if called on LLM trace.
    pub fn embedding_norm(&self) -> &ArrayCell {
        self.embedding_norm
            .as_ref()
            .expect("embedding_norm is only available for classifier traces")
    }

    /// Get output_pooling trace (classifier-specific).
    /// Panics if called on LLM trace.
    pub fn output_pooling(&self) -> &ArrayCell {
        self.output_pooling
            .as_ref()
            .expect("output_pooling is only available for classifier traces")
    }
}
