use std::{cell::RefCell, rc::Rc};

use super::super::ModelShape;
use crate::{
    Array, DeviceContext, config::EmbeddingConfig, parameters::ParameterTree,
};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

#[allow(dead_code)]
pub enum EmbeddingsBuffers<C: DeviceContext> {
    Tied {
        /// [vocab_size, model_dim]
        weights: ArrayCell<C>,
    },
    Untied {
        /// [vocab_size, model_dim]
        input_weights: ArrayCell<C>,
        /// [vocab_size, model_dim]
        output_weights: ArrayCell<C>,
    },
    QuantizedTied {
        /// [vocab_size, model_dim]
        weights: ArrayCell<C>,
        /// [vocab_size]
        scales: ArrayCell<C>,
    },
    MLXSemiQuantizedUntied {
        /// [vocab_size, model_dim]
        input_weights: ArrayCell<C>,
        /// [vocab_size, model_dim]
        packed_output_weights: ArrayCell<C>,
        /// [vocab_size, num_groups]
        output_scales: ArrayCell<C>,
        /// [vocab_size, num_groups]
        output_biases: ArrayCell<C>,
    },
    MLXQuantizedUntied {
        /// [vocab_size, model_dim]
        packed_input_weights: ArrayCell<C>,
        /// [vocab_size, num_groups]
        input_scales: ArrayCell<C>,
        /// [vocab_size, num_groups]
        input_biases: ArrayCell<C>,
        /// [vocab_size, model_dim]
        packed_output_weights: ArrayCell<C>,
        /// [vocab_size, num_groups]
        output_scales: ArrayCell<C>,
        /// [vocab_size, num_groups]
        output_biases: ArrayCell<C>,
    },
}

impl<C: DeviceContext> EmbeddingsBuffers<C> {
    pub fn new(
        context: &C,
        embeddings_config: &EmbeddingConfig,
        model_shape: &ModelShape,
    ) -> Self {
        unsafe {
            match embeddings_config {
                EmbeddingConfig::Tied {
                    common: _,
                    precision: _,
                } => Self::Tied {
                    weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                EmbeddingConfig::Untied {
                    common: _,
                    precision: _,
                } => Self::Untied {
                    input_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                    )),
                    output_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_output_shape(),
                        model_shape.activation_data_type(),
                    )),
                },
                EmbeddingConfig::QuantizedTied {
                    embedding_quantization_mode,
                    ..
                } => {
                    let [vocab_size, model_dim] =
                        model_shape.quantized_embeddings_weights_shape();
                    Self::QuantizedTied {
                        weights: RefCell::new(
                            context.array_uninitialized(
                                &[
                                    vocab_size,
                                    model_dim
                                        / embedding_quantization_mode
                                            .packing_divisor(),
                                ],
                                embedding_quantization_mode.storage_type(),
                            ),
                        ),
                        scales: RefCell::new(context.array_uninitialized(
                            &model_shape.quantized_embeddings_scales_shape(),
                            model_shape.activation_data_type(),
                        )),
                    }
                },
                EmbeddingConfig::MLXQuantizedTied {
                    group_size,
                    embedding_quantization_mode,
                    ..
                } => {
                    let [vocab_size, model_dim] =
                        model_shape.quantized_embeddings_weights_shape();
                    let num_groups = model_dim / group_size;
                    Self::QuantizedTied {
                        weights: RefCell::new(
                            context.array_uninitialized(
                                &[
                                    vocab_size,
                                    model_dim
                                        / embedding_quantization_mode
                                            .packing_divisor(),
                                ],
                                embedding_quantization_mode.storage_type(),
                            ),
                        ),
                        scales: RefCell::new(context.array_uninitialized(
                            &[vocab_size, num_groups],
                            model_shape.activation_data_type(),
                        )),
                    }
                },
                EmbeddingConfig::MLXSemiQuantizedUntied {
                    group_size,
                    embedding_quantization_mode,
                    ..
                } => {
                    let [vocab_size, model_dim] =
                        model_shape.quantized_embeddings_weights_shape();
                    let num_groups = model_dim / group_size;
                    Self::MLXSemiQuantizedUntied {
                        input_weights: RefCell::new(
                            context.array_uninitialized(
                                &model_shape.embeddings_input_shape(),
                                model_shape.activation_data_type(),
                            ),
                        ),
                        packed_output_weights: RefCell::new(
                            context.array_uninitialized(
                                &[
                                    vocab_size,
                                    model_dim
                                        / embedding_quantization_mode
                                            .packing_divisor(),
                                ],
                                embedding_quantization_mode.storage_type(),
                            ),
                        ),
                        output_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                        output_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                    }
                },
                EmbeddingConfig::MLXQuantizedUntied {
                    group_size,
                    embedding_quantization_mode,
                    ..
                } => {
                    let [vocab_size, model_dim] =
                        model_shape.quantized_embeddings_weights_shape();
                    let num_groups = model_dim / group_size;
                    Self::MLXQuantizedUntied {
                        packed_input_weights: RefCell::new(
                            context.array_uninitialized(
                                &[
                                    vocab_size,
                                    model_dim
                                        / embedding_quantization_mode
                                            .packing_divisor(),
                                ],
                                embedding_quantization_mode.storage_type(),
                            ),
                        ),
                        input_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                        input_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                        packed_output_weights: RefCell::new(
                            context.array_uninitialized(
                                &[
                                    vocab_size,
                                    model_dim
                                        / embedding_quantization_mode
                                            .packing_divisor(),
                                ],
                                embedding_quantization_mode.storage_type(),
                            ),
                        ),
                        output_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                        output_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                            ),
                        ),
                    }
                },
            }
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<C>>,
    ) {
        let embeddings_tree = parameter_tree.subtree("embedding").unwrap();
        match self {
            EmbeddingsBuffers::Tied {
                weights,
            } => {
                let embeddings_view = embeddings_tree.leaf("weights").unwrap();
                weights.borrow_mut().copy_from(&embeddings_view);
            },
            EmbeddingsBuffers::Untied {
                input_weights,
                output_weights,
            } => {
                let mapping = vec![
                    ("input_weights", input_weights),
                    ("output_weights", output_weights),
                ];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from(&view);
                }
            },
            EmbeddingsBuffers::QuantizedTied {
                weights,
                scales,
            } => {
                let mapping = vec![("weights", weights), ("scales", scales)];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from(&view);
                }
            },
            EmbeddingsBuffers::MLXSemiQuantizedUntied {
                input_weights,
                packed_output_weights,
                output_scales,
                output_biases,
            } => {
                let mapping = vec![
                    ("input_weights", input_weights),
                    ("output_weights", packed_output_weights),
                    ("output_scales", output_scales),
                    ("output_biases", output_biases),
                ];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from(&view);
                }
            },
            EmbeddingsBuffers::MLXQuantizedUntied {
                packed_input_weights,
                input_scales,
                input_biases,
                packed_output_weights,
                output_scales,
                output_biases,
            } => {
                let mapping = vec![
                    ("input_weights", packed_input_weights),
                    ("input_scales", input_scales),
                    ("input_biases", input_biases),
                    ("output_weights", packed_output_weights),
                    ("output_scales", output_scales),
                    ("output_biases", output_biases),
                ];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from(&view);
                }
            },
        }
    }
}
