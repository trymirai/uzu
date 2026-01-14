use std::{cell::RefCell, rc::Rc};

use super::super::ModelShape;
use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray},
    config::EmbeddingConfig,
    parameters::ParameterTree,
};

type ArrayCell = RefCell<MetalArray>;

#[allow(dead_code)]
pub enum EmbeddingsBuffers {
    Tied {
        /// [vocab_size, model_dim]
        weights: ArrayCell,
    },
    Untied {
        /// [vocab_size, model_dim]
        input_weights: ArrayCell,
        /// [vocab_size, model_dim]
        output_weights: ArrayCell,
    },
    QuantizedTied {
        /// [vocab_size, model_dim]
        weights: ArrayCell,
        /// [vocab_size]
        scales: ArrayCell,
    },
    MLXSemiQuantizedUntied {
        /// [vocab_size, model_dim]
        input_weights: ArrayCell,
        /// [vocab_size, model_dim]
        packed_output_weights: ArrayCell,
        /// [vocab_size, num_groups]
        output_scales: ArrayCell,
        /// [vocab_size, num_groups]
        output_biases: ArrayCell,
    },
    MLXQuantizedUntied {
        /// [vocab_size, model_dim/pack]
        packed_input_weights: ArrayCell,
        /// [vocab_size, num_groups]
        input_scales: ArrayCell,
        /// [vocab_size, num_groups]
        input_biases: ArrayCell,
        /// [vocab_size, model_dim/pack]
        packed_output_weights: ArrayCell,
        /// [vocab_size, num_groups]
        output_scales: ArrayCell,
        /// [vocab_size, num_groups]
        output_biases: ArrayCell,
    },
}

impl EmbeddingsBuffers {
    pub fn new(
        context: &MTLContext,
        embeddings_config: &EmbeddingConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let array_label = |config: &EmbeddingConfig, name: &str| -> String {
            let prefix = "embeddings_buffers";
            let config_name = match config {
                EmbeddingConfig::Tied {
                    ..
                } => "tied",
                EmbeddingConfig::Untied {
                    ..
                } => "untied",
                EmbeddingConfig::QuantizedTied {
                    ..
                } => "quantized_tied",
                EmbeddingConfig::MLXQuantizedTied {
                    ..
                } => "mlx_quantized_tied",
                EmbeddingConfig::MLXSemiQuantizedUntied {
                    ..
                } => "mlx_semi_quantized_untied",
                EmbeddingConfig::MLXQuantizedUntied {
                    ..
                } => "mlx_quantized_untied",
            };
            format!("{prefix}_{config_name}_{name}")
        };

        unsafe {
            match embeddings_config {
                EmbeddingConfig::Tied {
                    common: _,
                    precision: _,
                } => Self::Tied {
                    weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                        array_label(embeddings_config, "weights"),
                    )),
                },
                EmbeddingConfig::Untied {
                    common: _,
                    precision: _,
                } => Self::Untied {
                    input_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_input_shape(),
                        model_shape.activation_data_type(),
                        array_label(embeddings_config, "input_weights"),
                    )),
                    output_weights: RefCell::new(context.array_uninitialized(
                        &model_shape.embeddings_output_shape(),
                        model_shape.activation_data_type(),
                        array_label(embeddings_config, "output_weights"),
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
                                array_label(embeddings_config, "weights"),
                            ),
                        ),
                        scales: RefCell::new(context.array_uninitialized(
                            &model_shape.quantized_embeddings_scales_shape(),
                            model_shape.activation_data_type(),
                            array_label(embeddings_config, "scales"),
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
                                array_label(embeddings_config, "weights"),
                            ),
                        ),
                        scales: RefCell::new(context.array_uninitialized(
                            &[vocab_size, num_groups],
                            model_shape.activation_data_type(),
                            array_label(embeddings_config, "scales"),
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
                                array_label(embeddings_config, "input_weights"),
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
                                array_label(
                                    embeddings_config,
                                    "packed_output_weights",
                                ),
                            ),
                        ),
                        output_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "output_scales"),
                            ),
                        ),
                        output_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "output_biases"),
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
                    let pack = embedding_quantization_mode.packing_divisor();
                    Self::MLXQuantizedUntied {
                        packed_input_weights: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, model_dim / pack],
                                embedding_quantization_mode.storage_type(),
                                array_label(
                                    embeddings_config,
                                    "packed_input_weights",
                                ),
                            ),
                        ),
                        input_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "input_scales"),
                            ),
                        ),
                        input_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "input_biases"),
                            ),
                        ),
                        packed_output_weights: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, model_dim / pack],
                                embedding_quantization_mode.storage_type(),
                                array_label(
                                    embeddings_config,
                                    "packed_output_weights",
                                ),
                            ),
                        ),
                        output_scales: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "output_scales"),
                            ),
                        ),
                        output_biases: RefCell::new(
                            context.array_uninitialized(
                                &[vocab_size, num_groups],
                                model_shape.activation_data_type(),
                                array_label(embeddings_config, "output_biases"),
                            ),
                        ),
                    }
                },
            }
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) {
        let embeddings_tree = parameter_tree.subtree("embedding").unwrap();
        match self {
            EmbeddingsBuffers::Tied {
                weights,
            } => {
                let embeddings_view = embeddings_tree.leaf("weights").unwrap();
                weights.borrow_mut().copy_from_array(&embeddings_view);
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
                    buffer.borrow_mut().copy_from_array(&view);
                }
            },
            EmbeddingsBuffers::QuantizedTied {
                weights,
                scales,
            } => {
                let mapping = vec![("weights", weights), ("scales", scales)];
                for (name, buffer) in mapping {
                    let view = embeddings_tree.leaf(name).unwrap();
                    buffer.borrow_mut().copy_from_array(&view);
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
                    buffer.borrow_mut().copy_from_array(&view);
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
                    buffer.borrow_mut().copy_from_array(&view);
                }
            },
        }
    }
}
