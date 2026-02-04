use crate::backends::metal::Metal;
use std::{cell::RefCell, fs::File, io::BufReader, path::Path, rc::Rc};

use super::{
    KernelDataType, MTLCommandBuffer, MTLCommandQueue, MTLContext, ModelShape,
    ProtocolObject, Retained,
    compilation_parameters::CompilationConfig,
    encodable_block::{
        Activation, ClassifierLayer, ClassifierPredictionHead, Normalization,
        Pooling, Rope,
        transformer_layer::{embed_block, linear_block},
    },
    forward_pass::{
        ArrayId, EncodableBlock, RopeType, ScratchBuffers, SharedBuffers,
    },
    kernel::dsl::SigmoidMetalKernel,
};
use crate::{
    DataType,
    backends::{
        common::Context, common::kernel::SigmoidKernel,
        metal::error::ClassifierError,
    },
    config::{ClassifierModelConfig, ModelMetadata},
    parameters::ParameterLoader,
    session::types::Error,
};

pub struct ClassifierContext {
    pub mtl_context: Rc<MTLContext>,
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,

    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    pub scratch_buffers: ScratchBuffers<Rc<MTLContext>>,

    pub model_config: ClassifierModelConfig,
    pub model_shape: ModelShape,

    pub embed: Box<dyn EncodableBlock<Metal>>,
    pub embedding_norm: Normalization,
    pub layers: Box<[ClassifierLayer]>,
    pub output_norm: Normalization,
    pub global_rope: Rc<Box<dyn EncodableBlock<Metal>>>,
    pub local_rope: Option<Rc<Box<dyn EncodableBlock<Metal>>>>,

    pub pooling: Box<dyn EncodableBlock<Metal>>,
    pub prediction_head: Box<dyn EncodableBlock<Metal>>,

    pub sigmoid_kernel: SigmoidMetalKernel,
}

impl ClassifierContext {
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        let context =
            MTLContext::new().map_err(|_| Error::UnableToCreateMetalContext)?;

        let command_buffer = context
            .create_command_buffer()
            .map_err(|_| Error::UnableToCreateMetalContext)?;

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file =
            File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(BufReader::new(config_file))
                .map_err(|_| Error::UnableToLoadConfig)?;

        let classifier_model_config = model_metadata
            .model_config
            .as_classifier()
            .ok_or(Error::UnableToLoadConfig)?;

        let decoder_config = Rc::new(
            classifier_model_config
                .model_config
                .to_decoder_config()
                .map_err(|_| Error::UnableToLoadConfig)?,
        );
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let compilation_config = Rc::new(CompilationConfig::default());
        let weights_path = model_path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }
        let weights_file = File::open(&weights_path)
            .map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, &context)
            .map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(
            &context,
            &decoder_config,
            &model_shape,
        )));
        let transformer_tree =
            root_loader_view.subtree("transformer").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "transformer".to_string(),
                ))
            })?;

        {
            let mut shared_bufs = shared_buffers.borrow_mut();
            if let Some(global_rope) = &mut shared_bufs.global_rope {
                global_rope.update_data(
                    &transformer_tree,
                    String::from("global_rope"),
                );
            }
            if let Some(local_rope) = &mut shared_bufs.local_rope {
                local_rope
                    .update_data(&transformer_tree, String::from("local_rope"));
            }
        }

        let data_type = decoder_config
            .layer_config
            .mixer_config
            .as_attention()
            .ok_or(Error::Classifier(ClassifierError::NonAttentionMixer))?
            .qkv_projection_config
            .activation_precision()
            .into();

        let embed = embed_block(&decoder_config, &context, &root_loader_view);

        let global_rope =
            Self::create_rope_block(&context, data_type, RopeType::Global)
                .map_err(Error::Classifier)?;
        let local_rope = classifier_model_config
            .model_config
            .transformer_config
            .local_rope_config
            .as_ref()
            .map(|_| {
                Self::create_rope_block(&context, data_type, RopeType::Local)
            })
            .transpose()
            .map_err(Error::Classifier)?;

        let layers = classifier_model_config
            .model_config
            .transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let mut rope = global_rope.clone();
                let attn = layer_config
                    .attention_config()
                    .ok_or(ClassifierError::NonAttentionMixer)?;

                if attn.sliding_window_size.is_some() {
                    if let Some(local_rope_block) = local_rope.clone() {
                        rope = local_rope_block;
                    }
                }

                let num_heads = attn.num_heads.ok_or_else(|| {
                    ClassifierError::MissingConfigField(format!(
                        "num_heads in layer {}",
                        layer_index
                    ))
                })?;
                let head_dim = attn.head_dim.ok_or_else(|| {
                    ClassifierError::MissingConfigField(format!(
                        "head_dim in layer {}",
                        layer_index
                    ))
                })?;
                let num_groups = attn.num_groups.ok_or_else(|| {
                    ClassifierError::MissingConfigField(format!(
                        "num_groups in layer {}",
                        layer_index
                    ))
                })?;

                let layer_tree = transformer_tree
                    .subtree(&format!("layers.{}", layer_index))
                    .map_err(|_| {
                        ClassifierError::WeightSubtreeNotFound(format!(
                            "layers.{}",
                            layer_index
                        ))
                    })?;

                Ok(ClassifierLayer::new(
                    context.clone(),
                    layer_config,
                    compilation_config.clone(),
                    layer_index,
                    classifier_model_config.model_config.model_dim,
                    classifier_model_config
                        .model_config
                        .transformer_config
                        .hidden_dim,
                    num_heads,
                    head_dim,
                    num_groups,
                    attn.scale,
                    &layer_tree,
                    rope,
                ))
            })
            .collect::<Result<Vec<_>, ClassifierError>>()
            .map_err(Error::Classifier)?
            .into_boxed_slice();

        let output_norm_tree =
            transformer_tree.subtree("output_norm").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "output_norm".to_string(),
                ))
            })?;
        let output_norm = Normalization::new(
            &context,
            data_type,
            classifier_model_config
                .model_config
                .transformer_config
                .output_norm_config
                .clone(),
            ArrayId::Main,
            ArrayId::Main,
            &output_norm_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!(
                "output norm: {:?}",
                e
            )))
        })?;

        let context_length =
            classifier_model_config.model_config.context_length;
        let scratch_buffers = ScratchBuffers::new(
            &context,
            &decoder_config,
            &model_shape,
            context_length,
            context_length,
        );

        let embedding_norm_tree =
            root_loader_view.subtree("embedding_norm").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "embedding_norm".to_string(),
                ))
            })?;
        let embedding_norm = Normalization::new(
            &context,
            data_type,
            classifier_model_config.model_config.embedding_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &embedding_norm_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!(
                "embedding norm: {:?}",
                e
            )))
        })?;

        let model_dim = classifier_model_config.model_config.model_dim;
        let num_labels = classifier_model_config.model_config.num_labels;
        let prediction_head_config =
            &classifier_model_config.model_config.prediction_head_config;
        let prediction_head_tree =
            root_loader_view.subtree("prediction_head").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "prediction_head".to_string(),
                ))
            })?;

        let prediction_head_data_type: DataType =
            prediction_head_config.dense_config.activation_precision().into();

        let prediction_head_dense_tree =
            prediction_head_tree.subtree("dense").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "prediction_head.dense".to_string(),
                ))
            })?;
        let prediction_head_dense = linear_block::<1>(
            &prediction_head_config.dense_config,
            prediction_head_config.use_dense_bias,
            model_dim,
            [model_dim],
            &context,
            &prediction_head_dense_tree,
            ArrayId::ClassifierPooling,
            ArrayId::ClassifierPredictionHeadDense,
        );

        let prediction_head_activation = Box::new(
            Activation::new(
                &context,
                prediction_head_data_type,
                prediction_head_config.activation.clone(),
                ArrayId::ClassifierPredictionHeadDense,
                ArrayId::ClassifierPredictionHeadDense,
            )
            .map_err(|e| {
                Error::Classifier(ClassifierError::KernelCreationFailed(
                    format!("prediction head activation: {:?}", e),
                ))
            })?,
        );

        let prediction_head_norm_tree =
            prediction_head_tree.subtree("norm").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "prediction_head.norm".to_string(),
                ))
            })?;
        let prediction_head_norm = Normalization::new(
            &context,
            prediction_head_data_type,
            prediction_head_config.normalization_config.clone(),
            ArrayId::ClassifierPredictionHeadDense,
            ArrayId::ClassifierPredictionHeadNorm,
            &prediction_head_norm_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!(
                "prediction head norm: {:?}",
                e
            )))
        })?;

        let prediction_head_readout_tree =
            prediction_head_tree.subtree("readout").map_err(|_| {
                Error::Classifier(ClassifierError::WeightSubtreeNotFound(
                    "prediction_head.readout".to_string(),
                ))
            })?;
        let prediction_head_final_linear = linear_block::<1>(
            &prediction_head_config.readout_config,
            true,
            model_dim,
            [num_labels],
            &context,
            &prediction_head_readout_tree,
            ArrayId::ClassifierPredictionHeadNorm,
            ArrayId::ClassifierPredictionHeadLogits,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!(
                "prediction head readout: {:?}",
                e
            )))
        })?;

        let sigmoid_kernel =
            SigmoidMetalKernel::new(&context, data_type.into()).map_err(
                |e| {
                    eprintln!("Failed to create sigmoid kernel: {:?}", e);
                    Error::UnableToCreateMetalContext
                },
            )?;

        let pooling = Box::new(
            Pooling::new(
                &context,
                data_type.into(),
                classifier_model_config.model_config.classifier_pooling.clone(),
                model_dim,
            )
            .map_err(|e| {
                eprintln!("Failed to create pooling: {:?}", e);
                Error::UnableToCreateMetalContext
            })?,
        );

        let prediction_head = Box::new(ClassifierPredictionHead::new(
            prediction_head_dense.map_err(|e| {
                Error::Classifier(ClassifierError::KernelCreationFailed(
                    format!("prediction head dense: {:?}", e),
                ))
            })?,
            prediction_head_activation,
            Box::new(prediction_head_norm),
            prediction_head_final_linear,
            num_labels,
        ));

        Ok(Self {
            mtl_context: context,
            command_buffer,
            shared_buffers,
            scratch_buffers,
            model_config: classifier_model_config.clone(),
            model_shape,
            embed,
            embedding_norm,
            layers,
            output_norm,
            global_rope,
            local_rope,
            pooling,
            prediction_head,
            sigmoid_kernel,
        })
    }

    fn create_rope_block(
        mtl_context: &MTLContext,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Result<Rc<Box<dyn EncodableBlock<Metal>>>, ClassifierError> {
        let kernel_data_type: KernelDataType = data_type.into();

        let rotation: Box<dyn EncodableBlock<Metal>> = Box::new(
            Rope::new(mtl_context, kernel_data_type, rope_type).map_err(
                |e| {
                    ClassifierError::KernelCreationFailed(format!(
                        "RoPE: {:?}",
                        e
                    ))
                },
            )?,
        );
        Ok(Rc::new(rotation))
    }

    pub fn reset_command_buffer(&mut self) {
        self.command_buffer = self
            .mtl_context
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer")
            .to_owned();
    }
}
