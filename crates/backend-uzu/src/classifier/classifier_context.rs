use std::{fs::File, path::Path, rc::Rc};

use crate::{
    backends::common::{Backend, Context},
    classifier::ClassifierError,
    config::{decoder::DecoderConfig, model::classifier_model::ClassifierModelConfig, rope::AnyRoPEConfig},
    data_type::DataType,
    encodable_block::{
        ClassifierLayer, ClassifierPredictionHead, Embedding, LayerRopeKind, Linear, Normalization, Pooling, QkUnpack,
        Rope,
    },
    forward_pass::model_shape::ModelShape,
    parameters::ParameterLoader,
    session::types::Error,
};

pub struct ClassifierContext<B: Backend> {
    pub context: Rc<B::Context>,

    pub model_config: ClassifierModelConfig,
    pub rope_configs: Box<[AnyRoPEConfig]>,
    #[cfg(all(test, feature = "tracing"))]
    pub model_shape: ModelShape,

    pub embed: Embedding<B>,
    pub embedding_norm: Normalization<B>,
    pub layers: Box<[ClassifierLayer<B>]>,
    pub output_norm: Normalization<B>,

    pub pooling: Pooling<B>,
    pub prediction_head: ClassifierPredictionHead<B>,
    pub logits_data_type: DataType,
}

impl<B: Backend> ClassifierContext<B> {
    pub fn new(
        model_path: &Path,
        model_config: &ClassifierModelConfig,
    ) -> Result<Self, Error> {
        let context = B::Context::new().map_err(|e| Error::UnableToCreateContext(e.into()))?;
        let classifier_config = &model_config.classifier_config;

        let decoder_config = Rc::new(DecoderConfig {
            embedding_config: classifier_config.embedding_config.clone(),
            transformer_config: classifier_config.transformer_config.clone(),
            vocab_size: classifier_config.vocab_size,
            ple_model_config: None,
        });

        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let root_loader_view =
            loader.tree().subtree("classifier").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        let model_shape = ModelShape::from_decoder_config(&decoder_config, DataType::BF16);

        let transformer_tree =
            root_loader_view.subtree("transformer").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let mut rope_configs = Vec::<(AnyRoPEConfig, usize)>::new();
        let layer_rope_kinds: Box<[LayerRopeKind]> = classifier_config
            .transformer_config
            .layer_configs
            .iter()
            .map(|layer_config| {
                let Some(rope_config) = &layer_config.rope_config else {
                    return LayerRopeKind::NoKernel;
                };
                let head_dim = *rope_config.head_dim();
                let index = rope_configs
                    .iter()
                    .position(|(existing_config, existing_head_dim)| {
                        existing_config == rope_config && *existing_head_dim == head_dim
                    })
                    .unwrap_or_else(|| {
                        rope_configs.push((rope_config.clone(), head_dim));
                        rope_configs.len() - 1
                    });
                LayerRopeKind::Indexed(index)
            })
            .collect();

        let output_norm_tree =
            transformer_tree.subtree("output_norm").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        let (embed, readout_input_hadamard_factors) = Embedding::new(
            context.as_ref(),
            decoder_config.vocab_size as u32,
            decoder_config.transformer_config.model_dim as u32,
            &decoder_config.embedding_config,
            &root_loader_view.subtree("embedding").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?,
            &model_shape,
        )
        .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        if readout_input_hadamard_factors.is_some() {
            return Err(Error::Classifier(ClassifierError::Custom(
                "classifier does not support embedding readout hadamard".to_string(),
            )));
        }

        let rope = Rc::new(
            Rope::<B>::new(context.as_ref(), &model_shape, false)
                .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("RoPE: {:?}", e))))?,
        );
        let qk_unpack = Rc::new(
            QkUnpack::<B>::new(context.as_ref(), model_shape.data_type)
                .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("QkUnpack: {:?}", e))))?,
        );

        let layers = classifier_config
            .transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let layer_tree = transformer_tree
                    .subtree(&format!("layers.{}", layer_index))
                    .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

                ClassifierLayer::new(
                    context.as_ref(),
                    &classifier_config.transformer_config,
                    layer_config,
                    layer_index,
                    layer_rope_kinds[layer_index],
                    &layer_tree,
                    rope.clone(),
                    qk_unpack.clone(),
                    model_shape.data_type,
                )
                .map_err(|error| Error::UnableToCreateClassifierLayer(Box::new(error)))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .into_boxed_slice();

        let output_norm = Normalization::new(
            context.as_ref(),
            model_shape.data_type,
            classifier_config.transformer_config.model_dim,
            classifier_config.transformer_config.output_norm_config.clone(),
            &output_norm_tree,
        )
        .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("output norm: {:?}", e))))?;

        let embedding_norm_tree =
            root_loader_view.subtree("embedding_norm").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let embedding_norm = Normalization::new(
            context.as_ref(),
            model_shape.data_type,
            classifier_config.transformer_config.model_dim,
            classifier_config.embedding_norm_config.clone(),
            &embedding_norm_tree,
        )
        .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("embedding norm: {:?}", e))))?;

        let model_dim = classifier_config.model_dim;
        let num_labels = classifier_config.num_labels;
        let prediction_head_config = &classifier_config.prediction_head_config;
        let prediction_head_tree =
            root_loader_view.subtree("prediction_head").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        let prediction_head_dense_tree =
            prediction_head_tree.subtree("dense").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let prediction_head_dense = <dyn Linear<B>>::new::<1>(
            model_dim,
            [model_dim],
            prediction_head_config.use_dense_bias,
            context.as_ref(),
            model_shape.data_type,
            &prediction_head_dense_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head dense: {:?}", e)))
        })?;

        let prediction_head_norm_tree =
            prediction_head_tree.subtree("norm").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let prediction_head_norm = Normalization::new(
            context.as_ref(),
            model_shape.data_type,
            model_dim,
            prediction_head_config.normalization_config.clone(),
            &prediction_head_norm_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head norm: {:?}", e)))
        })?;

        let prediction_head_readout_tree =
            prediction_head_tree.subtree("readout").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let prediction_head_final_linear = <dyn Linear<B>>::new::<1>(
            model_dim,
            [num_labels],
            true,
            context.as_ref(),
            model_shape.data_type,
            &prediction_head_readout_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head readout: {:?}", e)))
        })?;

        let pooling = Pooling::<B>::new(
            context.as_ref(),
            model_shape.data_type,
            classifier_config.classifier_pooling.clone(),
            model_dim,
        )
        .map_err(|e| Error::UnableToCreateContext(e.into()))?;

        let prediction_head = ClassifierPredictionHead::new(
            context.as_ref(),
            prediction_head_dense,
            prediction_head_config.activation.clone(),
            model_shape.data_type,
            prediction_head_norm,
            prediction_head_final_linear,
            model_dim,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head activation: {:?}", e)))
        })?;

        loader.tree().assert_all_tensors_validated().map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        Ok(Self {
            context,
            model_config: model_config.clone(),
            rope_configs: rope_configs.into_iter().map(|(config, _)| config).collect(),
            logits_data_type: model_shape.data_type,
            #[cfg(all(test, feature = "tracing"))]
            model_shape,
            embed,
            embedding_norm,
            layers,
            output_norm,
            pooling,
            prediction_head,
        })
    }
}
