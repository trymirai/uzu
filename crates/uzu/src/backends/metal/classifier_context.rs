use std::{
    cell::RefCell, collections::HashMap, fs::File, io::BufReader, path::Path,
    rc::Rc,
};

use mpsgraph::{CommandBuffer as MPSCommandBuffer, Device as MPSDevice, Graph};
use objc2::rc::{Retained, autoreleasepool};

use super::{
    KernelDataType, MTLContext, ModelShape,
    compilation_parameters::CompilationConfig,
    encodable_block::{
        ClassifierLayer, ClassifierPredictionHead, Normalization, Pooling,
        Rope,
        transformer_layer::{embed_block, linear_block},
    },
    forward_pass::{
        ArrayId, EncodableBlock, IOArrays, MPSGraphBlock, RopeType,
        ScratchBuffers, SharedBuffers,
    },
    graph::{common::activation, placeholder, shaped_type},
    kernel::{PoolingKernel, SigmoidKernel},
};
use crate::{
    DataType,
    config::{ClassifierModelConfig, ModelMetadata},
    parameters::ParameterLoader,
    session::types::Error,
};

pub struct ClassifierContext {
    pub mtl_context: Rc<MTLContext>,
    pub command_buffer: Retained<MPSCommandBuffer>,

    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    pub scratch_buffers: ScratchBuffers,

    pub model_config: ClassifierModelConfig,
    pub model_shape: ModelShape,

    pub embed: Box<dyn EncodableBlock>,
    pub embedding_norm: Normalization,
    pub layers: Box<[ClassifierLayer]>,
    pub output_norm: Normalization,
    pub global_rope: Rc<Box<dyn EncodableBlock>>,
    pub local_rope: Option<Rc<Box<dyn EncodableBlock>>>,

    pub pooling: Box<dyn EncodableBlock>,
    pub prediction_head: Box<dyn EncodableBlock>,

    pub sigmoid_kernel: SigmoidKernel,
}

impl ClassifierContext {
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        let mtl_device = metal::Device::system_default()
            .ok_or(Error::UnableToCreateMetalContext)?;
        let mtl_command_queue =
            mtl_device.new_command_queue_with_max_command_buffer_count(1024);

        let command_buffer =
            MPSCommandBuffer::from_command_queue(&mtl_command_queue);

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

        let decoder_config =
            Rc::new(classifier_model_config.model_config.to_decoder_config());
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let mtl_context = Rc::new(
            MTLContext::new(mtl_device, mtl_command_queue)
                .map_err(|_| Error::UnableToCreateMetalContext)?,
        );

        let compilation_config = Rc::new(CompilationConfig::default());
        let weights_path = model_path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }
        let weights_file = File::open(&weights_path)
            .map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, &mtl_context)
            .map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
        )));
        {
            let mut shared_bufs = shared_buffers.borrow_mut();
            shared_bufs.embeddings.update_data(&root_loader_view);
            let transformer_tree =
                root_loader_view.subtree("transformer").unwrap();
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
            .expect("Classifier only supports Attention layers")
            .qkv_projection_config
            .activation_precision()
            .into();

        let transformer_loader =
            root_loader_view.subtree("transformer").unwrap();

        let embed = embed_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
            &root_loader_view,
        );

        let global_rope =
            Self::create_rope_block(&mtl_context, data_type, RopeType::Global);
        let local_rope = classifier_model_config
            .model_config
            .transformer_config
            .local_rope_config
            .as_ref()
            .map(|_| {
                Self::create_rope_block(
                    &mtl_context,
                    data_type,
                    RopeType::Local,
                )
            });

        let layers = classifier_model_config
            .model_config
            .transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let mut rope = global_rope.clone();
                let mixer = &layer_config.mixer_config;

                if mixer.sliding_window_size().is_some() {
                    if let Some(local_rope_block) = local_rope.clone() {
                        rope = local_rope_block;
                    }
                }

                ClassifierLayer::new(
                    &mtl_context,
                    layer_config,
                    compilation_config.clone(),
                    layer_index,
                    classifier_model_config.model_config.model_dim,
                    classifier_model_config
                        .model_config
                        .transformer_config
                        .hidden_dim,
                    mixer.num_heads().unwrap_or(12),
                    mixer.head_dim().unwrap_or(64),
                    mixer.num_groups().unwrap_or(12),
                    mixer.attention_scale(),
                    &transformer_loader
                        .subtree(&format!("layers.{}", layer_index))
                        .unwrap(),
                    rope,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let output_norm = Normalization::new(
            &mtl_context,
            data_type,
            classifier_model_config
                .model_config
                .transformer_config
                .output_norm_config
                .clone(),
            ArrayId::Main,
            ArrayId::Main,
            &transformer_loader.subtree("output_norm").unwrap(),
        )
        .expect("Failed to create output norm kernel");

        let context_length =
            classifier_model_config.model_config.context_length;
        let scratch_buffers = ScratchBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
            context_length,
            context_length,
        );

        let embedding_norm = Normalization::new(
            &mtl_context,
            data_type,
            classifier_model_config.model_config.embedding_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &root_loader_view.subtree("embedding_norm").unwrap(),
        )
        .expect("Failed to create embedding norm kernel");

        let model_dim = classifier_model_config.model_config.model_dim;
        let num_labels = classifier_model_config.model_config.num_labels;
        let prediction_head_config =
            &classifier_model_config.model_config.prediction_head_config;
        let prediction_head_tree =
            root_loader_view.subtree("prediction_head").unwrap();

        let prediction_head_data_type: DataType =
            prediction_head_config.dense_config.activation_precision().into();

        let prediction_head_dense = linear_block::<1>(
            &prediction_head_config.dense_config,
            prediction_head_config.use_dense_bias,
            model_dim,
            [model_dim],
            &mtl_context,
            &prediction_head_tree.subtree("dense").unwrap(),
            ArrayId::ClassifierPooling,
            ArrayId::ClassifierPredictionHeadDense,
            &compilation_config.descriptor_general,
        );

        let prediction_head_activation = autoreleasepool(|_| {
            let graph = Graph::new();
            let input_shape = [-1, model_dim as isize];
            let input_placeholder =
                placeholder(&graph, &input_shape, prediction_head_data_type);

            let output = activation(
                &graph,
                &prediction_head_config.activation,
                &input_placeholder,
                prediction_head_data_type,
            );

            let shaped_type_obj =
                shaped_type(&input_shape, prediction_head_data_type);
            let feeds =
                HashMap::from([(&*input_placeholder, &*shaped_type_obj)]);

            let executable = graph.compile(
                &MPSDevice::with_device(&mtl_context.device),
                &feeds,
                &[&output],
                None,
                Some(&compilation_config.descriptor_general),
            );

            let arguments = IOArrays::new(
                vec![ArrayId::ClassifierPredictionHeadDense].into_boxed_slice(),
                vec![ArrayId::ClassifierPredictionHeadDense].into_boxed_slice(),
            );

            let execution_descriptor =
                mpsgraph::ExecutableExecutionDescriptor::new();
            execution_descriptor.set_enable_commit_and_continue(true);

            Box::new(MPSGraphBlock::new(
                executable,
                execution_descriptor,
                arguments,
            )) as Box<dyn EncodableBlock>
        });

        let prediction_head_norm = Normalization::new(
            &mtl_context,
            prediction_head_data_type,
            prediction_head_config.normalization_config.clone(),
            ArrayId::ClassifierPredictionHeadDense,
            ArrayId::ClassifierPredictionHeadNorm,
            &prediction_head_tree.subtree("norm").unwrap(),
        )
        .expect("Failed to create prediction head norm kernel");

        let prediction_head_final_linear = linear_block::<1>(
            &prediction_head_config.readout_config,
            true,
            model_dim,
            [num_labels],
            &mtl_context,
            &prediction_head_tree.subtree("readout").unwrap(),
            ArrayId::ClassifierPredictionHeadNorm,
            ArrayId::ClassifierPredictionHeadLogits,
            &compilation_config.descriptor_general,
        );

        let pooling_kernel = PoolingKernel::new(&mtl_context, data_type)
            .map_err(|e| {
                eprintln!("Failed to create pooling kernel: {:?}", e);
                Error::UnableToCreateMetalContext
            })?;
        let sigmoid_kernel = SigmoidKernel::new(&mtl_context, data_type)
            .map_err(|e| {
                eprintln!("Failed to create sigmoid kernel: {:?}", e);
                Error::UnableToCreateMetalContext
            })?;

        let pooling = Box::new(Pooling::new(
            pooling_kernel,
            classifier_model_config.model_config.classifier_pooling.clone(),
            model_dim,
        ));

        let prediction_head = Box::new(ClassifierPredictionHead::new(
            prediction_head_dense,
            prediction_head_activation,
            Box::new(prediction_head_norm),
            prediction_head_final_linear,
            num_labels,
        ));

        Ok(Self {
            mtl_context,
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
    ) -> Rc<Box<dyn EncodableBlock>> {
        let kernel_data_type: KernelDataType = data_type.into();

        let rotation: Box<dyn EncodableBlock> = Box::new(
            Rope::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create Rope"),
        );
        Rc::new(rotation)
    }

    pub fn reset_command_buffer(&mut self) {
        self.command_buffer = MPSCommandBuffer::from_command_queue(
            &self.mtl_context.command_queue,
        );
    }
}
