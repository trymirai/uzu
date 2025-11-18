use std::{cell::RefCell, fs::File, io::BufReader, path::Path, rc::Rc};

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::Retained;

use crate::{
    DataType,
    backends::metal::{
        KVCache, MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ForwardPassBuffers, RopeType, SharedBuffers,
            encodable_with_state::EncodableWithState,
            transformer_layer::embed_block,
        },
        kernel::{
            NormalizationEncodable, PoolingKernel, RopeKernelEncodable,
            SigmoidKernel,
        },
    },
    classifier::ClassifierLayerExecutable,
    config::{ClassifierModelConfig, ModelMetadata},
    parameters::ParameterLoader,
    session::types::Error,
};

pub struct ClassifierContext {
    pub mtl_context: Rc<MTLContext>,
    pub command_buffer: Retained<MPSCommandBuffer>,

    pub kv_cache: Rc<RefCell<KVCache>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    pub scratch_buffers: ForwardPassBuffers,

    pub model_config: ClassifierModelConfig,
    pub model_shape: ModelShape,

    pub embed: Box<dyn EncodableWithState>,
    pub embedding_norm: NormalizationEncodable,
    pub layers: Box<[ClassifierLayerExecutable]>,
    pub output_norm: NormalizationEncodable,
    pub global_rope: Rc<Box<dyn EncodableWithState>>,
    pub local_rope: Option<Rc<Box<dyn EncodableWithState>>>,

    pub prediction_head_dense: Box<dyn EncodableWithState>,
    pub prediction_head_activation: Box<dyn EncodableWithState>,
    pub prediction_head_norm: NormalizationEncodable,
    pub prediction_head_final_linear: Box<dyn EncodableWithState>,

    pub pooling_kernel: PoolingKernel,
    pub sigmoid_kernel: SigmoidKernel,

    pub pooled_buffer: metal::Buffer,
    pub dense_output_buffer: metal::Buffer,
    pub norm_output_buffer: metal::Buffer,
    pub final_logits_buffer: metal::Buffer,
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
            .ok_or(Error::UnableToLoadConfig)?
            .clone();

        let decoder_config = Rc::new(
            classifier_model_config.classifier_config.to_decoder_config(),
        );
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
        // Load weights into shared buffers
        // For BERT, RoPE is under transformer/, not at root, so we need to load them separately
        {
            let mut shared_bufs = shared_buffers.borrow_mut();
            // Load embeddings from root
            shared_bufs.embeddings.update_data(&root_loader_view);
            // Load RoPE from transformer subtree
            let transformer_tree =
                root_loader_view.subtree("transformer").unwrap();
            shared_bufs
                .global_rope
                .update_data(&transformer_tree, String::from("global_rope"));
            if let Some(local_rope) = &mut shared_bufs.local_rope {
                local_rope
                    .update_data(&transformer_tree, String::from("local_rope"));
            }
        }

        let data_type = decoder_config
            .layer_config
            .attention_config
            .qkv_projection_config
            .activation_precision()
            .into();

        let transformer_loader =
            root_loader_view.subtree("transformer").unwrap();

        let embed = embed_block(
            &decoder_config,
            &mtl_context,
            &compilation_config.descriptor_general,
        );

        let global_rope =
            Self::create_rope_block(&mtl_context, data_type, RopeType::Global);
        let local_rope = classifier_model_config
            .classifier_config
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
            .classifier_config
            .transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let mut rope = global_rope.clone();
                if let Some(_window) = classifier_model_config
                    .classifier_config
                    .sliding_window_sizes
                    .as_ref()
                    .and_then(|v| v.get(layer_index).copied().flatten())
                {
                    if let Some(local_rope_block) = local_rope.clone() {
                        rope = local_rope_block;
                    }
                }

                ClassifierLayerExecutable::new(
                    &mtl_context,
                    layer_config,
                    compilation_config.clone(),
                    layer_index,
                    classifier_model_config.classifier_config.model_dim,
                    classifier_model_config
                        .classifier_config
                        .transformer_config
                        .hidden_dim,
                    classifier_model_config.classifier_config.num_heads,
                    classifier_model_config.classifier_config.head_dim,
                    classifier_model_config.classifier_config.num_groups,
                    classifier_model_config.classifier_config.attention_scale,
                    &transformer_loader
                        .subtree(&format!("layers.{}", layer_index))
                        .unwrap(),
                    rope,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let output_norm = NormalizationEncodable::new(
            &mtl_context,
            data_type,
            classifier_model_config
                .classifier_config
                .transformer_config
                .output_norm_config
                .clone(),
            ArrayId::Main,
            ArrayId::Main,
            &transformer_loader.subtree("output_norm").unwrap(),
        )
        .expect("Failed to create output norm kernel");

        let context_length =
            classifier_model_config.classifier_config.context_length;
        let scratch_buffers = ForwardPassBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
            context_length,
            context_length,
        );

        let kv_cache = Rc::new(RefCell::new(KVCache::new(
            &mtl_context,
            &model_shape,
            0,
            context_length,
        )));

        eprintln!(
            "[DEBUG] ClassifierContext::new - Creating embedding normalization..."
        );
        use crate::backends::metal::{
            forward_pass::ArrayId, kernel::NormalizationEncodable,
        };
        let embedding_norm = NormalizationEncodable::new(
            &mtl_context,
            data_type,
            classifier_model_config
                .classifier_config
                .embedding_norm_config
                .clone(),
            ArrayId::Main,
            ArrayId::Main,
            &root_loader_view.subtree("embedding_norm").unwrap(),
        )
        .expect("Failed to create embedding norm kernel");

        eprintln!(
            "[DEBUG] ClassifierContext::new - Creating prediction head..."
        );
        let model_dim = classifier_model_config.classifier_config.model_dim;
        let num_labels = classifier_model_config.classifier_config.num_labels;
        let prediction_head_config =
            &classifier_model_config.classifier_config.prediction_head_config;
        let prediction_head_tree =
            root_loader_view.subtree("prediction_head").unwrap();

        use std::collections::HashMap;

        use mpsgraph::{Device as MPSDevice, Graph};
        use objc2::rc::autoreleasepool;

        use crate::backends::metal::{
            forward_pass::{
                IOArrays, MPSGraphBlock, transformer_layer::linear_block,
            },
            graph::{common::activation, placeholder, shaped_type},
        };

        let prediction_head_data_type: DataType =
            prediction_head_config.dense_config.activation_precision().into();

        let prediction_head_dense = linear_block::<1>(
            &prediction_head_config.dense_config,
            prediction_head_config.use_dense_bias,
            model_dim,
            [model_dim],
            &mtl_context,
            &prediction_head_tree.subtree("dense").unwrap(),
            ArrayId::ClassifierPredictionHeadPooled,
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
            )) as Box<dyn EncodableWithState>
        });

        let prediction_head_norm = NormalizationEncodable::new(
            &mtl_context,
            prediction_head_data_type,
            prediction_head_config.normalization_config.clone(),
            ArrayId::ClassifierPredictionHeadDense,
            ArrayId::ClassifierPredictionHeadNorm,
            &prediction_head_tree.subtree("norm").unwrap(),
        )
        .expect("Failed to create prediction head norm kernel");

        let prediction_head_final_linear = linear_block::<1>(
            &prediction_head_config.final_linear_config,
            true,
            model_dim,
            [num_labels],
            &mtl_context,
            &prediction_head_tree.subtree("final_linear").unwrap(),
            ArrayId::ClassifierPredictionHeadNorm,
            ArrayId::ClassifierPredictionHeadLogits,
            &compilation_config.descriptor_general,
        );

        eprintln!(
            "[DEBUG] ClassifierContext::new - Creating pooling and sigmoid kernels..."
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

        let batch_size = 1;

        let pooled_buffer = mtl_context.device.new_buffer(
            (batch_size * model_dim * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let dense_output_buffer = mtl_context.device.new_buffer(
            (batch_size * model_dim * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let norm_output_buffer = mtl_context.device.new_buffer(
            (batch_size * model_dim * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let final_logits_buffer = mtl_context.device.new_buffer(
            (batch_size * num_labels * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            mtl_context,
            command_buffer,
            kv_cache,
            shared_buffers,
            scratch_buffers,
            model_config: classifier_model_config,
            model_shape,
            embed,
            embedding_norm,
            layers,
            output_norm,
            global_rope,
            local_rope,
            prediction_head_dense,
            prediction_head_activation,
            prediction_head_norm,
            prediction_head_final_linear,
            pooling_kernel,
            sigmoid_kernel,
            pooled_buffer,
            dense_output_buffer,
            norm_output_buffer,
            final_logits_buffer,
        })
    }

    fn create_rope_block(
        mtl_context: &MTLContext,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Rc<Box<dyn EncodableWithState>> {
        use crate::backends::metal::KernelDataType;
        let kernel_data_type: KernelDataType = data_type.into();

        let rotation: Box<dyn EncodableWithState> = Box::new(
            RopeKernelEncodable::new(mtl_context, kernel_data_type, rope_type)
                .expect("Failed to create RopeKernelEncodable"),
        );
        Rc::new(rotation)
    }

    pub fn reset_command_buffer(&mut self) {
        self.command_buffer = MPSCommandBuffer::from_command_queue(
            &self.mtl_context.command_queue,
        );
    }
}
