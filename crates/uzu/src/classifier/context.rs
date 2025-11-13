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
            PoolingKernel, RMSNormKernelEncodable, RopeKernelEncodable,
            SigmoidKernel,
        },
    },
    classifier::{ClassifierLayerExecutable, PredictionHeadExecutables},
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
    pub embedding_norm: Box<dyn EncodableWithState>,
    pub layers: Box<[ClassifierLayerExecutable]>,
    pub output_norm: Box<dyn EncodableWithState>,
    pub global_rope: Rc<Box<dyn EncodableWithState>>,
    pub local_rope: Option<Rc<Box<dyn EncodableWithState>>>,

    pub prediction_head: PredictionHeadExecutables,
    pub pooling_kernel: PoolingKernel,
    pub sigmoid_kernel: SigmoidKernel,

    pub pooled_buffer: metal::Buffer,
    pub dense_output_buffer: metal::Buffer,
    pub norm_output_buffer: metal::Buffer,
    pub final_logits_buffer: metal::Buffer,
}

impl ClassifierContext {
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        eprintln!("[DEBUG] ClassifierContext::new - Creating Metal device...");
        let mtl_device = metal::Device::system_default()
            .ok_or(Error::UnableToCreateMetalContext)?;
        let mtl_command_queue =
            mtl_device.new_command_queue_with_max_command_buffer_count(1024);

        let command_buffer =
            MPSCommandBuffer::from_command_queue(&mtl_command_queue);

        eprintln!("[DEBUG] ClassifierContext::new - Loading config...");
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

        eprintln!("[DEBUG] ClassifierContext::new - Creating MTLContext...");
        let mtl_context = Rc::new(
            MTLContext::new(mtl_device, mtl_command_queue)
                .map_err(|_| Error::UnableToCreateMetalContext)?,
        );

        let compilation_config = Rc::new(CompilationConfig::default());
        eprintln!("[DEBUG] ClassifierContext::new - Loading weights...");
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

        let output_norm = Box::new(
            RMSNormKernelEncodable::new(
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
            .expect("Failed to create output norm kernel"),
        );

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
        use crate::backends::metal::forward_pass::ArrayId;
        let embedding_norm = Box::new(
            RMSNormKernelEncodable::new(
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
            .expect("Failed to create embedding norm kernel"),
        );

        eprintln!(
            "[DEBUG] ClassifierContext::new - Creating prediction head..."
        );
        let model_dim = classifier_model_config.classifier_config.model_dim;
        let num_labels = classifier_model_config.classifier_config.num_labels;
        let prediction_head = PredictionHeadExecutables::new(
            mtl_context.clone(),
            &classifier_model_config.classifier_config.prediction_head_config,
            model_dim,
            num_labels,
            &root_loader_view.subtree("prediction_head").unwrap(),
            compilation_config.clone(),
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
        eprintln!("[DEBUG] ClassifierContext::new - All components created");

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
            prediction_head,
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
