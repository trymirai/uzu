use std::{cell::RefCell, fs::File, io::BufReader, path::Path, rc::Rc};

use metal::Event;
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::Retained;

use crate::{
    DataType,
    backends::metal::{
        DecoderExecutables, KVCache, KVCacheUpdate, KernelDataType, MTLContext,
        ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{ForwardPassBuffers, SharedBuffers},
        kernel::SamplingKernelEncodable,
    },
    config::{LanguageModelConfig, ModelMetadata},
    parameters::ParameterLoader,
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, ResolvableValue},
        types::Error,
    },
};

pub struct GeneratorContext {
    pub mtl_context: Rc<MTLContext>,
    pub command_buffer: Retained<MPSCommandBuffer>,
    pub kv_update_event: Event,
    pub kv_update_signal: u64,

    pub kv_cache: Rc<RefCell<KVCache>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    pub scratch_buffers: ForwardPassBuffers,

    pub model_config: LanguageModelConfig,
    pub model_shape: ModelShape,
    pub executables: DecoderExecutables,
    pub kv_cache_update: Box<KVCacheUpdate>,
    pub gpu_sampler: SamplingKernelEncodable,
}

impl GeneratorContext {
    pub fn new(
        model_path: &Path,
        decoding_config: &DecodingConfig,
    ) -> Result<Self, Error> {
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
        let decoder_config =
            Rc::new(model_metadata.model_config.decoder_config.clone());
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let prefill_step_size = decoding_config
            .prefill_step_size
            .resolve(&model_metadata.model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();
        let max_prefix_length: usize = decoding_config
            .context_length
            .resolve(&model_metadata.model_config);
        let max_suffix_length: usize =
            std::cmp::max(prefill_step_size, generate_suffix_length);

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
        shared_buffers.borrow_mut().update_data(&root_loader_view);

        let scratch_buffers = ForwardPassBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        );

        let executables = DecoderExecutables::new(
            mtl_context.clone(),
            decoder_config.clone(),
            &root_loader_view,
            compilation_config.clone(),
        );

        let kv_cache = Rc::new(RefCell::new(KVCache::new(
            &mtl_context,
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let intermediate_data_type: DataType =
            decoder_config.output_norm_config.scale_precision.into();
        let kernel_data_type: KernelDataType = intermediate_data_type.into();
        let kv_cache_update = Box::new(
            KVCacheUpdate::new(
                &mtl_context,
                kernel_data_type,
                max_prefix_length,
            )
            .map_err(|_| Error::UnableToCreateMetalContext)?,
        );

        let gpu_sampler = SamplingKernelEncodable::new(
            &mtl_context,
            kernel_data_type,
            max_suffix_length,
            decoder_config.vocab_size,
            decoding_config.sampling_seed.resolve(),
        )
        .map_err(|_| Error::UnableToCreateMetalContext)?;

        let kv_update_event = mtl_context.device.new_event();
        let kv_update_signal = 1;

        let context = Self {
            mtl_context,
            command_buffer,
            kv_update_event,
            kv_update_signal,
            kv_cache,
            shared_buffers,
            scratch_buffers,
            model_config: model_metadata.model_config.clone(),
            model_shape,
            executables,
            kv_cache_update,
            gpu_sampler,
        };

        return Ok(context);
    }

    pub fn reset_command_buffer(&mut self) {
        objc2::rc::autoreleasepool(|_pool| {
            self.command_buffer = MPSCommandBuffer::from_command_queue(
                &self.mtl_context.command_queue,
            );
        });
    }
}
