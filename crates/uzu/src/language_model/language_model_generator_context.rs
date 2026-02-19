use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::BufReader,
    path::Path,
    rc::Rc,
};

use crate::{
    DataType,
    backends::common::{
        Backend, Context, Kernels, NativeBuffer,
        kernel::{
            MaskUpdateKernel, TokenCopySampledKernel, TokenCopyToResultsKernel, kv_cache_update::KVCacheUpdate,
            matmul::MatmulKernels,
        },
    },
    config::{DecoderConfig, LanguageModelConfig, ModelMetadata},
    encodable_block::{Decoder, Sampling},
    forward_pass::{
        cache_layers::CacheLayers, model_shape::ModelShape, scratch_buffers::ScratchBuffers, state::SharedBuffers,
    },
    language_model::rng::PRng,
    parameters::ParameterLoader,
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, ResolvableValue},
        types::Error,
    },
};

/// Pre-allocated buffers for async generation pipeline.
/// Indexed by pass_idx to avoid race conditions between GPU passes.
pub struct AsyncBuffers<B: Backend> {
    /// Positions buffer: [max_tokens] i32
    /// Pre-populated with [prefill_count, prefill_count+1, ...]
    pub positions: B::NativeBuffer,
    /// Seeds buffer: [max_tokens] u64
    /// Pre-populated with deterministic seed sequence
    pub seeds: B::NativeBuffer,
    /// Results buffer: [batch_size] u32
    /// Each pass writes its sampled token to results[pass_idx % batch_size]
    pub results: B::NativeBuffer,
    /// Event for GPU-side synchronization between passes
    pub event: B::Event,
    /// Current event counter (pass N waits on N, signals N+1)
    pub counter: Cell<u64>,
    /// Number of tokens after prefill (base for position calculation)
    pub prefill_count: Cell<usize>,
    /// Batch size (number of passes to keep in flight)
    pub batch_size: usize,
}

impl<B: Backend> AsyncBuffers<B> {
    pub fn new(
        context: &B::Context,
        max_tokens: usize,
        batch_size: usize,
    ) -> Self {
        let positions =
            context.create_buffer(max_tokens * std::mem::size_of::<i32>()).expect("Failed to create positions buffer");
        let seeds =
            context.create_buffer(max_tokens * std::mem::size_of::<u64>()).expect("Failed to create seeds buffer");
        let results =
            context.create_buffer(batch_size * std::mem::size_of::<u32>()).expect("Failed to create results buffer");
        let event = context.create_event().expect("Failed to create event");

        Self {
            positions,
            seeds,
            results,
            event,
            counter: Cell::new(0),
            prefill_count: Cell::new(0),
            batch_size,
        }
    }

    /// Prepare positions buffer: [prefill_count-1, prefill_count, ...]
    /// Uses prefill_count-1 to match sync path which uses self.tokens.len()-1
    pub fn prepare_positions(
        &self,
        prefill_count: usize,
        tokens_to_generate: usize,
    ) {
        self.prefill_count.set(prefill_count);
        let base_position = prefill_count.saturating_sub(1);
        let ptr = self.positions.cpu_ptr().as_ptr() as *mut i32;
        for i in 0..tokens_to_generate {
            unsafe {
                *ptr.add(i) = (base_position + i) as i32;
            }
        }
    }

    /// Prepare seeds buffer with deterministic sequence
    pub fn prepare_seeds(
        &self,
        seed: &PRng,
        prefix_len: usize,
        tokens_to_generate: usize,
    ) {
        let ptr = self.seeds.cpu_ptr().as_ptr() as *mut u64;
        for i in 0..tokens_to_generate {
            unsafe {
                *ptr.add(i) = seed.derive((prefix_len + i - 1) as u64);
            }
        }
    }

    /// Reset event counter before async generation
    pub fn reset_counter(&self) {
        self.counter.set(0);
    }

    /// Read sampled token from results buffer at given pass index
    pub fn read_result(
        &self,
        pass_idx: usize,
    ) -> u32 {
        let ptr = self.results.cpu_ptr().as_ptr() as *const u32;
        unsafe { *ptr.add(pass_idx % self.batch_size) }
    }
}

pub struct LanguageModelGeneratorContext<B: Backend> {
    pub context: Rc<B::Context>,
    pub command_buffer: B::CommandBuffer,

    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    pub scratch_buffers: ScratchBuffers<B>,

    pub model_config: LanguageModelConfig,
    pub decoder_config: Rc<DecoderConfig>,
    pub model_shape: ModelShape,
    pub executables: Decoder<B>,
    pub kv_cache_update: Box<KVCacheUpdate<B>>,
    pub gpu_sampler: Sampling<B>,
    pub seed: PRng,

    /// Kernels for copying sampled tokens in async pipeline
    pub token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    pub token_copy_results: <B::Kernels as Kernels>::TokenCopyToResultsKernel,
    /// Kernel for updating attention mask between async passes
    pub mask_update: Option<<B::Kernels as Kernels>::MaskUpdateKernel>,
    /// Pre-allocated buffers for async generation
    pub async_buffers: AsyncBuffers<B>,
}

impl<B: Backend + 'static> LanguageModelGeneratorContext<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new(
        model_path: &Path,
        decoding_config: &DecodingConfig,
    ) -> Result<Self, Error> {
        let context = B::Context::new().map_err(|_| Error::UnableToCreateBackendContext)?;

        let command_buffer = context.create_command_buffer().expect("Failed to create command buffer");

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        // Extract language model config
        let language_model_config = model_metadata.model_config.as_language_model().ok_or(Error::UnableToLoadConfig)?;

        let decoder_config = Rc::new(language_model_config.decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let prefill_step_size = decoding_config.prefill_step_size.resolve(language_model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();
        let max_prefix_length: usize = decoding_config.context_length.resolve(language_model_config);
        let max_suffix_length: usize = std::cmp::max(prefill_step_size, generate_suffix_length);

        let weights_path = model_path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        shared_buffers.borrow_mut().update_data(&root_loader_view);

        let scratch_buffers =
            ScratchBuffers::new(context.as_ref(), &decoder_config, &model_shape, max_prefix_length, max_suffix_length);

        let executables = Decoder::new(context.clone(), decoder_config.clone(), &root_loader_view);

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let intermediate_data_type: DataType = decoder_config.output_norm_config.scale_precision.into();
        let kv_cache_update = Box::new(
            KVCacheUpdate::new(context.as_ref(), intermediate_data_type, max_prefix_length)
                .map_err(|_| Error::UnableToCreateBackendContext)?,
        );

        let gpu_sampler =
            Sampling::<B>::new(&context, intermediate_data_type, max_suffix_length, decoder_config.vocab_size)
                .map_err(|_| Error::UnableToCreateBackendContext)?;

        let token_copy_sampled = <B::Kernels as Kernels>::TokenCopySampledKernel::new(&context)
            .map_err(|_| Error::UnableToCreateBackendContext)?;
        let token_copy_results = <B::Kernels as Kernels>::TokenCopyToResultsKernel::new(&context)
            .map_err(|_| Error::UnableToCreateBackendContext)?;

        // Create mask update kernel if model has attention layers
        let mask_update = if decoder_config.has_attention_layers() {
            Some(
                <B::Kernels as Kernels>::MaskUpdateKernel::new(&context, intermediate_data_type)
                    .map_err(|_| Error::UnableToCreateBackendContext)?,
            )
        } else {
            None
        };

        let async_batch_size = decoding_config.async_batch_size.resolve::<B>(model_path, context.as_ref());
        let async_buffers = AsyncBuffers::new(context.as_ref(), max_prefix_length, async_batch_size);

        let seed = PRng::new(decoding_config.sampling_seed.resolve());

        let context = Self {
            context,
            command_buffer,
            cache_layers,
            shared_buffers,
            scratch_buffers,
            model_config: language_model_config.clone(),
            decoder_config,
            model_shape,
            executables,
            kv_cache_update,
            gpu_sampler,
            seed,
            token_copy_sampled,
            token_copy_results,
            mask_update,
            async_buffers,
        };

        Ok(context)
    }

    pub fn reset_command_buffer(&mut self) {
        self.command_buffer = self.context.create_command_buffer().expect("Failed to create command buffer");
    }
}
