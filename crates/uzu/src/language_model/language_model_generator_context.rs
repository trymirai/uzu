use std::{
    cell::{Cell, RefCell},
    fs::File,
    path::Path,
    rc::Rc,
};

use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::{
        Backend, Context, Kernels,
        kernel::{TokenCopySampledKernel, kv_cache_update::KVCacheUpdate},
    },
    config::{LanguageModelConfig, ModelMetadata},
    encodable_block::{Decoder, Sampling},
    forward_pass::{cache_layers::CacheLayers, model_shape::ModelShape, state::SharedBuffers},
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
    pub positions: Array<B>,
    /// Seeds buffer: [max_tokens] u64
    /// Pre-populated with deterministic seed sequence
    pub seeds: Array<B>,
    /// Results buffers: one scalar slot per in-flight pass.
    pub results: Box<[Array<B>]>,
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
        let positions = context.create_array_uninitialized(&[max_tokens], DataType::I32, "async_positions");
        let seeds = context.create_array_uninitialized(&[max_tokens], DataType::U64, "async_seeds");
        let results =
            (0..batch_size).map(|_| context.create_array_uninitialized(&[1], DataType::U32, "async_result")).collect();
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
        &mut self,
        prefill_count: usize,
        tokens_to_generate: usize,
    ) {
        self.prefill_count.set(prefill_count);
        let base_position = prefill_count.saturating_sub(1);
        let positions = self.positions.as_slice_mut::<i32>();
        for (index, position) in positions.iter_mut().take(tokens_to_generate).enumerate() {
            *position = (base_position + index) as i32;
        }
    }

    /// Prepare seeds buffer with deterministic sequence
    pub fn prepare_seeds(
        &mut self,
        seed: &PRng,
        prefix_len: usize,
        tokens_to_generate: usize,
    ) {
        let seeds = self.seeds.as_slice_mut::<u64>();
        for (index, value) in seeds.iter_mut().take(tokens_to_generate).enumerate() {
            *value = seed.derive((prefix_len + index - 1) as u64);
        }
    }

    /// Reset event counter before async generation
    pub fn reset_counter(&self) {
        self.counter.set(0);
    }

    pub fn position(
        &self,
        index: usize,
    ) -> Array<B> {
        self.positions.view_at_offset(index * std::mem::size_of::<i32>(), &[1])
    }

    pub fn seed(
        &self,
        index: usize,
    ) -> Array<B> {
        self.seeds.view_at_offset(index * std::mem::size_of::<u64>(), &[1])
    }
}

pub struct LanguageModelGeneratorContext<B: Backend> {
    pub context: Rc<B::Context>,

    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,
    pub shared_buffers: Rc<SharedBuffers<B>>,

    pub model_config: LanguageModelConfig,
    pub model_shape: ModelShape,
    pub executables: Decoder<B>,
    pub kv_cache_update: Box<KVCacheUpdate<B>>,
    pub gpu_sampler: Sampling<B>,
    pub seed: PRng,

    /// Kernels for copying sampled tokens in async pipeline
    pub token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    /// Pre-allocated buffers for async generation
    pub async_buffers: AsyncBuffers<B>,
}

impl<B: Backend> LanguageModelGeneratorContext<B> {
    pub fn new(
        model_path: &Path,
        decoding_config: &DecodingConfig,
        model_metadata: &ModelMetadata,
    ) -> Result<Self, Error> {
        let context = B::Context::new().map_err(|e| Error::UnableToCreateContext(e.into()))?;

        // Extract language model config
        let language_model_config = model_metadata.model_config.as_language_model().ok_or(Error::UnableToLoadConfig)?;

        let decoder_config = language_model_config.decoder_config().map_err(|_| Error::UnableToLoadConfig)?;
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

        let mut shared_buffers = SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape);
        shared_buffers.update_data(&root_loader_view);
        let shared_buffers = Rc::new(shared_buffers);

        let executables = Decoder::new(context.as_ref(), &decoder_config, &root_loader_view);

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let intermediate_data_type: DataType = decoder_config.output_norm_config.scale_precision.into();
        let kv_cache_update = Box::new(
            KVCacheUpdate::new(context.as_ref(), intermediate_data_type, max_prefix_length)
                .map_err(|e| Error::UnableToCreateContext(e.into()))?,
        );

        let gpu_sampler =
            Sampling::<B>::new(&context, intermediate_data_type, max_suffix_length, decoder_config.vocab_size)
                .map_err(|e| Error::UnableToCreateContext(e.into()))?;

        let token_copy_sampled = <B::Kernels as Kernels>::TokenCopySampledKernel::new(&context)
            .map_err(|e| Error::UnableToCreateContext(e.into()))?;

        let async_batch_size = decoding_config.async_batch_size.resolve::<B>(model_path, context.as_ref());
        let async_buffers = AsyncBuffers::new(context.as_ref(), max_prefix_length, async_batch_size);

        let seed = PRng::new(decoding_config.sampling_seed.resolve());

        let context = Self {
            context,
            cache_layers,
            shared_buffers,
            model_config: language_model_config.clone(),
            model_shape,
            executables,
            kv_cache_update,
            gpu_sampler,
            seed,
            token_copy_sampled,
            async_buffers,
        };

        Ok(context)
    }
}
