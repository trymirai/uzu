use std::{
    cell::{Cell, RefCell},
    cmp::min,
    fs::File,
    mem::size_of,
    path::Path,
    rc::Rc,
};

use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{Allocation, AllocationType, Backend, Context, Kernels, kernel::TokenCopySampledKernel},
    config::{decoder::DecoderConfig, model::language_model::LanguageModelConfig},
    data_type::DataType,
    encodable_block::{Decoder, KVCacheUpdate, Sampling},
    forward_pass::{cache_layers::CacheLayers, model_shape::ModelShape},
    language_model::rng::PRng,
    parameters::ParameterLoader,
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, ContextLength, ResolvableValue},
        types::Error,
    },
};

/// Pre-allocated state for async generation pipeline.
/// Indexed by pass_idx to avoid race conditions between GPU passes.
pub struct AsyncBuffers<B: Backend> {
    /// Positions array: [max_tokens] i32
    /// Pre-populated with [prefill_count, prefill_count+1, ...]
    pub positions: Array<B>,
    /// Seeds array: [max_tokens] u64
    /// Pre-populated with deterministic seed sequence
    pub seeds: Array<B>,
    /// Current event counter (pass N waits on N, signals N+1)
    pub counter: Cell<u64>,
    /// Number of tokens after prefill (base for position calculation)
    pub prefill_count: Cell<usize>,
    /// Batch size (number of passes to keep in flight)
    pub batch_size: usize,
    /// Packed [ring_offset, ring_length, token...].
    pub repetition_context_ring: Allocation<B>,
    pub repetition_context_ring_capacity: usize,
}

impl<B: Backend> AsyncBuffers<B> {
    pub fn new(
        context: &B::Context,
        max_tokens: usize,
        batch_size: usize,
    ) -> Self {
        let positions = context.create_array_uninitialized(&[max_tokens], DataType::I32);
        let seeds = context.create_array_uninitialized(&[max_tokens], DataType::U64);
        let repetition_context_ring_capacity = max_tokens;
        let repetition_context_ring = context
            .create_allocation((2 + repetition_context_ring_capacity) * size_of::<u32>(), AllocationType::Global)
            .expect("Failed to create repetition context ring allocation");

        Self {
            positions,
            seeds,
            counter: Cell::new(0),
            prefill_count: Cell::new(0),
            batch_size,
            repetition_context_ring,
            repetition_context_ring_capacity,
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
}

pub struct LanguageModelGeneratorContext<B: Backend> {
    pub context: Rc<B::Context>,

    pub cache_layers: Rc<RefCell<CacheLayers<B>>>,

    pub model_config: LanguageModelConfig,
    pub model_shape: ModelShape,
    pub executables: Decoder<B>,
    pub kv_cache_update: Box<KVCacheUpdate<B>>,
    pub gpu_sampler: Sampling<B>,
    pub seed: PRng,

    /// Kernels for copying sampled tokens in async pipeline
    pub token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    pub token_copy_sampled_context_ring: <B::Kernels as Kernels>::TokenCopySampledKernel,
    /// Pre-allocated buffers for async generation
    pub async_buffers: AsyncBuffers<B>,
}

impl<B: Backend> LanguageModelGeneratorContext<B> {
    pub fn new(
        model_path: &Path,
        decoding_config: &DecodingConfig,
        model_config: &LanguageModelConfig,
    ) -> Result<Self, Error> {
        let context = B::Context::new().map_err(|e| Error::UnableToCreateContext(e.into()))?;

        let prefill_step_size = decoding_config.prefill_step_size.resolve(model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();
        let max_prefix_length: usize =
            Self::get_context_length_internal(&context, &model_config.decoder_config, decoding_config);
        let max_suffix_length: usize = std::cmp::max(prefill_step_size, generate_suffix_length);

        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let root_loader_view =
            loader.tree().subtree("decoder").map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let model_shape = ModelShape::from_decoder_config(&model_config.decoder_config, DataType::BF16);

        let executables = Decoder::new(context.as_ref(), &model_config.decoder_config, &root_loader_view, &model_shape)
            .map_err(|error| Error::UnableToCreateDecoder(Box::new(error)))?;

        // Runtime batch dim never exceeds max_suffix_length (buffers are sized to
        // it), so enumerating 1..=max_suffix_length preheats every matmul exactly.
        let batch_sizes: Vec<u32> = (1..=max_suffix_length as u32).collect();
        executables
            .precompile(context.as_ref(), &batch_sizes)
            .map_err(|error| Error::UnableToCreateDecoder(Box::new(error)))?;

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let kv_cache_update = Box::new(
            KVCacheUpdate::new(context.as_ref(), model_shape.data_type, max_prefix_length)
                .map_err(|e| Error::UnableToCreateContext(e.into()))?,
        );

        let gpu_sampler = Sampling::<B>::new(model_shape.data_type, model_config.decoder_config.vocab_size);

        let token_copy_sampled = <B::Kernels as Kernels>::TokenCopySampledKernel::new(&context, false)
            .map_err(|e| Error::UnableToCreateContext(e.into()))?;
        let token_copy_sampled_context_ring = <B::Kernels as Kernels>::TokenCopySampledKernel::new(&context, true)
            .map_err(|e| Error::UnableToCreateContext(e.into()))?;

        let async_batch_size = decoding_config
            .async_batch_size
            .resolve::<B>(model_path, context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let async_buffers = AsyncBuffers::new(context.as_ref(), max_prefix_length, async_batch_size);

        loader.tree().assert_all_tensors_validated().map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;

        let seed = PRng::new(decoding_config.sampling_seed.resolve());

        Ok(Self {
            context,
            cache_layers,
            model_config: model_config.clone(),
            model_shape,
            executables,
            kv_cache_update,
            gpu_sampler,
            seed,
            token_copy_sampled,
            token_copy_sampled_context_ring,
            async_buffers,
        })
    }

    pub fn get_context_length(
        &self,
        decoding_config: &DecodingConfig,
    ) -> usize {
        Self::get_context_length_internal(&self.context, &self.model_config.decoder_config, decoding_config)
    }

    fn get_context_length_internal(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        decoding_config: &DecodingConfig,
    ) -> usize {
        let model_length = decoder_config.transformer_config.max_sequence_length().unwrap_or(65536);
        let proposed_value = match decoding_config.context_length {
            ContextLength::Default => {
                if context.sparse_buffers_supported() {
                    model_length
                } else if cfg!(target_os = "ios") {
                    8192
                } else {
                    16384
                }
            },
            ContextLength::Maximal => model_length,
            ContextLength::Custom(value) => value,
        };
        min(proposed_value, model_length)
    }
}
