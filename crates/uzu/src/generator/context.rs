use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::BufReader,
    path::Path,
    rc::Rc,
};

use metal::Event as MTLEvent;
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::Retained;

use crate::{
    DataType,
    backends::metal::{
        CacheLayers, DecoderExecutables, KVCacheUpdate, KernelDataType,
        MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{ForwardPassBuffers, SharedBuffers},
        kernel::{SamplingKernelEncodable, TokenCopyKernel},
    },
    config::{LanguageModelConfig, ModelMetadata},
    generator::rng::DerivableSeed,
    parameters::ParameterLoader,
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, ResolvableValue},
        types::Error,
    },
};

/// Pre-allocated buffers for async generation pipeline.
/// Indexed by pass_idx to avoid race conditions between GPU passes.
pub struct AsyncBuffers {
    /// Positions buffer: [max_tokens] i32
    /// Pre-populated with [prefill_count, prefill_count+1, ...]
    pub positions: metal::Buffer,
    /// Seeds buffer: [max_tokens] u64
    /// Pre-populated with deterministic seed sequence
    pub seeds: metal::Buffer,
    /// Results buffer: [lookahead] u32
    /// Each pass writes its sampled token to results[pass_idx % lookahead]
    pub results: metal::Buffer,
    /// Metal event for GPU-side synchronization between passes
    pub event: MTLEvent,
    /// Current event counter (pass N waits on N, signals N+1)
    pub counter: Cell<u64>,
    /// Number of tokens after prefill (base for position calculation)
    pub prefill_count: Cell<usize>,
    /// Lookahead count (number of passes to keep in flight)
    pub lookahead: usize,
}

impl AsyncBuffers {
    pub fn new(
        device: &metal::DeviceRef,
        max_tokens: usize,
        lookahead: usize,
    ) -> Self {
        let positions = device.new_buffer(
            (max_tokens * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let seeds = device.new_buffer(
            (max_tokens * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let results = device.new_buffer(
            (lookahead * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let event = device.new_event();

        Self {
            positions,
            seeds,
            results,
            event,
            counter: Cell::new(0),
            prefill_count: Cell::new(0),
            lookahead,
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
        let ptr = self.positions.contents() as *mut i32;
        for i in 0..tokens_to_generate {
            unsafe {
                *ptr.add(i) = (base_position + i) as i32;
            }
        }
    }

    /// Prepare seeds buffer with deterministic sequence
    pub fn prepare_seeds(
        &self,
        seed_source: &mut DerivableSeed,
        tokens_to_generate: usize,
    ) {
        let ptr = self.seeds.contents() as *mut u64;
        for i in 0..tokens_to_generate {
            unsafe {
                *ptr.add(i) = seed_source.next();
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
        let ptr = self.results.contents() as *const u32;
        unsafe { *ptr.add(pass_idx % self.lookahead) }
    }
}

pub struct GeneratorContext {
    pub mtl_context: Rc<MTLContext>,
    pub command_buffer: Retained<MPSCommandBuffer>,

    pub cache_layers: Rc<RefCell<CacheLayers>>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    pub scratch_buffers: ForwardPassBuffers,

    pub model_config: LanguageModelConfig,
    pub model_shape: ModelShape,
    pub executables: DecoderExecutables,
    pub kv_cache_update: Box<KVCacheUpdate>,
    pub gpu_sampler: SamplingKernelEncodable,
    pub next_seed: DerivableSeed,

    /// Kernel for copying sampled tokens in async pipeline
    pub token_copy: TokenCopyKernel,
    /// Pre-allocated buffers for async generation
    pub async_buffers: AsyncBuffers,
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

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
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
        )
        .map_err(|_| Error::UnableToCreateMetalContext)?;

        let token_copy = TokenCopyKernel::new(&mtl_context)
            .map_err(|_| Error::UnableToCreateMetalContext)?;

        let async_lookahead = 16;
        let async_buffers = AsyncBuffers::new(
            &mtl_context.device,
            max_prefix_length,
            async_lookahead,
        );

        let base_seed = decoding_config.sampling_seed.resolve();
        let next_seed = DerivableSeed::new(base_seed);

        let context = Self {
            mtl_context,
            command_buffer,
            cache_layers,
            shared_buffers,
            scratch_buffers,
            model_config: model_metadata.model_config.clone(),
            model_shape,
            executables,
            kv_cache_update,
            gpu_sampler,
            next_seed,
            token_copy,
            async_buffers,
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
