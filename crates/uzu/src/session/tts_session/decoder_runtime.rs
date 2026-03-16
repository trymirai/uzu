use super::decoder_support::*;
use super::*;
use crate::backends::common::Buffer;

struct TokenDecoderLoadedModel<B: Backend> {
    shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    scratch_buffers: ScratchBuffers<B>,
    executables: Decoder<B>,
    sampler: GpuSampling<B>,
    token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    token_copy_results: <B::Kernels as Kernels>::TokenCopyToResultsKernel,
}

impl<B: Backend> TokenDecoderLoadedModel<B> {
    fn load(
        context: &Rc<B::Context>,
        model_path: &Path,
        decoder_config: &Rc<crate::config::DecoderConfig>,
        model_shape: &ModelShape,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> Result<Self, Error> {
        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers =
            TokenDecoderContext::<B>::build_shared_buffers(context, decoder_config, model_shape, &root_loader_view, transformer_subtree)?;
        let scratch_buffers =
            ScratchBuffers::new(context.as_ref(), decoder_config, model_shape, max_prefix_length, max_suffix_length);
        let executables = Decoder::new_with_subtrees(
            context.clone(),
            decoder_config.clone(),
            &root_loader_view,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
        );
        let logits_data_type = scratch_buffers.logits.borrow().data_type();
        let sampler =
            GpuSampling::new(context.as_ref(), logits_data_type, max_suffix_length, decoder_config.vocab_size)
                .map_err(unable_to_create_context)?;
        let token_copy_sampled =
            <B::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;
        let token_copy_results =
            <B::Kernels as Kernels>::TokenCopyToResultsKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;

        Ok(Self {
            shared_buffers,
            scratch_buffers,
            executables,
            sampler,
            token_copy_sampled,
            token_copy_results,
        })
    }
}

struct TokenDecoderContext<B: Backend> {
    context: Rc<B::Context>,
    command_buffer: Rc<RefCell<<B::CommandBuffer as CommandBuffer>::Encoding>>,
    cache_layers: Rc<RefCell<CacheLayers<B>>>,
    shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    scratch_buffers: ScratchBuffers<B>,
    model_shape: ModelShape,
    decoder_config: Rc<crate::config::DecoderConfig>,
    executables: Decoder<B>,
    sampler: GpuSampling<B>,
    kv_cache_update: KVCacheUpdate<B>,
    token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    token_copy_results: <B::Kernels as Kernels>::TokenCopyToResultsKernel,
    async_chain_positions: Rc<RefCell<B::Buffer>>,
    async_chain_seeds: Rc<RefCell<B::Buffer>>,
    async_chain_results: Rc<RefCell<B::Buffer>>,
    async_chain_capacity: usize,
}

impl<B: Backend> TokenDecoderContext<B> {
    pub(super) fn context(&self) -> &Rc<B::Context> {
        &self.context
    }

    fn new(
        context: Rc<B::Context>,
        model_path: &Path,
        decoder_config: Rc<crate::config::DecoderConfig>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        runtime_config: &TextDecoderRuntimeConfig,
    ) -> Result<(Self, DataType, bool), Error> {
        let command_buffer = Rc::new(RefCell::new(
            context.create_command_buffer().expect("Failed to create command buffer").start_encoding(),
        ));
        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = text_decoder_prefill_step_size(runtime_config, decoder_config.context_length).max(32);
        let should_fill_attention_bias =
            model_shape.sliding_window_length_per_layer.iter().any(|value| value.is_some());
        let activation_data_type = model_shape.activation_data_type();
        let loaded_model = TokenDecoderLoadedModel::<B>::load(
            &context,
            model_path,
            &decoder_config,
            &model_shape,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
            max_prefix_length,
            max_suffix_length,
        )?;
        let async_chain_capacity = max_suffix_length.max(1);
        let (async_chain_positions, async_chain_seeds, async_chain_results) =
            Self::build_async_chain_buffers(&context, async_chain_capacity)?;
        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));
        let intermediate_data_type: DataType = decoder_config.output_norm_config.scale_precision.into();
        let kv_cache_update = KVCacheUpdate::new(context.as_ref(), intermediate_data_type, max_prefix_length)
            .map_err(unable_to_create_context)?;

        Ok((
            Self {
                context,
                command_buffer,
                cache_layers,
                shared_buffers: loaded_model.shared_buffers,
                scratch_buffers: loaded_model.scratch_buffers,
                model_shape,
                decoder_config,
                executables: loaded_model.executables,
                sampler: loaded_model.sampler,
                kv_cache_update,
                token_copy_sampled: loaded_model.token_copy_sampled,
                token_copy_results: loaded_model.token_copy_results,
                async_chain_positions,
                async_chain_seeds,
                async_chain_results,
                async_chain_capacity,
            },
            activation_data_type,
            should_fill_attention_bias,
        ))
    }

    fn build_shared_buffers(
        context: &Rc<B::Context>,
        decoder_config: &Rc<crate::config::DecoderConfig>,
        model_shape: &ModelShape,
        root_loader_view: &crate::parameters::ParameterTree<B::Context>,
        transformer_subtree: &str,
    ) -> Result<Rc<RefCell<SharedBuffers<B>>>, Error> {
        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), decoder_config, model_shape)));
        let transformer_tree = root_loader_view.subtree(transformer_subtree).map_err(|_| Error::UnableToLoadWeights)?;
        let mut shared_buffers_borrow = shared_buffers.borrow_mut();
        if let Some(global_rope) = &mut shared_buffers_borrow.global_rope {
            global_rope.update_data(&transformer_tree, "global_rope");
        }
        if let Some(local_rope) = &mut shared_buffers_borrow.local_rope {
            local_rope.update_data(&transformer_tree, "local_rope");
        }
        drop(shared_buffers_borrow);
        Ok(shared_buffers)
    }

    fn build_async_chain_buffers(
        context: &Rc<B::Context>,
        async_chain_capacity: usize,
    ) -> Result<(Rc<RefCell<B::Buffer>>, Rc<RefCell<B::Buffer>>, Rc<RefCell<B::Buffer>>), Error> {
        let positions = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<i32>())
                .map_err(unable_to_create_context)?,
        ));
        let seeds = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u64>())
                .map_err(unable_to_create_context)?,
        ));
        let results = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(unable_to_create_context)?,
        ));
        Ok((positions, seeds, results))
    }
}

pub(super) struct TokenDecoderRunner<B: Backend> {
    ctx: TokenDecoderContext<B>,
    pub(super) single_hidden_capture: ArrayCell<B>,
    pub(super) single_override_embedding: ArrayCell<B>,
    tensor_copy: <B::Kernels as Kernels>::TensorCopyKernel,
    tensor_add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    should_fill_attention_bias: bool,
    next_position: usize,
    instrumentation: RunnerInstrumentation,
}

impl<B: Backend> TokenDecoderRunner<B> {
    pub(super) fn context(&self) -> &Rc<B::Context> {
        self.ctx.context()
    }

    pub(super) fn new_with_context(
        context: Rc<B::Context>,
        model_path: &Path,
        decoder_config: Rc<crate::config::DecoderConfig>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        runtime_config: &TextDecoderRuntimeConfig,
    ) -> Result<Self, Error> {
        let (ctx, activation_data_type, should_fill_attention_bias) = TokenDecoderContext::new(
            context,
            model_path,
            decoder_config,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
            runtime_config,
        )?;
        let context: Rc<B::Context> = Rc::clone(ctx.context());
        let model_dim = ctx.decoder_config.model_dim;
        let tensor_copy =
            <B::Kernels as Kernels>::TensorCopyKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let tensor_add_scale =
            <B::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let single_hidden_capture = RefCell::new(context.create_array(
            &[1, model_dim],
            activation_data_type,
            "tts_single_hidden_capture",
        ));
        let single_override_embedding = RefCell::new(context.create_array(
            &[1, model_dim],
            activation_data_type,
            "tts_single_override_embedding",
        ));

        Ok(Self {
            ctx,
            single_hidden_capture,
            single_override_embedding,
            tensor_copy,
            tensor_add_scale,
            single_token_vocab_masks: HashMap::new(),
            two_token_vocab_masks: HashMap::new(),
            should_fill_attention_bias,
            next_position: 0,
            instrumentation: RunnerInstrumentation::default(),
        })
    }

    pub(super) fn reset(&mut self) {
        self.ctx.cache_layers.borrow_mut().clear();
        self.next_position = 0;
    }

    pub(super) fn prefill_without_sampling(
        &mut self,
        token_ids: &[u64],
    ) -> Result<(), Error> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut sampling = TextSamplingState::with_params(0, 0.0, 1.0);
        let _ = self.decode_next_step(token_ids, EmbeddingInjection::None, None, &mut sampling, None, false, None)?;
        Ok(())
    }

    pub(super) fn decode_next_token_with_hidden_capture(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
    ) -> Result<u64, Error> {
        self.decode_next_step(token_ids, embedding_injection, None, sampling, precomputed_token_bitmask, true, None)
    }

    pub(super) fn decode_next_token_with_hidden_capture_and_pre_injection(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
    ) -> Result<u64, Error> {
        self.decode_next_step(
            token_ids,
            embedding_injection,
            None,
            sampling,
            precomputed_token_bitmask,
            true,
            pre_injection_encode,
        )
    }

    pub(super) fn decode_next_token(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
    ) -> Result<u64, Error> {
        self.decode_next_step(token_ids, embedding_injection, vocab_limit, sampling, None, false, None)
    }

    pub(super) fn decode_followup_tokens_batched(
        &mut self,
        first_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if followup_count == 0 {
            return Ok(());
        }

        if followup_count > self.ctx.async_chain_capacity {
            return Err(Error::GenerateFailed);
        }

        let vocab_mask_limit = if let Some(limit_raw) = vocab_limit {
            let limit = limit_raw.min(self.ctx.decoder_config.vocab_size);
            if limit == 0 || limit >= self.ctx.decoder_config.vocab_size {
                None
            } else {
                self.prepare_two_token_vocab_mask(limit)?;
                self.prepare_single_token_vocab_mask(limit)?;
                Some(limit)
            }
        } else {
            None
        };

        {
            let positions_borrow = self.ctx.async_chain_positions.borrow();
            let positions_ptr = positions_borrow.cpu_ptr().as_ptr() as *mut i32;
            for pass in 0..followup_count {
                unsafe {
                    *positions_ptr.add(pass) = (self.next_position + pass) as i32;
                }
            }
        }

        {
            let seeds_borrow = self.ctx.async_chain_seeds.borrow();
            let seeds_ptr = seeds_borrow.cpu_ptr().as_ptr() as *mut u64;
            if !matches!(sampling.method(), SamplingMethod::Greedy) {
                for pass in 0..followup_count {
                    unsafe {
                        *seeds_ptr.add(pass) = sampling.next_seed();
                    }
                }
            } else {
                for pass in 0..followup_count {
                    unsafe {
                        *seeds_ptr.add(pass) = 0;
                    }
                }
            }
        }

        *self.ctx.command_buffer.borrow_mut() =
            self.ctx.context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
        let encoding_parameters = EncodingParameters::new();
        let context_rc = self.ctx.context.clone();
        let cache_layers_rc = self.ctx.cache_layers.clone();
        let shared_buffers_rc = self.ctx.shared_buffers.clone();
        let positions_rc = self.ctx.async_chain_positions.clone();
        let seeds_rc = self.ctx.async_chain_seeds.clone();
        let token_bitmask = vocab_mask_limit.and_then(|limit| self.get_single_token_vocab_mask(limit));
        let sampling_method = sampling.method();
        for pass in 0..followup_count {
            let token_ids = [if pass == 0 {
                first_token
            } else {
                0
            }];
            let mut state = ForwardPassState::new_llm(
                context_rc.clone(),
                &self.ctx.decoder_config,
                &self.ctx.model_shape,
                &self.ctx.scratch_buffers,
                cache_layers_rc.clone(),
                shared_buffers_rc.clone(),
                &token_ids,
                &[self.next_position + pass],
                token_bitmask,
                &[0],
                1,
                0,
                1,
                false,
                None,
                pass > 0,
                self.should_fill_attention_bias,
                Some((positions_rc.clone(), pass)),
                Some((seeds_rc.clone(), pass)),
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling_method);
            }

            {
                let mut command_buffer = self.ctx.command_buffer.borrow_mut();
                self.ctx.executables
                    .embed
                    .encode_lookup(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                for layer in self.ctx.executables.layers.iter() {
                    layer
                        .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                        .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                }
                self.ctx.executables
                    .norm
                    .encode(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.ctx.executables
                    .embed
                    .encode_readout(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.ctx.sampler
                    .encode(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }

            let sampling_output = state.sampling_output().ok_or(Error::GenerateFailed)?;
            let sampling_output_binding = sampling_output.borrow();
            let sampling_output_buffer = sampling_output_binding.buffer();
            let token_ids_binding = self.ctx.scratch_buffers.token_ids.borrow();
            let token_ids_buffer = token_ids_binding.buffer();

            {
                let mut cb = self.ctx.command_buffer.borrow_mut();
                let sampling_output_buffer = sampling_output_buffer.borrow();
                let mut token_ids_buffer = token_ids_buffer.borrow_mut();
                if pass + 1 < followup_count {
                    self.ctx.token_copy_sampled.encode(&*sampling_output_buffer, &mut *token_ids_buffer, &mut *cb);
                }
                let results_offset = pass * std::mem::size_of::<u32>();
                let mut results_buffer = self.ctx.async_chain_results.borrow_mut();
                self.ctx.token_copy_results.encode(
                    &*sampling_output_buffer,
                    (&mut *results_buffer, results_offset),
                    &mut *cb,
                );
            }

            self.ctx.cache_layers.borrow_mut().update_after_acceptance(
                &[0],
                None,
                self.ctx.command_buffer.borrow_mut().deref_mut(),
                &self.ctx.kv_cache_update,
            );
            self.ctx.cache_layers.borrow_mut().register_accepted_tokens(&[self.next_position + pass]);
        }

        self.next_position = self.next_position.saturating_add(followup_count);
        self.submit_and_wait_current_command_buffer()?;

        let results_borrow = self.ctx.async_chain_results.borrow();
        let results_ptr = results_borrow.cpu_ptr().as_ptr() as *const u32;
        for pass in 0..followup_count {
            let sampled = unsafe { *results_ptr.add(pass) };
            on_token(pass, u64::from(sampled))?;
        }
        Ok(())
    }

    pub(super) fn decode_followup_tokens_sequential(
        &mut self,
        mut previous_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for pass in 0..followup_count {
            let sampled = self.decode_next_token(&[previous_token], EmbeddingInjection::None, vocab_limit, sampling)?;
            on_token(pass, sampled)?;
            previous_token = sampled;
        }
        Ok(())
    }

    pub(super) fn prepare_single_token_vocab_mask(
        &mut self,
        vocab_limit: usize,
    ) -> Result<(), Error> {
        let limit = vocab_limit.min(self.ctx.decoder_config.vocab_size);
        if limit == 0 || limit >= self.ctx.decoder_config.vocab_size {
            return Ok(());
        }
        if self.single_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
        let mut mask = vec![0_u32; row_words];
        for token_index in 0..limit {
            let word = token_index / 32;
            let bit = token_index % 32;
            mask[word] |= 1_u32 << bit;
        }
        self.single_token_vocab_masks.insert(limit, mask.into_boxed_slice());
        Ok(())
    }

    pub(super) fn prepare_two_token_vocab_mask(
        &mut self,
        vocab_limit: usize,
    ) -> Result<(), Error> {
        let limit = vocab_limit.min(self.ctx.decoder_config.vocab_size);
        if limit == 0 || limit >= self.ctx.decoder_config.vocab_size {
            return Ok(());
        }
        if self.two_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
        let mut mask = vec![0_u32; row_words.checked_mul(2).ok_or(Error::GenerateFailed)?];
        for token_index in 0..limit {
            let word = token_index / 32;
            let bit = token_index % 32;
            mask[row_words + word] |= 1_u32 << bit;
        }
        self.two_token_vocab_masks.insert(limit, mask.into_boxed_slice());
        Ok(())
    }

    fn get_single_token_vocab_mask(
        &self,
        vocab_limit: usize,
    ) -> Option<&[u32]> {
        self.single_token_vocab_masks.get(&vocab_limit).map(|mask| mask.as_ref())
    }

    fn get_two_token_vocab_mask(
        &self,
        vocab_limit: usize,
    ) -> Option<&[u32]> {
        self.two_token_vocab_masks.get(&vocab_limit).map(|mask| mask.as_ref())
    }

    pub(super) fn decode_next_step(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        mut pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
    ) -> Result<u64, Error> {
        objc2::rc::autoreleasepool(|_| {
            if token_ids.is_empty() {
                return Err(Error::GenerateFailed);
            }

            let token_count = token_ids.len();
            let sampling_start = token_count - 1;
            let sampling_length = 1usize;

            let mut single_position = [0_usize; 1];
            let mut two_positions = [0_usize; 2];
            let positions_storage;
            let positions: &[usize] = if token_count == 1 {
                single_position[0] = self.next_position;
                &single_position
            } else if token_count == 2 {
                two_positions[0] = self.next_position;
                two_positions[1] = self.next_position + 1;
                &two_positions
            } else {
                positions_storage = (self.next_position..self.next_position + token_count).collect::<Vec<_>>();
                positions_storage.as_slice()
            };

            let mut single_seed = [0_u64; 1];
            let mut two_seeds = [0_u64; 2];
            let mut token_seeds_storage;
            let token_seeds: &mut [u64] = if token_count == 1 {
                &mut single_seed
            } else if token_count == 2 {
                &mut two_seeds
            } else {
                token_seeds_storage = vec![0_u64; token_count];
                token_seeds_storage.as_mut_slice()
            };
            if !matches!(sampling.method(), SamplingMethod::Greedy) {
                token_seeds[sampling_start] = sampling.next_seed();
            }

            enum TokenBitmaskSource<'a> {
                None,
                Borrowed(&'a [u32]),
                Owned(Vec<u32>),
            }

            let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
            let token_bitmask_source = if let Some(mask) = precomputed_token_bitmask {
                let expected_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
                if mask.len() != expected_words {
                    return Err(Error::GenerateFailed);
                }
                TokenBitmaskSource::Borrowed(mask)
            } else if let Some(limit_raw) = vocab_limit {
                let limit = limit_raw.min(self.ctx.decoder_config.vocab_size);
                if limit == 0 {
                    return Err(Error::GenerateFailed);
                }
                if limit >= self.ctx.decoder_config.vocab_size {
                    TokenBitmaskSource::None
                } else if token_count == 1 {
                    if let Some(mask) = self.get_single_token_vocab_mask(limit) {
                        TokenBitmaskSource::Borrowed(mask)
                    } else {
                        let mut mask = vec![0_u32; row_words];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[word] |= 1_u32 << bit;
                        }
                        TokenBitmaskSource::Owned(mask)
                    }
                } else if token_count == 2 {
                    if let Some(mask) = self.get_two_token_vocab_mask(limit) {
                        TokenBitmaskSource::Borrowed(mask)
                    } else {
                        let total_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
                        let mut mask = vec![0_u32; total_words];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[sampling_start * row_words + word] |= 1_u32 << bit;
                        }
                        TokenBitmaskSource::Owned(mask)
                    }
                } else {
                    let total_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
                    let mut mask = vec![0_u32; total_words];
                    for token_index in 0..limit {
                        let word = token_index / 32;
                        let bit = token_index % 32;
                        mask[sampling_start * row_words + word] |= 1_u32 << bit;
                    }
                    TokenBitmaskSource::Owned(mask)
                }
            } else {
                TokenBitmaskSource::None
            };

            let token_bitmask: Option<&[u32]> = match &token_bitmask_source {
                TokenBitmaskSource::None => None,
                TokenBitmaskSource::Borrowed(mask) => Some(*mask),
                TokenBitmaskSource::Owned(mask) => Some(mask.as_slice()),
            };

            let mut state = ForwardPassState::new_llm(
                self.ctx.context.clone(),
                &self.ctx.decoder_config,
                &self.ctx.model_shape,
                &self.ctx.scratch_buffers,
                self.ctx.cache_layers.clone(),
                self.ctx.shared_buffers.clone(),
                token_ids,
                positions,
                token_bitmask,
                token_seeds,
                token_count,
                sampling_start,
                sampling_length,
                false,
                None,
                false,
                self.should_fill_attention_bias,
                None,
                None,
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling.method());
            }

            let encoding_parameters = EncodingParameters::new();
            let mut single_accepted = [0_usize; 1];
            let two_accepted = [0_usize, 1_usize];
            let accepted_suffix_indices_storage;
            let accepted_suffix_indices: &[usize] = if token_count == 1 {
                single_accepted[0] = 0;
                &single_accepted
            } else if token_count == 2 {
                &two_accepted
            } else {
                accepted_suffix_indices_storage = (0..token_count).collect::<Vec<_>>();
                accepted_suffix_indices_storage.as_slice()
            };

            if matches!(embedding_injection, EmbeddingInjection::OverrideFirstRowInternal) && capture_hidden {
                return Err(Error::GenerateFailed);
            }
            *self.ctx.command_buffer.borrow_mut() =
                self.ctx.context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
            {
                let mut command_buffer = self.ctx.command_buffer.borrow_mut();
                self.ctx.executables
                    .embed
                    .encode_lookup(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                if let Some(pre_encode) = pre_injection_encode.as_mut() {
                    pre_encode(self, &state, command_buffer.deref_mut())?;
                }
            }
            match embedding_injection {
                EmbeddingInjection::None => {},
                EmbeddingInjection::AddPreloaded {
                    post_scale,
                } => {
                    self.encode_add_scale_from_single_bias(&state, token_count, post_scale.unwrap_or(1.0))?;
                },
                EmbeddingInjection::OverrideFirstRowInternal => {
                    self.encode_override_first_row_from_device(&state, &self.single_override_embedding)?;
                },
            }
            for layer in self.ctx.executables.layers.iter() {
                layer
                    .encode(&mut state, &encoding_parameters, self.ctx.command_buffer.borrow_mut().deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            if capture_hidden {
                self.encode_capture_last_hidden_into_single_buffer(&state, token_count)?;
            }
            self.ctx.executables
                .norm
                .encode(&mut state, self.ctx.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.ctx.executables
                .embed
                .encode_readout(&mut state, self.ctx.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.ctx.sampler
                .encode(&mut state, self.ctx.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.ctx.cache_layers.borrow_mut().update_after_acceptance(
                accepted_suffix_indices,
                None,
                self.ctx.command_buffer.borrow_mut().deref_mut(),
                &self.ctx.kv_cache_update,
            );
            self.submit_and_wait_current_command_buffer()?;
            let token = read_sampled_token_from_sampling_output(&state)?;
            self.ctx.cache_layers.borrow_mut().register_accepted_tokens(positions);
            self.next_position = self.next_position.saturating_add(token_count);
            Ok(token)
        })
    }

    fn encode_capture_last_hidden_into_single_buffer(
        &self,
        state: &ForwardPassState<B>,
        token_count: usize,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let bytes_per_element = main.data_type().size_in_bytes();
        let row_offset = (token_count - 1)
            .checked_mul(model_dim)
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(Error::GenerateFailed)?;
        let src_offset = main.offset().checked_add(row_offset).ok_or(Error::GenerateFailed)?;
        let capture = self.single_hidden_capture.borrow();
        if capture.shape() != [1, model_dim] || capture.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        {
            let mut cb = self.ctx.command_buffer.borrow_mut();
            let main_buffer = main.buffer();
            let main_buffer = main_buffer.borrow();
            let capture_buffer = capture.buffer();
            let mut capture_buffer = capture_buffer.borrow_mut();
            self.tensor_copy.encode((&*main_buffer, src_offset), &mut *capture_buffer, model_dim_u32, &mut *cb);
        }
        Ok(())
    }

    fn encode_override_first_row_from_device(
        &self,
        state: &ForwardPassState<B>,
        override_embedding: &ArrayCell<B>,
    ) -> Result<(), Error> {
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let override_embedding = override_embedding.borrow();
        if override_embedding.shape() != [1, model_dim] || override_embedding.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        {
            let mut cb = self.ctx.command_buffer.borrow_mut();
            let override_buffer = override_embedding.buffer();
            let override_buffer = override_buffer.borrow();
            let main_buffer = main.buffer();
            let mut main_buffer = main_buffer.borrow_mut();
            self.tensor_copy.encode(
                (&*override_buffer, override_embedding.offset()),
                (&mut *main_buffer, main.offset()),
                model_dim_u32,
                &mut *cb,
            );
        }
        Ok(())
    }

    fn encode_add_scale_from_single_bias(
        &self,
        state: &ForwardPassState<B>,
        token_count: usize,
        scale: f32,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let total_len = token_count.checked_mul(model_dim).ok_or(Error::GenerateFailed)?;
        let total_len_u32 = u32::try_from(total_len).map_err(|_| Error::GenerateFailed)?;

        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let bias = self.single_override_embedding.borrow();
        if bias.shape() != [1, model_dim] || bias.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        {
            let mut cb = self.ctx.command_buffer.borrow_mut();
            let bias_buffer = bias.buffer();
            let bias_buffer = bias_buffer.borrow();
            let main_output_buffer = main.buffer();
            let mut main_output_buffer = main_output_buffer.borrow_mut();
            // TensorAddScale is elementwise, so in-place read/write aliasing is valid here.
            let main_input_buffer: &B::Buffer = unsafe { &*((&*main_output_buffer as *const B::Buffer)) };
            self.tensor_add_scale.encode(
                (main_input_buffer, main.offset()),
                &*bias_buffer,
                (&mut *main_output_buffer, main.offset()),
                model_dim_u32,
                total_len_u32,
                scale,
                &mut *cb,
            );
        }
        Ok(())
    }

    fn submit_and_wait_current_command_buffer(&mut self) -> Result<(), Error> {
        let replacement =
            self.ctx.context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
        let command_buffer = {
            let mut command_buffer = self.ctx.command_buffer.borrow_mut();
            std::mem::replace(command_buffer.deref_mut(), replacement)
        };
        self.instrumentation.command_buffers_submitted += 1;
        command_buffer
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| Error::CommandBufferFailed(Box::new(err)))?;
        self.instrumentation.host_waits += 1;
        Ok(())
    }

    pub(super) fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
    }

    pub(super) fn clear_instrumentation(&mut self) {
        self.instrumentation = RunnerInstrumentation::default();
    }
}

fn read_sampled_token_from_sampling_output<B: Backend>(state: &ForwardPassState<B>) -> Result<u64, Error> {
    let output = state.sampling_output().ok_or(Error::GenerateFailed)?;
    let output = output.borrow();
    let tokens = output.as_slice::<u32>();
    let token = tokens.first().copied().ok_or(Error::GenerateFailed)?;
    Ok(u64::from(token))
}

pub(super) fn load_stub_seed(weights_path: PathBuf) -> Option<u64> {
    let file = File::open(weights_path).ok()?;
    let (global_offset, metadata) = read_safetensors_metadata(&file).ok()?;
    let tensor = metadata.tensors.get("text_decoder.seed")?;

    let (begin, end) = tensor.data_offsets;
    let size = end.checked_sub(begin)?;
    let data_type: DataType = tensor.dtype.into();
    let offset = global_offset.checked_add(begin)?;

    match data_type {
        DataType::I32 if size == 4 => {
            let mut bytes = [0_u8; 4];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            let value = i32::from_le_bytes(bytes);
            (value >= 0).then_some(value as u64)
        },
        DataType::I64 if size == 8 => {
            let mut bytes = [0_u8; 8];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            let value = i64::from_le_bytes(bytes);
            (value >= 0).then_some(value as u64)
        },
        DataType::U64 if size == 8 => {
            let mut bytes = [0_u8; 8];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            Some(u64::from_le_bytes(bytes))
        },
        _ => None,
    }
}

pub(super) fn generate_stub_tokens(
    num_codebooks: usize,
    frames: usize,
    token_upper_bound: usize,
    seed: u64,
) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = Vec::with_capacity(num_codebooks * frames);
    for _codebook in 0..num_codebooks {
        for _frame in 0..frames {
            tokens.push(rng.random_range(0..token_upper_bound) as u32);
        }
    }
    tokens
}

pub(super) fn generate_stub_semantic_grid(
    stub: &StubTextDecoderRuntime,
    text_tokens: &[u64],
    codec_cardinality: usize,
    seed: u64,
    max_semantic_frames: usize,
) -> Result<AudioTokenGrid, Error> {
    let frames = text_tokens.len().min(max_semantic_frames.max(1));
    let token_upper_bound = stub.codebook_size.min(codec_cardinality);
    if token_upper_bound == 0 {
        return Err(Error::UnableToLoadConfig);
    }

    let tokens = generate_stub_tokens(stub.num_codebooks, frames, token_upper_bound, seed);
    AudioTokenGrid::new(
        tokens.into_boxed_slice(),
        1,
        stub.num_codebooks,
        frames,
        vec![frames].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .map_err(Error::from)
}
