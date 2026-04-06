use super::{decoder_support::*, *};
use crate::{array::Array, backends::common::Buffer, session::types::TtsModelConfigError};

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
        max_suffix_length: usize,
    ) -> Result<Self, Error> {
        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = TokenDecoderContext::<B>::build_shared_buffers(
            context,
            decoder_config,
            model_shape,
            &root_loader_view,
            transformer_subtree,
        )?;
        let scratch_buffers = ScratchBuffers::new(context.as_ref(), decoder_config, model_shape, max_suffix_length);
        let executables = Decoder::new_with_embedding_and_readout_subtrees(
            context.as_ref(),
            &decoder_config,
            &root_loader_view,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
        );
        let logits_data_type = scratch_buffers.logits.data_type();
        let sampler =
            GpuSampling::new(context.as_ref(), logits_data_type, max_suffix_length, decoder_config.vocab_size)
                .map_err(unable_to_create_context)?;
        let token_copy_sampled =
            <B::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref()).map_err(unable_to_create_context)?;
        let token_copy_results = <B::Kernels as Kernels>::TokenCopyToResultsKernel::new(context.as_ref())
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
    ) -> Result<(Self, DataType), Error> {
        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = text_decoder_prefill_step_size(runtime_config, decoder_config.context_length).max(32);
        let activation_data_type = model_shape.activation_data_type();
        let loaded_model = TokenDecoderLoadedModel::<B>::load(
            &context,
            model_path,
            &decoder_config,
            &model_shape,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
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
    pub(super) single_hidden_capture: Array<B>,
    pub(super) single_override_embedding: Array<B>,
    tensor_copy: <B::Kernels as Kernels>::TensorCopyKernel,
    tensor_add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
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
        let (ctx, activation_data_type) = TokenDecoderContext::new(
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
        let tensor_copy = <B::Kernels as Kernels>::TensorCopyKernel::new(context.as_ref(), activation_data_type)
            .map_err(unable_to_create_context)?;
        let tensor_add_scale =
            <B::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let single_hidden_capture =
            context.create_array_zeros(&[1, model_dim], activation_data_type, "tts_single_hidden_capture");
        let single_override_embedding =
            context.create_array_zeros(&[1, model_dim], activation_data_type, "tts_single_override_embedding");

        Ok(Self {
            ctx,
            single_hidden_capture,
            single_override_embedding,
            tensor_copy,
            tensor_add_scale,
            single_token_vocab_masks: HashMap::new(),
            two_token_vocab_masks: HashMap::new(),
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

        objc2::rc::autoreleasepool(|_| {
            let token_count = token_ids.len();

            let positions: Vec<usize> = (self.next_position..self.next_position + token_count).collect();
            let token_seeds = vec![0_u64; token_count];

            let mut state = ForwardPassState::new_llm(
                self.ctx.context.clone(),
                &self.ctx.decoder_config,
                &self.ctx.model_shape,
                &self.ctx.scratch_buffers,
                self.ctx.cache_layers.clone(),
                self.ctx.shared_buffers.clone(),
                token_ids,
                None,
                &positions,
                None, // no bitmask
                &token_seeds,
                token_count,
                0, // sampling_start: irrelevant
                0, // sampling_length: no sampling
                false,
                false,
                None,
                None,
            );

            let encoding_parameters = EncodingParameters::new();

            let context = Rc::clone(&self.ctx.context);
            let mut encoder = Encoder::new(context.as_ref()).map_err(unable_to_create_context)?;

            self.ctx
                .executables
                .embed
                .encode_lookup(&mut state, &mut encoder)
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            for layer in self.ctx.executables.layers.iter() {
                layer
                    .encode(&mut state, &encoding_parameters, &mut encoder)
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            self.encode_cache_acceptance_update_on(&mut encoder, token_count);

            self.submit_and_wait_command_buffer(encoder)?;
            self.register_positions_and_advance(token_count);
            Ok(())
        })
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

    /// Encode the first fast token plus followup passes onto the given
    /// command buffer, without submitting or reading results.  The caller
    /// must submit the command buffer later (potentially after encoding
    /// more work from another runner).
    ///
    /// Returns `total_count = 1 + followup_count`.  After the eventual
    /// submit+wait the results live in `async_chain_results[0..total_count]`.
    pub(super) fn encode_first_and_followup_tokens_on(
        &mut self,
        encoder: &mut Encoder<B>,
        initial_token_ids: &[u64],
        initial_embedding_injection: EmbeddingInjection,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
    ) -> Result<usize, Error> {
        let total_count =
            1_usize.checked_add(followup_count).ok_or(TtsModelConfigError::AsyncChainFollowupCountOverflow)?;
        if total_count > self.ctx.async_chain_capacity {
            return Err(TtsModelConfigError::AsyncChainTokenCountExceedsCapacity {
                total_count,
                capacity: self.ctx.async_chain_capacity,
            }
            .into());
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

        let initial_token_count = initial_token_ids.len();
        if initial_token_count == 0 {
            return Err(Error::GenerateFailed);
        }

        // Resolve the bitmask into a local copy so that it does not borrow
        // from `self` when we later call `encode_single_forward_pass`.
        let initial_token_bitmask_owned: Option<Box<[u32]>> = if let Some(limit) = vocab_mask_limit {
            if initial_token_count == 1 {
                self.get_single_token_vocab_mask(limit).map(|m| m.into())
            } else if initial_token_count == 2 {
                self.get_two_token_vocab_mask(limit).map(|m| m.into())
            } else {
                return Err(TtsModelConfigError::UnsupportedInitialTokenCountForVocabMask {
                    initial_token_count,
                }
                .into());
            }
        } else {
            None
        };
        let initial_token_bitmask: Option<&[u32]> = initial_token_bitmask_owned.as_deref();

        // Consume the initial seed FIRST to preserve RNG ordering with the
        // old two-submit path (initial token consumed its seed before followups).
        let initial_seed = if !matches!(sampling.method(), SamplingMethod::Greedy) {
            Some(sampling.next_seed())
        } else {
            None
        };

        // Pre-fill followup positions and seeds into the async chain buffers.
        let followup_base_position = self.next_position + initial_token_ids.len();
        {
            let positions_borrow = self.ctx.async_chain_positions.borrow();
            let positions_ptr = positions_borrow.cpu_ptr().as_ptr() as *mut i32;
            for pass in 0..followup_count {
                unsafe {
                    *positions_ptr.add(pass) = (followup_base_position + pass) as i32;
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

        // Encode the initial forward pass (no new CB, no submit).
        let state = self.encode_single_forward_pass_on(
            encoder,
            initial_token_ids,
            initial_embedding_injection,
            sampling,
            initial_token_bitmask,
            false, // capture_hidden
            pre_injection_encode,
            initial_seed,
        )?;

        // Copy the sampled token for chaining and into results[0].
        {
            let sampling_output = state.sampling_output().ok_or(Error::SamplingFailed)?;
            let sampling_output_buffer = sampling_output.buffer();
            let token_ids_buffer = self.ctx.scratch_buffers.token_ids.buffer();

            let sampling_output_buffer = sampling_output_buffer.borrow();
            let mut token_ids_buffer = token_ids_buffer.borrow_mut();
            if followup_count > 0 {
                self.ctx.token_copy_sampled.encode(&*sampling_output_buffer, &mut *token_ids_buffer, encoder);
            }
            let mut results_buffer = self.ctx.async_chain_results.borrow_mut();
            self.ctx.token_copy_results.encode(&*sampling_output_buffer, (&mut *results_buffer, 0), encoder);
        }

        self.encode_cache_acceptance_update_on(encoder, initial_token_count);
        self.register_positions_and_advance(initial_token_count);

        if followup_count > 0 {
            self.encode_followup_passes_on(encoder, followup_count, 0, vocab_mask_limit, sampling.method(), 1)?;
        }

        self.next_position = followup_base_position.saturating_add(followup_count);
        // Do NOT submit -- the caller will submit after encoding more work.
        Ok(total_count)
    }

    /// Encode `followup_count` autoregressive passes onto the current command
    /// buffer (which must already be open). `first_token` is the embedding
    /// input for the first pass; subsequent passes read from the
    /// `token_copy_sampled` output of the preceding pass.
    ///
    /// `results_offset_slots` is the slot index in `async_chain_results` where
    /// the first pass's result should be written (0 for standalone, 1+ for
    /// merged paths).
    fn encode_followup_passes_on(
        &mut self,
        encoder: &mut Encoder<B>,
        followup_count: usize,
        first_token: u64,
        vocab_mask_limit: Option<usize>,
        sampling_method: SamplingMethod,
        results_offset_slots: usize,
    ) -> Result<(), Error> {
        let encoding_parameters = EncodingParameters::new();
        let context_rc = self.ctx.context.clone();
        let cache_layers_rc = self.ctx.cache_layers.clone();
        let shared_buffers_rc = self.ctx.shared_buffers.clone();
        let positions_rc = self.ctx.async_chain_positions.clone();
        let seeds_rc = self.ctx.async_chain_seeds.clone();
        let token_bitmask = vocab_mask_limit.and_then(|limit| self.get_single_token_vocab_mask(limit));

        for pass in 0..followup_count {
            let results_slot = results_offset_slots + pass;
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
                None,
                &[self.next_position + pass],
                token_bitmask,
                &[0],
                1,
                0,
                1,
                false,
                pass > 0 || results_offset_slots > 0,
                Some((positions_rc.clone(), pass)),
                Some((seeds_rc.clone(), pass)),
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling_method);
            }

            {
                self.ctx
                    .executables
                    .embed
                    .encode_lookup(&mut state, encoder)
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                for layer in self.ctx.executables.layers.iter() {
                    layer
                        .encode(&mut state, &encoding_parameters, encoder)
                        .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                }
                self.ctx
                    .executables
                    .norm
                    .encode(&mut state, encoder)
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.ctx
                    .executables
                    .embed
                    .encode_readout(&mut state, encoder)
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.ctx.sampler.encode(&mut state, encoder).map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }

            let sampling_output = state.sampling_output().ok_or(Error::SamplingFailed)?;
            let sampling_output_buffer = sampling_output.buffer();
            let token_ids_buffer = self.ctx.scratch_buffers.token_ids.buffer();

            {
                let sampling_output_buffer = sampling_output_buffer.borrow();
                let mut token_ids_buffer = token_ids_buffer.borrow_mut();
                if pass + 1 < followup_count {
                    self.ctx.token_copy_sampled.encode(&*sampling_output_buffer, &mut *token_ids_buffer, encoder);
                }
                let results_offset = results_slot * std::mem::size_of::<u32>();
                let mut results_buffer = self.ctx.async_chain_results.borrow_mut();
                self.ctx.token_copy_results.encode(
                    &*sampling_output_buffer,
                    (&mut *results_buffer, results_offset),
                    encoder,
                );
            }

            self.ctx.cache_layers.borrow_mut().update_after_acceptance(&[0], None, encoder, &self.ctx.kv_cache_update);
            self.ctx.cache_layers.borrow_mut().register_accepted_tokens(1);
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
        let mut mask = vec![
            0_u32;
            row_words.checked_mul(2).ok_or(TtsModelConfigError::TwoTokenVocabMaskSizeOverflow {
                row_words
            })?
        ];
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

    fn encode_single_forward_pass_on(
        &mut self,
        encoder: &mut Encoder<B>,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        mut pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
        preconsumed_seed: Option<u64>,
    ) -> Result<ForwardPassState<B>, Error> {
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
            token_seeds[sampling_start] = preconsumed_seed.unwrap_or_else(|| sampling.next_seed());
        }

        if let Some(mask) = token_bitmask {
            let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
            let expected_words =
                token_count.checked_mul(row_words).ok_or(TtsModelConfigError::TokenBitmaskSizeOverflow)?;
            if mask.len() != expected_words {
                return Err(TtsModelConfigError::TokenBitmaskLengthMismatch {
                    actual_words: mask.len(),
                    expected_words,
                    token_count,
                    row_words,
                }
                .into());
            }
        }

        if matches!(embedding_injection, EmbeddingInjection::OverrideFirstRowInternal) && capture_hidden {
            return Err(Error::GenerateFailed);
        }

        let mut state = ForwardPassState::new_llm(
            self.ctx.context.clone(),
            &self.ctx.decoder_config,
            &self.ctx.model_shape,
            &self.ctx.scratch_buffers,
            self.ctx.cache_layers.clone(),
            self.ctx.shared_buffers.clone(),
            token_ids,
            None,
            positions,
            token_bitmask,
            token_seeds,
            token_count,
            sampling_start,
            sampling_length,
            false,
            false,
            None,
            None,
        );
        if let Some(method) = state.sampling_method_mut() {
            *method = Some(sampling.method());
        }

        let encoding_parameters = EncodingParameters::new();
        self.ctx
            .executables
            .embed
            .encode_lookup(&mut state, encoder)
            .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        if let Some(pre_encode) = pre_injection_encode.as_mut() {
            pre_encode(self, &state, encoder)?;
        }
        match embedding_injection {
            EmbeddingInjection::None => {},
            EmbeddingInjection::AddPreloaded {
                post_scale,
            } => {
                self.encode_add_scale_from_single_bias_on(encoder, &state, token_count, post_scale.unwrap_or(1.0))?;
            },
            EmbeddingInjection::OverrideFirstRowInternal => {
                self.encode_override_first_row_from_device_on(encoder, &state, &self.single_override_embedding)?;
            },
        }
        for layer in self.ctx.executables.layers.iter() {
            layer
                .encode(&mut state, &encoding_parameters, encoder)
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        }
        self.ctx.executables.norm.encode(&mut state, encoder).map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        if capture_hidden {
            // After output norm with residual_add, Shortcut holds the un-normalized full residual
            self.encode_capture_last_hidden_into_single_buffer_on(encoder, &state, token_count)?;
        }
        self.ctx
            .executables
            .embed
            .encode_readout(&mut state, encoder)
            .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        self.ctx.sampler.encode(&mut state, encoder).map_err(|err| Error::EncodeFailed(Box::new(err)))?;

        Ok(state)
    }

    /// Encode the KV-cache acceptance update for `token_count` tokens onto
    /// the current command buffer.
    fn encode_cache_acceptance_update_on(
        &mut self,
        encoder: &mut Encoder<B>,
        token_count: usize,
    ) {
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
        self.ctx.cache_layers.borrow_mut().update_after_acceptance(
            accepted_suffix_indices,
            None,
            encoder,
            &self.ctx.kv_cache_update,
        );
    }

    /// Register accepted token positions in the KV cache and advance
    /// `next_position` by `token_count`.
    fn register_positions_and_advance(
        &mut self,
        token_count: usize,
    ) {
        let positions_storage;
        let positions: &[usize] = if token_count == 1 {
            &[self.next_position]
        } else if token_count == 2 {
            positions_storage = [self.next_position, self.next_position + 1];
            &positions_storage[..]
        } else {
            self.ctx.cache_layers.borrow_mut().register_accepted_tokens(token_count);
            self.next_position = self.next_position.saturating_add(token_count);
            return;
        };
        self.ctx.cache_layers.borrow_mut().register_accepted_tokens(positions.len());
        self.next_position = self.next_position.saturating_add(token_count);
    }

    pub(super) fn decode_next_step(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
    ) -> Result<u64, Error> {
        objc2::rc::autoreleasepool(|_| {
            if token_ids.is_empty() {
                return Err(Error::GenerateFailed);
            }

            let token_count = token_ids.len();

            // Resolve the token bitmask into a local copy so that it does not
            // borrow from `self` when we later call `encode_single_forward_pass`.
            let sampling_start = token_count - 1;
            let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
            let token_bitmask_owned: Option<Box<[u32]>> = if let Some(mask) = precomputed_token_bitmask {
                let expected_words =
                    token_count.checked_mul(row_words).ok_or(TtsModelConfigError::PrecomputedBitmaskSizeOverflow)?;
                if mask.len() != expected_words {
                    return Err(TtsModelConfigError::PrecomputedBitmaskLengthMismatch {
                        actual_words: mask.len(),
                        expected_words,
                    }
                    .into());
                }
                Some(mask.into())
            } else if let Some(limit_raw) = vocab_limit {
                let limit = limit_raw.min(self.ctx.decoder_config.vocab_size);
                if limit == 0 {
                    return Err(TtsModelConfigError::VocabLimitResolvedToZero.into());
                }
                if limit >= self.ctx.decoder_config.vocab_size {
                    None
                } else if token_count == 1 {
                    if let Some(mask) = self.get_single_token_vocab_mask(limit) {
                        Some(mask.into())
                    } else {
                        let mut mask = vec![0_u32; row_words];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[word] |= 1_u32 << bit;
                        }
                        Some(mask.into_boxed_slice())
                    }
                } else if token_count == 2 {
                    if let Some(mask) = self.get_two_token_vocab_mask(limit) {
                        Some(mask.into())
                    } else {
                        let total_words = token_count.checked_mul(row_words).ok_or(
                            TtsModelConfigError::InlineBitmaskSizeOverflow {
                                token_count,
                                row_words,
                            },
                        )?;
                        let mut mask = vec![0_u32; total_words];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[sampling_start * row_words + word] |= 1_u32 << bit;
                        }
                        Some(mask.into_boxed_slice())
                    }
                } else {
                    // token_count > 2 with vocab_limit requires a precomputed bitmask;
                    // silently dropping the limit would produce incorrect sampling.
                    return Err(TtsModelConfigError::VocabLimitRequiresPrecomputedBitmask {
                        token_count,
                    }
                    .into());
                }
            } else {
                None
            };

            let token_bitmask: Option<&[u32]> = token_bitmask_owned.as_deref();

            let context = Rc::clone(&self.ctx.context);
            let mut encoder = Encoder::new(context.as_ref()).map_err(unable_to_create_context)?;

            let state = self.encode_single_forward_pass_on(
                &mut encoder,
                token_ids,
                embedding_injection,
                sampling,
                token_bitmask,
                capture_hidden,
                pre_injection_encode,
                None,
            )?;
            self.encode_cache_acceptance_update_on(&mut encoder, token_count);

            self.submit_and_wait_command_buffer(encoder)?;
            let token = read_sampled_token_from_sampling_output(&state)?;
            self.register_positions_and_advance(token_count);
            Ok(token)
        })
    }

    fn encode_capture_last_hidden_into_single_buffer_on(
        &self,
        encoder: &mut Encoder<B>,
        state: &ForwardPassState<B>,
        token_count: usize,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| TtsModelConfigError::ModelDimExceedsU32 {
            model_dim,
        })?;
        let shortcut = state.array(ArrayId::Shortcut);
        let bytes_per_element = shortcut.data_type().size_in_bytes();
        let row_offset = (token_count - 1)
            .checked_mul(model_dim)
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(TtsModelConfigError::HiddenCaptureRowOffsetOverflow {
                token_count,
                model_dim,
            })?;
        let src_offset =
            shortcut.offset().checked_add(row_offset).ok_or(TtsModelConfigError::HiddenCaptureSourceOffsetOverflow)?;
        let capture = &self.single_hidden_capture;
        if capture.shape() != [1, model_dim] || capture.data_type() != shortcut.data_type() {
            return Err(TtsModelConfigError::HiddenCaptureTensorMismatch {
                expected_shape: [1, model_dim].into(),
                expected_data_type: shortcut.data_type(),
                actual_shape: capture.shape().into(),
                actual_data_type: capture.data_type(),
            }
            .into());
        }

        let shortcut_buffer = shortcut.buffer();
        let shortcut_buffer = shortcut_buffer.borrow();
        let capture_buffer = capture.buffer();
        let mut capture_buffer = capture_buffer.borrow_mut();
        self.tensor_copy.encode((&*shortcut_buffer, src_offset), &mut *capture_buffer, model_dim_u32, encoder);
        Ok(())
    }

    fn encode_override_first_row_from_device_on(
        &self,
        encoder: &mut Encoder<B>,
        state: &ForwardPassState<B>,
        override_embedding: &Array<B>,
    ) -> Result<(), Error> {
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| TtsModelConfigError::ModelDimExceedsU32 {
            model_dim,
        })?;
        let main = state.array(ArrayId::Main);
        if override_embedding.shape() != [1, model_dim] || override_embedding.data_type() != main.data_type() {
            return Err(TtsModelConfigError::OverrideEmbeddingTensorMismatch {
                expected_shape: [1, model_dim].into(),
                expected_data_type: main.data_type(),
                actual_shape: override_embedding.shape().into(),
                actual_data_type: override_embedding.data_type(),
            }
            .into());
        }

        let override_buffer = override_embedding.buffer();
        let override_buffer = override_buffer.borrow();
        let main_buffer = main.buffer();
        let mut main_buffer = main_buffer.borrow_mut();
        self.tensor_copy.encode(
            (&*override_buffer, override_embedding.offset()),
            (&mut *main_buffer, main.offset()),
            model_dim_u32,
            encoder,
        );
        Ok(())
    }

    fn encode_add_scale_from_single_bias_on(
        &self,
        encoder: &mut Encoder<B>,
        state: &ForwardPassState<B>,
        token_count: usize,
        scale: f32,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| TtsModelConfigError::ModelDimExceedsU32 {
            model_dim,
        })?;
        let total_len = token_count.checked_mul(model_dim).ok_or(TtsModelConfigError::AddScaleTotalLengthOverflow {
            token_count,
            model_dim,
        })?;
        let total_len_u32 =
            u32::try_from(total_len).map_err(|_| TtsModelConfigError::AddScaleTotalLengthExceedsU32 {
                total_len,
            })?;

        let main = state.array(ArrayId::Main);
        let bias = &self.single_override_embedding;
        if bias.shape() != [1, model_dim] || bias.data_type() != main.data_type() {
            return Err(TtsModelConfigError::AddScaleBiasTensorMismatch {
                expected_shape: [1, model_dim].into(),
                expected_data_type: main.data_type(),
                actual_shape: bias.shape().into(),
                actual_data_type: bias.data_type(),
            }
            .into());
        }

        let bias_buffer = bias.buffer();
        let bias_buffer = bias_buffer.borrow();
        let main_output_buffer = main.buffer();
        let mut main_output_buffer = main_output_buffer.borrow_mut();
        // TensorAddScale is elementwise, so in-place read/write aliasing is valid here.
        let main_input_buffer: &B::Buffer = unsafe { &*(&*main_output_buffer as *const B::Buffer) };
        self.tensor_add_scale.encode(
            (main_input_buffer, main.offset()),
            &*bias_buffer,
            (&mut *main_output_buffer, main.offset()),
            model_dim_u32,
            total_len_u32,
            scale,
            encoder,
        );
        Ok(())
    }

    /// Encode a single decode step (embedding, transformer, readout, sampling,
    /// and optionally hidden capture) onto the **current** command buffer.
    /// Does NOT create a new command buffer, does NOT submit.
    ///
    /// The caller is responsible for submitting the command buffer and reading
    /// the sampled token from `async_chain_results[0]` after the eventual
    /// submit.
    pub(super) fn encode_next_step_on(
        &mut self,
        encoder: &mut Encoder<B>,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_, B>>,
    ) -> Result<(), Error> {
        let token_count = token_ids.len();

        let state = self.encode_single_forward_pass_on(
            encoder,
            token_ids,
            embedding_injection,
            sampling,
            precomputed_token_bitmask,
            capture_hidden,
            pre_injection_encode,
            None,
        )?;

        // Copy sampled token to async_chain_results[0] so the caller can
        // read it after the eventual submit.
        {
            let sampling_output = state.sampling_output().ok_or(Error::SamplingFailed)?;
            let sampling_output_buffer = sampling_output.buffer();

            let sampling_output_buffer = sampling_output_buffer.borrow();
            let mut results_buffer = self.ctx.async_chain_results.borrow_mut();
            self.ctx.token_copy_results.encode(&*sampling_output_buffer, (&mut *results_buffer, 0), encoder);
        }

        self.encode_cache_acceptance_update_on(encoder, token_count);
        // Do NOT submit -- the caller will submit.
        self.register_positions_and_advance(token_count);
        Ok(())
    }

    pub(super) fn submit_and_wait_command_buffer(
        &mut self,
        encoder: Encoder<B>,
    ) -> Result<(), Error> {
        self.instrumentation.command_buffers_submitted += 1;
        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| Error::CommandBufferFailed(Box::new(err)))?;
        self.instrumentation.host_waits += 1;
        Ok(())
    }

    pub(super) fn copy_async_chain_results_to_on(
        &self,
        encoder: &mut Encoder<B>,
        src_slot: usize,
        dst: (&mut B::Buffer, usize),
        count: usize,
    ) -> Result<(), Error> {
        let src_offset = src_slot.checked_mul(std::mem::size_of::<u32>()).ok_or(
            TtsModelConfigError::AsyncChainCopySourceOffsetOverflow {
                src_slot,
            },
        )?;
        let copy_size =
            count.checked_mul(std::mem::size_of::<u32>()).ok_or(TtsModelConfigError::AsyncChainCopySizeOverflow {
                count,
            })?;
        let results_buffer = self.ctx.async_chain_results.borrow();
        encoder.encode_copy(&results_buffer, src_offset..src_offset + copy_size, dst.0, dst.1..dst.1 + copy_size);
        Ok(())
    }

    pub(super) fn read_async_chain_result(
        &self,
        slot: usize,
    ) -> Result<u32, Error> {
        let results_buffer = self.ctx.async_chain_results.borrow();
        let ptr = results_buffer.cpu_ptr().as_ptr() as *const u32;
        let capacity = self.ctx.async_chain_capacity;
        if slot >= capacity {
            return Err(TtsModelConfigError::AsyncChainResultSlotOutOfBounds {
                slot,
                capacity,
            }
            .into());
        }
        Ok(unsafe { *ptr.add(slot) })
    }

    pub(super) fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
    }

    pub(super) fn clear_instrumentation(&mut self) {
        self.instrumentation = RunnerInstrumentation::default();
    }
}

fn read_sampled_token_from_sampling_output<B: Backend>(state: &ForwardPassState<B>) -> Result<u64, Error> {
    let output = state.sampling_output().ok_or(Error::SamplingFailed)?;
    let tokens = output.as_slice::<u32>();
    let token = tokens.first().copied().ok_or(Error::SamplingFailed)?;
    Ok(u64::from(token))
}
