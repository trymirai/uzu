use super::{decoder_support::*, *};
use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{Allocation, Buffer},
    encodable_block::{DecoderArguments, DecoderDecodeInput, SamplingInputs},
    forward_pass::token_inputs::TokenInputs,
    session::types::TtsModelConfigError,
};

struct TokenDecoderLoadedModel<B: Backend> {
    shared_buffers: Rc<SharedBuffers<B>>,
    executables: Decoder<B>,
    sampler: GpuSampling<B>,
    token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
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
        let executables = Decoder::new_with_embedding_and_readout_subtrees(
            context.as_ref(),
            &decoder_config,
            &root_loader_view,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
        );
        let logits_data_type = model_shape.activation_data_type();
        let sampler = GpuSampling::new(context.as_ref(), logits_data_type).map_err(unable_to_create_context)?;
        let token_copy_sampled =
            <B::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref()).map_err(unable_to_create_context)?;

        Ok(Self {
            shared_buffers,
            executables,
            sampler,
            token_copy_sampled,
        })
    }
}

struct TokenDecoderContext<B: Backend> {
    context: Rc<B::Context>,
    cache_layers: Rc<RefCell<CacheLayers<B>>>,
    shared_buffers: Rc<SharedBuffers<B>>,
    model_shape: ModelShape,
    decoder_config: Rc<crate::config::DecoderConfig>,
    runtime_config: TextDecoderRuntimeConfig,
    executables: Decoder<B>,
    sampler: GpuSampling<B>,
    kv_cache_update: KVCacheUpdate<B>,
    token_copy_sampled: <B::Kernels as Kernels>::TokenCopySampledKernel,
    async_chain_positions: Array<B>,
    async_chain_seeds: Array<B>,
    async_chain_results: Allocation<B>,
    async_chain_capacity: usize,
    current_max_prefix_length: usize,
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
        let max_suffix_length = text_decoder_prefill_step_size(runtime_config, decoder_config.context_length).max(32);
        let max_prefix_length = max_suffix_length;
        let activation_data_type = model_shape.activation_data_type();
        let loaded_model = TokenDecoderLoadedModel::<B>::load(
            &context,
            model_path,
            &decoder_config,
            &model_shape,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
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
                model_shape,
                decoder_config,
                runtime_config: runtime_config.clone(),
                executables: loaded_model.executables,
                sampler: loaded_model.sampler,
                kv_cache_update,
                token_copy_sampled: loaded_model.token_copy_sampled,
                async_chain_positions,
                async_chain_seeds,
                async_chain_results,
                async_chain_capacity,
                current_max_prefix_length: max_prefix_length,
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
    ) -> Result<Rc<SharedBuffers<B>>, Error> {
        let mut shared_buffers = SharedBuffers::new(context.as_ref(), decoder_config, model_shape);
        let transformer_tree = root_loader_view.subtree(transformer_subtree).map_err(|_| Error::UnableToLoadWeights)?;
        if let Some(global_rope) = &mut shared_buffers.global_rope {
            global_rope.update_data(&transformer_tree, "global_rope");
        }
        if let Some(local_rope) = &mut shared_buffers.local_rope {
            local_rope.update_data(&transformer_tree, "local_rope");
        }
        Ok(Rc::new(shared_buffers))
    }

    fn build_async_chain_buffers(
        context: &Rc<B::Context>,
        async_chain_capacity: usize,
    ) -> Result<(Array<B>, Array<B>, Allocation<B>), Error> {
        let positions =
            context.create_array_uninitialized(&[async_chain_capacity], crate::DataType::I32, "async_positions");
        let seeds = context.create_array_uninitialized(&[async_chain_capacity], crate::DataType::U64, "async_seeds");
        let results = context
            .create_array_uninitialized(&[async_chain_capacity], crate::DataType::U32, "async_results")
            .into_allocation();
        Ok((positions, seeds, results))
    }
}

pub(super) struct TokenDecoderRunner<B: Backend> {
    ctx: TokenDecoderContext<B>,
    pub(super) single_hidden_capture: Allocation<B>,
    pub(super) single_override_embedding: Allocation<B>,
    tensor_add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    next_position: usize,
    instrumentation: RunnerInstrumentation,
}

impl<B: Backend> TokenDecoderRunner<B> {
    pub(super) fn activation_data_type(&self) -> DataType {
        self.ctx.model_shape.activation_data_type()
    }

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
        let tensor_add_scale =
            <B::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let single_hidden_capture = context
            .create_array_zeros(&[1, model_dim], activation_data_type, "tts_single_hidden_capture")
            .into_allocation();
        let single_override_embedding = context
            .create_array_zeros(&[1, model_dim], activation_data_type, "tts_single_override_embedding")
            .into_allocation();

        Ok(Self {
            ctx,
            single_hidden_capture,
            single_override_embedding,
            tensor_add_scale,
            single_token_vocab_masks: HashMap::new(),
            two_token_vocab_masks: HashMap::new(),
            next_position: 0,
            instrumentation: RunnerInstrumentation::default(),
        })
    }

    pub(super) fn reset(&mut self) {
        self.ctx.cache_layers.borrow_mut().clear(self.ctx.context.as_ref());
        self.next_position = 0;
    }

    fn create_sampling_inputs_from_slices(
        &self,
        token_seeds: &[u64],
        token_bitmask: Option<&[u32]>,
    ) -> SamplingInputs<B> {
        let bitmask_row_len = token_bitmask.map(|_| self.ctx.decoder_config.vocab_size.div_ceil(32));
        SamplingInputs::from_slices(self.ctx.context.as_ref(), token_seeds, token_bitmask, bitmask_row_len)
    }

    fn llm_token_inputs(
        &self,
        token_ids: &[u64],
        token_positions: &[usize],
        token_ids_array: Option<Array<B>>,
        token_positions_array: Option<Array<B>>,
        sampling_start: usize,
        sampling_length: usize,
    ) -> TokenInputs<B> {
        TokenInputs::new_llm(
            self.ctx.context.as_ref(),
            &self.ctx.model_shape,
            token_ids,
            None,
            token_positions,
            token_ids_array,
            token_positions_array,
            sampling_start,
            sampling_length,
        )
    }

    fn decoder_arguments<'a>(
        &'a self,
        token_inputs: &'a TokenInputs<B>,
        batch_dim: usize,
        sampling_start: usize,
        sampling_length: usize,
        cache_layers: &'a mut CacheLayers<B>,
    ) -> DecoderArguments<'a, B> {
        DecoderArguments {
            token_positions: token_inputs.token_positions(),
            token_parents: token_inputs.token_parents(),
            token_subtrie_ranges: token_inputs.token_subtrie_ranges(),
            shared_buffers: self.ctx.shared_buffers.as_ref(),
            cache_layers: Some(cache_layers),
            batch_dim,
            sampling_start,
            sampling_length,
            rope_max_sequence_length: self.ctx.model_shape.context_length(),
            rope_dim: self.ctx.model_shape.rope_dim(),
            #[cfg(feature = "tracing")]
            trace: None,
        }
    }

    fn encode_sampling_on(
        &self,
        encoder: &mut Encoder<B>,
        logits: &mut Allocation<B>,
        output: &mut Allocation<B>,
        sampling_method: SamplingMethod,
        batch_size: usize,
        sampling_inputs: SamplingInputs<B>,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
    ) -> Result<(), Error> {
        pending_sampling_inputs.push(sampling_inputs);
        let sampling_inputs = pending_sampling_inputs.last().expect("sampling inputs must be pending");
        self.ctx
            .sampler
            .encode(
                crate::encodable_block::SamplingArguments {
                    logits,
                    seeds: sampling_inputs.seeds.allocation(),
                    bitmask: sampling_inputs.bitmask.as_ref().map(Array::allocation),
                    output,
                    sampling_method,
                    batch_size,
                    vocab_size: self.ctx.decoder_config.vocab_size,
                },
                encoder,
            )
            .map_err(|err| Error::EncodeFailed(Box::new(err)))
    }

    fn copy_sampling_output_to_async_result_on(
        &mut self,
        encoder: &mut Encoder<B>,
        sampling_output: &Allocation<B>,
        result_slot: usize,
    ) {
        let copy_size = std::mem::size_of::<u32>();
        let result_offset = result_slot * copy_size;
        encoder.encode_copy(
            sampling_output,
            0..copy_size,
            &mut self.ctx.async_chain_results,
            result_offset..result_offset + copy_size,
        );
    }

    pub(super) fn prepare_for_generation(
        &mut self,
        max_prefix_length: usize,
    ) -> Result<(), Error> {
        let max_prefix_length = max_prefix_length.max(1).min(self.ctx.decoder_config.context_length);
        let max_suffix_length = text_decoder_prefill_step_size(&self.ctx.runtime_config, max_prefix_length).max(32);
        if self.ctx.current_max_prefix_length == max_prefix_length {
            return Ok(());
        }
        let context = self.ctx.context.as_ref();
        *self.ctx.cache_layers.borrow_mut() =
            CacheLayers::new(context, &self.ctx.model_shape, max_prefix_length, max_suffix_length);
        let intermediate_data_type: DataType = self.ctx.decoder_config.output_norm_config.scale_precision.into();
        self.ctx.kv_cache_update =
            KVCacheUpdate::new(context, intermediate_data_type, max_prefix_length).map_err(unable_to_create_context)?;
        self.ctx.current_max_prefix_length = max_prefix_length;
        Ok(())
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
            let token_inputs = self.llm_token_inputs(token_ids, &positions, None, None, 0, 0);

            let encoding_parameters = EncodingParameters::new();

            let context = Rc::clone(&self.ctx.context);
            let mut encoder = Encoder::new(context.as_ref()).map_err(unable_to_create_context)?;

            let mut cache_layers = self.ctx.cache_layers.borrow_mut();
            self.ctx
                .executables
                .encode_prefill(
                    self.decoder_arguments(&token_inputs, token_count, 0, 0, &mut *cache_layers),
                    token_inputs.token_ids(),
                    &encoding_parameters,
                    &mut encoder,
                )
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            drop(cache_layers);
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
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<B>>,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        pending_token_inputs: &mut Vec<TokenInputs<B>>,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
        pending_sampling_outputs: &mut Vec<Allocation<B>>,
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
            let positions = self.ctx.async_chain_positions.as_slice_mut::<i32>();
            for pass in 0..followup_count {
                positions[pass] = (followup_base_position + pass) as i32;
            }
        }
        {
            let seeds = self.ctx.async_chain_seeds.as_slice_mut::<u64>();
            if !matches!(sampling.method(), SamplingMethod::Greedy) {
                for seed in seeds.iter_mut().take(followup_count) {
                    *seed = sampling.next_seed();
                }
            } else {
                seeds[..followup_count].fill(0);
            }
        }

        // Encode the initial forward pass (no new CB, no submit).
        let sampling_output = self.encode_single_forward_pass_on(
            encoder,
            initial_token_ids,
            initial_embedding_injection,
            sampling,
            initial_token_bitmask,
            false, // capture_hidden
            pre_injection_encode,
            initial_seed,
            pending_token_inputs,
            pending_sampling_inputs,
        )?;

        let first_followup_token_ids = if followup_count > 0 {
            let token_ids_shape = [1];
            let token_ids_data_type = crate::DataType::U64;
            let mut next_token_ids = self
                .ctx
                .context
                .create_array_uninitialized(&token_ids_shape, token_ids_data_type, "async_token_id")
                .into_allocation();
            self.ctx.token_copy_sampled.encode(&sampling_output, &mut next_token_ids, encoder);
            Some(unsafe { Array::from_allocation(next_token_ids, 0, &token_ids_shape, token_ids_data_type) })
        } else {
            None
        };
        self.copy_sampling_output_to_async_result_on(encoder, &sampling_output, 0);

        self.encode_cache_acceptance_update_on(encoder, initial_token_count);
        self.register_positions_and_advance(initial_token_count);

        if followup_count > 0 {
            self.encode_followup_passes_on(
                encoder,
                followup_count,
                first_followup_token_ids,
                vocab_mask_limit,
                sampling.method(),
                1,
                pending_token_inputs,
                pending_sampling_inputs,
                pending_sampling_outputs,
            )?;
        }

        pending_sampling_outputs.push(sampling_output);

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
        first_token_ids: Option<Array<B>>,
        vocab_mask_limit: Option<usize>,
        sampling_method: SamplingMethod,
        results_offset_slots: usize,
        pending_token_inputs: &mut Vec<TokenInputs<B>>,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
        pending_sampling_outputs: &mut Vec<Allocation<B>>,
    ) -> Result<(), Error> {
        let encoding_parameters = EncodingParameters::new();
        let cache_layers_rc = self.ctx.cache_layers.clone();
        let token_bitmask =
            vocab_mask_limit.and_then(|limit| self.get_single_token_vocab_mask(limit)).map(Box::<[u32]>::from);
        let mut next_token_ids = first_token_ids;

        for pass in 0..followup_count {
            let results_slot = results_offset_slots + pass;
            let token_ids = [0];
            let seed = self.ctx.async_chain_seeds.as_slice::<u64>()[pass];
            let sampling_inputs = self.create_sampling_inputs_from_slices(&[seed], token_bitmask.as_deref());
            let token_inputs =
                self.llm_token_inputs(&token_ids, &[self.next_position + pass], next_token_ids.take(), None, 0, 1);
            pending_token_inputs.push(token_inputs);
            let token_inputs = pending_token_inputs.last().expect("token inputs must be pending");

            {
                let mut logits = {
                    let mut cache_layers = cache_layers_rc.borrow_mut();
                    self.ctx
                        .executables
                        .encode_decode(
                            self.decoder_arguments(&token_inputs, 1, 0, 1, &mut *cache_layers),
                            DecoderDecodeInput::TokenIds(token_inputs.token_ids()),
                            None,
                            &encoding_parameters,
                            encoder,
                        )
                        .map_err(|err| Error::EncodeFailed(Box::new(err)))?
                };
                let mut sampling_output = self.create_sampling_output(1);
                self.encode_sampling_on(
                    encoder,
                    &mut logits,
                    &mut sampling_output,
                    sampling_method,
                    1,
                    sampling_inputs,
                    pending_sampling_inputs,
                )?;
                if pass + 1 < followup_count {
                    let token_ids_shape = [1];
                    let token_ids_data_type = crate::DataType::U64;
                    let mut updated_token_ids = self
                        .ctx
                        .context
                        .create_array_uninitialized(&token_ids_shape, token_ids_data_type, "async_token_id")
                        .into_allocation();
                    self.ctx.token_copy_sampled.encode(&sampling_output, &mut updated_token_ids, encoder);
                    next_token_ids = Some(unsafe {
                        Array::from_allocation(updated_token_ids, 0, &token_ids_shape, token_ids_data_type)
                    });
                }
                self.copy_sampling_output_to_async_result_on(encoder, &sampling_output, results_slot);
                pending_sampling_outputs.push(sampling_output);
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
            mask[word] |= 2_u32.pow((token_index % 32) as u32);
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
            mask[row_words + word] |= 2_u32.pow((token_index % 32) as u32);
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

    fn create_sampling_output(
        &self,
        sampling_length: usize,
    ) -> Allocation<B> {
        self.ctx
            .context
            .create_array_uninitialized(&[sampling_length], crate::DataType::U32, "tts_sampling_output")
            .into_allocation()
    }

    fn encode_single_forward_pass_on(
        &mut self,
        encoder: &mut Encoder<B>,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        mut pre_injection_encode: Option<&mut PreInjectionEncodeCallback<B>>,
        preconsumed_seed: Option<u64>,
        pending_token_inputs: &mut Vec<TokenInputs<B>>,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
    ) -> Result<Allocation<B>, Error> {
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

        let sampling_seed_end = sampling_start + sampling_length;
        let sampling_token_seeds = &token_seeds[sampling_start..sampling_seed_end];
        let sampling_token_bitmask = token_bitmask.map(|mask| {
            let row_words = self.ctx.decoder_config.vocab_size.div_ceil(32);
            let row_start = sampling_start * row_words;
            let row_end = row_start + sampling_length * row_words;
            &mask[row_start..row_end]
        });
        let sampling_inputs = self.create_sampling_inputs_from_slices(sampling_token_seeds, sampling_token_bitmask);
        let token_inputs = self.llm_token_inputs(token_ids, positions, None, None, sampling_start, sampling_length);
        pending_token_inputs.push(token_inputs);
        let token_inputs = pending_token_inputs.last().expect("token inputs must be pending");

        let encoding_parameters = EncodingParameters::new();
        let mut main = self
            .ctx
            .executables
            .embed
            .encode_lookup(token_inputs.token_ids(), token_count, encoder)
            .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        if let Some(pre_encode) = pre_injection_encode.as_mut() {
            pre_encode(self, encoder)?;
        }
        match embedding_injection {
            EmbeddingInjection::None => {},
            EmbeddingInjection::AddPreloaded {
                post_scale,
            } => {
                self.encode_add_scale_from_single_bias_on(encoder, &mut main, token_count, post_scale.unwrap_or(1.0))?;
            },
            EmbeddingInjection::OverrideFirstRowInternal => {
                self.encode_override_first_row_from_device_on(encoder, &mut main, &self.single_override_embedding)?;
            },
        }
        let mut logits = {
            let mut cache_layers = self.ctx.cache_layers.borrow_mut();
            let decoder_arguments = DecoderArguments {
                token_positions: token_inputs.token_positions(),
                token_parents: token_inputs.token_parents(),
                token_subtrie_ranges: token_inputs.token_subtrie_ranges(),
                shared_buffers: self.ctx.shared_buffers.as_ref(),
                cache_layers: Some(&mut *cache_layers),
                batch_dim: token_count,
                sampling_start,
                sampling_length,
                rope_max_sequence_length: self.ctx.model_shape.context_length(),
                rope_dim: self.ctx.model_shape.rope_dim(),
                #[cfg(feature = "tracing")]
                trace: None,
            };
            let hidden_capture = capture_hidden.then_some(&mut self.single_hidden_capture);
            self.ctx
                .executables
                .encode_decode(
                    decoder_arguments,
                    DecoderDecodeInput::Embeddings(main),
                    hidden_capture,
                    &encoding_parameters,
                    encoder,
                )
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?
        };
        let mut sampling_output = self.create_sampling_output(sampling_length);
        self.encode_sampling_on(
            encoder,
            &mut logits,
            &mut sampling_output,
            sampling.method(),
            sampling_length,
            sampling_inputs,
            pending_sampling_inputs,
        )?;

        Ok(sampling_output)
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
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<B>>,
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
                            mask[word] |= 2_u32.pow((token_index % 32) as u32);
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
                            mask[sampling_start * row_words + word] |= 2_u32.pow((token_index % 32) as u32);
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
            let mut pending_token_inputs = Vec::new();
            let mut pending_sampling_inputs = Vec::new();

            let sampling_output = self.encode_single_forward_pass_on(
                &mut encoder,
                token_ids,
                embedding_injection,
                sampling,
                token_bitmask,
                capture_hidden,
                pre_injection_encode,
                None,
                &mut pending_token_inputs,
                &mut pending_sampling_inputs,
            )?;
            self.encode_cache_acceptance_update_on(&mut encoder, token_count);

            self.submit_and_wait_command_buffer(encoder)?;
            let token = read_sampled_token_from_sampling_output(&sampling_output)?;
            self.register_positions_and_advance(token_count);
            Ok(token)
        })
    }

    fn encode_override_first_row_from_device_on(
        &self,
        encoder: &mut Encoder<B>,
        main: &mut Allocation<B>,
        override_embedding: &Allocation<B>,
    ) -> Result<(), Error> {
        let model_dim = self.ctx.decoder_config.model_dim;
        let model_dim_bytes = model_dim.checked_mul(self.activation_data_type().size_in_bytes()).ok_or(
            TtsModelConfigError::ModelDimExceedsU32 {
                model_dim,
            },
        )?;
        encoder.encode_copy(override_embedding, 0..model_dim_bytes, main, 0..model_dim_bytes);
        Ok(())
    }

    fn encode_add_scale_from_single_bias_on(
        &self,
        encoder: &mut Encoder<B>,
        main: &mut Allocation<B>,
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
        let mut output = encoder.allocate_scratch(main.as_buffer_range().1.len()).map_err(unable_to_create_context)?;
        self.tensor_add_scale.encode(
            &*main,
            &self.single_override_embedding,
            &mut output,
            model_dim_u32,
            total_len_u32,
            scale,
            encoder,
        );
        *main = output;
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
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<B>>,
        pending_token_inputs: &mut Vec<TokenInputs<B>>,
        pending_sampling_inputs: &mut Vec<SamplingInputs<B>>,
        pending_sampling_outputs: &mut Vec<Allocation<B>>,
    ) -> Result<(), Error> {
        let token_count = token_ids.len();

        let sampling_output = self.encode_single_forward_pass_on(
            encoder,
            token_ids,
            embedding_injection,
            sampling,
            precomputed_token_bitmask,
            capture_hidden,
            pre_injection_encode,
            None,
            pending_token_inputs,
            pending_sampling_inputs,
        )?;

        // Copy sampled token to async_chain_results[0] so the caller can
        // read it after the eventual submit.
        self.copy_sampling_output_to_async_result_on(encoder, &sampling_output, 0);
        pending_sampling_outputs.push(sampling_output);

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
        let result = encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| Error::CommandBufferFailed(Box::new(err)));
        result?;
        self.instrumentation.host_waits += 1;
        Ok(())
    }

    pub(super) fn copy_async_chain_results_to_on(
        &mut self,
        encoder: &mut Encoder<B>,
        src_slot: usize,
        dst: &mut Allocation<B>,
        dst_offset: usize,
        count: usize,
    ) -> Result<(), Error> {
        let src_offset = src_slot.checked_mul(std::mem::size_of::<u32>()).ok_or(
            TtsModelConfigError::AsyncChainCopySourceOffsetOverflow {
                src_slot,
            },
        )?;
        let byte_len =
            count.checked_mul(std::mem::size_of::<u32>()).ok_or(TtsModelConfigError::AsyncChainCopySizeOverflow {
                count,
            })?;
        encoder.encode_copy(
            &self.ctx.async_chain_results,
            src_offset..src_offset + byte_len,
            dst,
            dst_offset..dst_offset + byte_len,
        );
        Ok(())
    }

    pub(super) fn read_async_chain_result(
        &self,
        slot: usize,
    ) -> Result<u32, Error> {
        let capacity = self.ctx.async_chain_capacity;
        if slot >= capacity {
            return Err(TtsModelConfigError::AsyncChainResultSlotOutOfBounds {
                slot,
                capacity,
            }
            .into());
        }
        let (buffer, range) = self.ctx.async_chain_results.as_buffer_range();
        Ok(unsafe {
            *((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start + slot * std::mem::size_of::<u32>())
                as *const u32)
        })
    }

    pub(super) fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
    }

    pub(super) fn clear_instrumentation(&mut self) {
        self.instrumentation = RunnerInstrumentation::default();
    }
}

fn read_sampled_token_from_sampling_output<B: Backend>(sampling_output: &Allocation<B>) -> Result<u64, Error> {
    let (buffer, range) = sampling_output.as_buffer_range();
    let token = unsafe { *((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start) as *const u32) };
    Ok(u64::from(token))
}
