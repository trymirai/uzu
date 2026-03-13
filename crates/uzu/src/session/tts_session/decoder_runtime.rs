use super::*;
use super::decoder_support::*;

pub(super) type MetalContext = <Metal as Backend>::Context;
pub(super) type MetalCommandBuffer = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;

pub(super) struct TokenDecoderRunner {
    pub(super) context: Rc<MetalContext>,
    command_buffer: Rc<RefCell<MetalCommandBuffer>>,
    cache_layers: Rc<RefCell<CacheLayers<Metal>>>,
    shared_buffers: Rc<RefCell<SharedBuffers<Metal>>>,
    scratch_buffers: ScratchBuffers<Metal>,
    model_shape: ModelShape,
    decoder_config: Rc<crate::config::DecoderConfig>,
    executables: Decoder<Metal>,
    sampler: GpuSampling<Metal>,
    repetition_penalty: <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel,
    kv_cache_update: KVCacheUpdate<Metal>,
    tensor_copy: <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel,
    tensor_add_scale: <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel,
    token_copy_sampled: <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel,
    token_copy_results: <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel,
    async_chain_positions: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_seeds: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_results: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_repetition_tokens: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_repetition_counts: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_capacity: usize,
    repetition_capacity: usize,
    repetition_tokens: ArrayCell<Metal>,
    repetition_counts: ArrayCell<Metal>,
    repetition_window_raw: Vec<u32>,
    pub(super) single_hidden_capture: ArrayCell<Metal>,
    pub(super) single_override_embedding: ArrayCell<Metal>,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    should_fill_attention_bias: bool,
    next_position: usize,
    instrumentation: RunnerInstrumentation,
}

impl TokenDecoderRunner {
    pub(super) fn new_with_context(
        context: Rc<MetalContext>,
        model_path: &Path,
        decoder_config: Rc<crate::config::DecoderConfig>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        runtime_config: &TextDecoderRuntimeConfig,
        repetition_window_size: usize,
    ) -> Result<Self, Error> {
        let command_buffer = Rc::new(RefCell::new(
            context.create_command_buffer().expect("Failed to create command buffer").start_encoding(),
        ));

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = text_decoder_prefill_step_size(runtime_config, decoder_config.context_length).max(32);
        let should_fill_attention_bias =
            model_shape.sliding_window_length_per_layer.iter().any(|value| value.is_some());
        let activation_data_type = model_shape.activation_data_type();

        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        shared_buffers.borrow_mut().update_data_with_transformer_subtree(&root_loader_view, transformer_subtree);

        let scratch_buffers =
            ScratchBuffers::new(context.as_ref(), &decoder_config, &model_shape, max_prefix_length, max_suffix_length);
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
        let repetition_penalty =
            <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel::new(context.as_ref(), logits_data_type)
                .map_err(unable_to_create_context)?;
        let tensor_copy =
            <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let tensor_add_scale =
            <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let token_copy_sampled =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;
        let token_copy_results =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;
        let async_chain_capacity = max_suffix_length.max(1);
        let async_chain_positions = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<i32>())
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_seeds = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u64>())
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_results = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(unable_to_create_context)?,
        ));
        let repetition_capacity = repetition_window_size.max(1);
        let async_chain_repetition_tokens = Rc::new(RefCell::new(
            context
                .create_buffer(
                    async_chain_capacity
                        .checked_mul(repetition_capacity)
                        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                        .ok_or_else(|| {
                            unable_to_create_context(std::io::Error::other(
                                "async chain repetition buffer size overflow",
                            ))
                        })?,
                )
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_repetition_counts = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(unable_to_create_context)?,
        ));
        let repetition_tokens =
            RefCell::new(context.create_array(&[repetition_capacity], DataType::U32, "tts_repetition_tokens"));
        let repetition_counts = RefCell::new(context.create_array(&[1], DataType::U32, "tts_repetition_counts"));
        let single_hidden_capture = RefCell::new(context.create_array(
            &[1, decoder_config.model_dim],
            activation_data_type,
            "tts_single_hidden_capture",
        ));
        let single_override_embedding = RefCell::new(context.create_array(
            &[1, decoder_config.model_dim],
            activation_data_type,
            "tts_single_override_embedding",
        ));

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let intermediate_data_type: DataType = decoder_config.output_norm_config.scale_precision.into();
        let kv_cache_update = KVCacheUpdate::new(context.as_ref(), intermediate_data_type, max_prefix_length)
            .map_err(unable_to_create_context)?;

        Ok(Self {
            context,
            command_buffer,
            cache_layers,
            shared_buffers,
            scratch_buffers,
            model_shape,
            decoder_config,
            executables,
            sampler,
            repetition_penalty,
            kv_cache_update,
            tensor_copy,
            tensor_add_scale,
            token_copy_sampled,
            token_copy_results,
            async_chain_positions,
            async_chain_seeds,
            async_chain_results,
            async_chain_repetition_tokens,
            async_chain_repetition_counts,
            async_chain_capacity,
            repetition_capacity,
            repetition_tokens,
            repetition_counts,
            repetition_window_raw: Vec::with_capacity(repetition_capacity),
            single_hidden_capture,
            single_override_embedding,
            single_token_vocab_masks: HashMap::new(),
            two_token_vocab_masks: HashMap::new(),
            should_fill_attention_bias,
            next_position: 0,
            instrumentation: RunnerInstrumentation::default(),
        })
    }

    pub(super) fn reset(&mut self) {
        self.cache_layers.borrow_mut().clear();
        self.next_position = 0;
        self.clear_repetition_window();
    }

    pub(super) fn clear_repetition_window(&mut self) {
        self.repetition_window_raw.clear();
        self.repetition_counts.borrow_mut().as_slice_mut::<u32>()[0] = 0;
    }

    pub(super) fn set_repetition_window(
        &mut self,
        previous_tokens: &[u32],
    ) -> Result<(), Error> {
        let mut tokens = self.repetition_tokens.borrow_mut();
        let raw_window = tail_repetition_window(previous_tokens, tokens.shape()[0]);
        self.repetition_window_raw.clear();
        self.repetition_window_raw.extend_from_slice(raw_window);
        let count = write_repetition_window_tail(tokens.as_slice_mut::<u32>(), previous_tokens);
        self.repetition_counts.borrow_mut().as_slice_mut::<u32>()[0] =
            u32::try_from(count).map_err(|_| Error::GenerateFailed)?;
        Ok(())
    }

    fn encode_repetition_penalty_if_needed_with_buffers<'tokens, 'counts>(
        &self,
        sampling: &TextSamplingState,
        count: u32,
        previous_tokens: impl BufferArg<'tokens, <Metal as Backend>::NativeBuffer>,
        previous_counts: impl BufferArg<'counts, <Metal as Backend>::NativeBuffer>,
        batch_size: u32,
        max_previous_tokens: u32,
    ) -> Result<(), Error> {
        if !sampling.uses_repetition_penalty() {
            return Ok(());
        }
        if count == 0 {
            return Ok(());
        }
        let logits = self.scratch_buffers.logits.borrow();
        let logits_buffer = logits.buffer();
        let mut logits_buffer = logits_buffer.borrow_mut();
        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            self.repetition_penalty.encode(
                &mut *logits_buffer,
                previous_tokens,
                previous_counts,
                batch_size,
                self.decoder_config.vocab_size as u32,
                max_previous_tokens,
                sampling.repetition_penalty(),
                encoder,
            );
        });
        Ok(())
    }

    fn encode_repetition_penalty_if_needed(
        &self,
        sampling: &TextSamplingState,
    ) -> Result<(), Error> {
        let repetition_tokens = self.repetition_tokens.borrow();
        let repetition_counts = self.repetition_counts.borrow();
        let count = repetition_counts.as_slice::<u32>()[0];
        let repetition_tokens_buffer = repetition_tokens.buffer();
        let repetition_tokens_buffer = repetition_tokens_buffer.borrow();
        let repetition_counts_buffer = repetition_counts.buffer();
        let repetition_counts_buffer = repetition_counts_buffer.borrow();
        self.encode_repetition_penalty_if_needed_with_buffers(
            sampling,
            count,
            &*repetition_tokens_buffer,
            &*repetition_counts_buffer,
            1_u32,
            repetition_tokens.shape()[0] as u32,
        )
    }

    fn populate_async_chain_repetition_windows(
        &mut self,
        repetition_windows: Option<&[Vec<u32>]>,
        first_codebook_index: usize,
        followup_count: usize,
    ) -> Result<(), Error> {
        let tokens_ptr = self.async_chain_repetition_tokens.borrow().cpu_ptr().as_ptr() as *mut u32;
        let counts_ptr = self.async_chain_repetition_counts.borrow().cpu_ptr().as_ptr() as *mut u32;
        for pass in 0..followup_count {
            let row = first_codebook_index + pass;
            let previous_tokens =
                repetition_windows.and_then(|windows| windows.get(row).map(Vec::as_slice)).unwrap_or(&[]);
            let slot_offset = pass.checked_mul(self.repetition_capacity).ok_or(Error::GenerateFailed)?;
            let slot = unsafe { std::slice::from_raw_parts_mut(tokens_ptr.add(slot_offset), self.repetition_capacity) };
            let count = write_repetition_window_tail(slot, previous_tokens);
            unsafe {
                *counts_ptr.add(pass) = u32::try_from(count).map_err(|_| Error::GenerateFailed)?;
            }
        }
        Ok(())
    }

    pub(super) fn prefill_without_sampling(
        &mut self,
        token_ids: &[u64],
    ) -> Result<(), Error> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut sampling = TextSamplingState::with_params(0, 0.0, 1.0, 1.0);
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
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_>>,
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
        first_codebook_index: usize,
        repetition_windows: Option<&[Vec<u32>]>,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if followup_count == 0 {
            return Ok(());
        }

        if followup_count > self.async_chain_capacity {
            return Err(Error::GenerateFailed);
        }

        let vocab_mask_limit = if let Some(limit_raw) = vocab_limit {
            let limit = limit_raw.min(self.decoder_config.vocab_size);
            if limit == 0 || limit >= self.decoder_config.vocab_size {
                None
            } else {
                self.prepare_two_token_vocab_mask(limit)?;
                self.prepare_single_token_vocab_mask(limit)?;
                Some(limit)
            }
        } else {
            None
        };

        self.populate_async_chain_repetition_windows(repetition_windows, first_codebook_index, followup_count)?;

        {
            let positions_ptr = self.async_chain_positions.borrow().cpu_ptr().as_ptr() as *mut i32;
            for pass in 0..followup_count {
                unsafe {
                    *positions_ptr.add(pass) = (self.next_position + pass) as i32;
                }
            }
        }

        {
            let seeds_ptr = self.async_chain_seeds.borrow().cpu_ptr().as_ptr() as *mut u64;
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

        self.command_buffer = Rc::new(RefCell::new(
            self.context.create_command_buffer().expect("Failed to create command buffer").start_encoding(),
        ));
        for pass in 0..followup_count {
            let token_ids = [if pass == 0 { first_token } else { 0 }];
            let token_bitmask = vocab_mask_limit.and_then(|limit| self.get_single_token_vocab_mask(limit));
            let mut state = ForwardPassState::new_llm(
                self.context.clone(),
                &self.decoder_config,
                &self.model_shape,
                &self.scratch_buffers,
                self.cache_layers.clone(),
                self.shared_buffers.clone(),
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
                Some((self.async_chain_positions.clone(), pass)),
                Some((self.async_chain_seeds.clone(), pass)),
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling.method());
            }

            let encoding_parameters = EncodingParameters::new();
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.executables
                    .embed
                    .encode_lookup(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                for layer in self.executables.layers.iter() {
                    layer
                        .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                        .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                }
                self.executables
                    .norm
                    .encode(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.executables
                    .embed
                    .encode_readout(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            let count =
                unsafe { *(self.async_chain_repetition_counts.borrow().cpu_ptr().as_ptr() as *const u32).add(pass) };
            let tokens_offset = pass * self.repetition_capacity * std::mem::size_of::<u32>();
            let counts_offset = pass * std::mem::size_of::<u32>();
            let async_chain_repetition_tokens = self.async_chain_repetition_tokens.borrow();
            let async_chain_repetition_counts = self.async_chain_repetition_counts.borrow();
            self.encode_repetition_penalty_if_needed_with_buffers(
                sampling,
                count,
                (&*async_chain_repetition_tokens, tokens_offset),
                (&*async_chain_repetition_counts, counts_offset),
                1_u32,
                self.repetition_capacity as u32,
            )?;
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.sampler
                    .encode(&mut state, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }

            let sampling_output = state.sampling_output().ok_or(Error::GenerateFailed)?;
            let sampling_output_binding = sampling_output.borrow();
            let sampling_output_buffer = sampling_output_binding.buffer();
            let token_ids_binding = self.scratch_buffers.token_ids.borrow();
            let token_ids_buffer = token_ids_binding.buffer();

            self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
                let sampling_output_buffer = sampling_output_buffer.borrow();
                let mut token_ids_buffer = token_ids_buffer.borrow_mut();
                if pass + 1 < followup_count {
                    self.token_copy_sampled.encode(&*sampling_output_buffer, &mut *token_ids_buffer, encoder);
                }
                let results_offset = pass * std::mem::size_of::<u32>();
                let mut results_buffer = self.async_chain_results.borrow_mut();
                self.token_copy_results.encode(
                    &*sampling_output_buffer,
                    (&mut *results_buffer, results_offset),
                    encoder,
                );
            });

            self.cache_layers.borrow_mut().update_after_acceptance(
                &[0],
                None,
                self.command_buffer.borrow_mut().deref_mut(),
                &self.kv_cache_update,
            );
            self.cache_layers.borrow_mut().register_accepted_tokens(&[self.next_position + pass]);
        }

        self.next_position = self.next_position.saturating_add(followup_count);
        self.submit_and_wait_current_command_buffer()?;

        let results_ptr = self.async_chain_results.borrow().cpu_ptr().as_ptr() as *const u32;
        for pass in 0..followup_count {
            let sampled = unsafe { *results_ptr.add(pass) };
            on_token(pass, u64::from(sampled))?;
        }
        Ok(())
    }

    pub(super) fn decode_followup_tokens_sequential(
        &mut self,
        first_codebook_index: usize,
        mut previous_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        repetition_windows: Option<&[Vec<u32>]>,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for pass in 0..followup_count {
            if let Some(windows) = repetition_windows {
                let row = first_codebook_index + pass;
                self.set_repetition_window(windows.get(row).map_or(&[], Vec::as_slice))?;
            } else {
                self.clear_repetition_window();
            }
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
        let limit = vocab_limit.min(self.decoder_config.vocab_size);
        if limit == 0 || limit >= self.decoder_config.vocab_size {
            return Ok(());
        }
        if self.single_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.decoder_config.vocab_size.div_ceil(32);
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
        let limit = vocab_limit.min(self.decoder_config.vocab_size);
        if limit == 0 || limit >= self.decoder_config.vocab_size {
            return Ok(());
        }
        if self.two_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.decoder_config.vocab_size.div_ceil(32);
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
        mut pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_>>,
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

            let row_words = self.decoder_config.vocab_size.div_ceil(32);
            let token_bitmask_source = if let Some(mask) = precomputed_token_bitmask {
                let expected_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
                if mask.len() != expected_words {
                    return Err(Error::GenerateFailed);
                }
                TokenBitmaskSource::Borrowed(mask)
            } else if let Some(limit_raw) = vocab_limit {
                let limit = limit_raw.min(self.decoder_config.vocab_size);
                if limit == 0 {
                    return Err(Error::GenerateFailed);
                }
                if limit >= self.decoder_config.vocab_size {
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
                        let mut mask = vec![0_u32; token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[sampling_start * row_words + word] |= 1_u32 << bit;
                        }
                        TokenBitmaskSource::Owned(mask)
                    }
                } else {
                    let mut mask = vec![0_u32; token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?];
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
                self.context.clone(),
                &self.decoder_config,
                &self.model_shape,
                &self.scratch_buffers,
                self.cache_layers.clone(),
                self.shared_buffers.clone(),
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
            self.command_buffer = Rc::new(RefCell::new(
                self.context.create_command_buffer().expect("Failed to create command buffer").start_encoding(),
            ));
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.executables
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
            for layer in self.executables.layers.iter() {
                layer
                    .encode(&mut state, &encoding_parameters, self.command_buffer.borrow_mut().deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            if capture_hidden {
                self.encode_capture_last_hidden_into_single_buffer(&state, token_count)?;
            }
            self.executables
                .norm
                .encode(&mut state, self.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.executables
                .embed
                .encode_readout(&mut state, self.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.encode_repetition_penalty_if_needed(sampling)?;
            self.sampler
                .encode(&mut state, self.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.cache_layers.borrow_mut().update_after_acceptance(
                accepted_suffix_indices,
                None,
                self.command_buffer.borrow_mut().deref_mut(),
                &self.kv_cache_update,
            );
            self.submit_and_wait_current_command_buffer()?;
            let token = read_sampled_token_from_sampling_output(&state)?;
            self.cache_layers.borrow_mut().register_accepted_tokens(positions);
            self.next_position = self.next_position.saturating_add(token_count);
            Ok(token)
        })
    }

    fn encode_capture_last_hidden_into_single_buffer(
        &self,
        state: &ForwardPassState<Metal>,
        token_count: usize,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.decoder_config.model_dim;
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

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            let main_buffer = main.buffer();
            let main_buffer = main_buffer.borrow();
            let capture_buffer = capture.buffer();
            let mut capture_buffer = capture_buffer.borrow_mut();
            self.tensor_copy.encode((&*main_buffer, src_offset), &mut *capture_buffer, model_dim_u32, encoder);
        });
        Ok(())
    }

    fn encode_override_first_row_from_device(
        &self,
        state: &ForwardPassState<Metal>,
        override_embedding: &ArrayCell<Metal>,
    ) -> Result<(), Error> {
        let model_dim = self.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let override_embedding = override_embedding.borrow();
        if override_embedding.shape() != [1, model_dim] || override_embedding.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
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
        });
        Ok(())
    }

    fn encode_add_scale_from_single_bias(
        &self,
        state: &ForwardPassState<Metal>,
        token_count: usize,
        scale: f32,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let total_len = token_count.checked_mul(model_dim).ok_or(Error::GenerateFailed)?;
        let total_len_u32 = u32::try_from(total_len).map_err(|_| Error::GenerateFailed)?;

        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let bias = self.single_override_embedding.borrow();
        if bias.shape() != [1, model_dim] || bias.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            let main_input_buffer = main.buffer();
            let main_input_buffer = main_input_buffer.borrow();
            let bias_buffer = bias.buffer();
            let bias_buffer = bias_buffer.borrow();
            let main_output_buffer = main.buffer();
            let mut main_output_buffer = main_output_buffer.borrow_mut();
            self.tensor_add_scale.encode(
                (&*main_input_buffer, main.offset()),
                &*bias_buffer,
                (&mut *main_output_buffer, main.offset()),
                model_dim_u32,
                total_len_u32,
                scale,
                encoder,
            );
        });
        Ok(())
    }

    fn submit_and_wait_current_command_buffer(&mut self) -> Result<(), Error> {
        let replacement = self.context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
        let command_buffer = {
            let mut command_buffer = self.command_buffer.borrow_mut();
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

fn read_sampled_token_from_sampling_output(state: &ForwardPassState<Metal>) -> Result<u64, Error> {
    let output = state.sampling_output().ok_or(Error::GenerateFailed)?;
    let output = output.borrow();
    let tokens = output.as_slice::<u32>();
    let token = tokens.first().copied().ok_or(Error::GenerateFailed)?;
    Ok(u64::from(token))
}

pub(super) fn write_f32_slice_into_array(
    array: &mut crate::array::Array<Metal>,
    values: &[f32],
) -> Result<(), Error> {
    if array.num_elements() != values.len() {
        return Err(Error::GenerateFailed);
    }
    match array.data_type() {
        DataType::F32 => {
            array.as_slice_mut::<f32>().copy_from_slice(values);
            Ok(())
        },
        DataType::F16 => {
            for (dst, &src) in array.as_slice_mut::<f16>().iter_mut().zip(values.iter()) {
                *dst = f16::from_f32(src);
            }
            Ok(())
        },
        DataType::BF16 => {
            for (dst, &src) in array.as_slice_mut::<bf16>().iter_mut().zip(values.iter()) {
                *dst = bf16::from_f32(src);
            }
            Ok(())
        },
        _ => Err(Error::GenerateFailed),
    }
}

impl MatrixF32 {
    #[cfg(test)]
    pub(super) fn row(
        &self,
        index: usize,
    ) -> Option<&[f32]> {
        if index >= self.rows {
            return None;
        }
        let start = index.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.values.get(start..end)
    }

    #[cfg(test)]
    pub(super) fn matmul_into(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Option<()> {
        if input.len() != self.cols || output.len() != self.rows {
            return None;
        }
        for (row_index, row) in self.values.chunks_exact(self.cols).enumerate() {
            let mut acc = 0.0_f32;
            for (&w, &x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            output[row_index] = acc;
        }
        Some(())
    }
}

pub(super) fn load_matrix_f32(
    weights_path: &Path,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<MatrixF32, Error> {
    let file = File::open(weights_path).map_err(|_| Error::UnableToLoadWeights)?;
    let (global_offset, metadata) = read_safetensors_metadata(&file).map_err(|_| Error::UnableToLoadWeights)?;
    let tensor = metadata.tensors.get(key).ok_or(Error::UnableToLoadWeights)?;

    if tensor.shape.len() != 2 {
        return Err(Error::UnableToLoadWeights);
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if rows != expected_rows || cols != expected_cols {
        return Err(Error::UnableToLoadConfig);
    }

    let (begin, end) = tensor.data_offsets;
    let size = end.checked_sub(begin).ok_or(Error::UnableToLoadWeights)?;
    let offset = global_offset.checked_add(begin).ok_or(Error::UnableToLoadWeights)?;
    let data_type: DataType = tensor.dtype.into();
    let expected_size = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(data_type.size_in_bytes()))
        .ok_or(Error::UnableToLoadWeights)?;
    if size != expected_size {
        return Err(Error::UnableToLoadWeights);
    }

    let mut bytes = vec![0_u8; size];
    file.read_exact_at(&mut bytes, offset as u64).map_err(|_| Error::UnableToLoadWeights)?;

    let values = match data_type {
        DataType::F32 => decode_f32_bytes(&bytes),
        DataType::F16 => decode_f16_bytes_to_f32(&bytes),
        DataType::BF16 => decode_bf16_bytes_to_f32(&bytes),
        _ => return Err(Error::UnableToLoadWeights),
    };
    if values.len() != rows.checked_mul(cols).ok_or(Error::UnableToLoadWeights)? {
        return Err(Error::UnableToLoadWeights);
    }

    Ok(MatrixF32 {
        rows,
        cols,
        values,
    })
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn decode_f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|chunk| f32::from(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))).collect()
}

fn decode_bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|chunk| f32::from(bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))).collect()
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
