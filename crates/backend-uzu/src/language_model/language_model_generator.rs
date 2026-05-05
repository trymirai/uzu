use std::{any::Any, iter::repeat_n, ops::Range, path::Path, time::Instant};

use itertools::{Either, Itertools, izip};

use super::{
    gpu_capture::GpuCaptureManager,
    language_model_generator_context::LanguageModelGeneratorContext,
    result::{GenerateResult, PrefillResult},
    rng::PRng,
};
use crate::{
    array::{Array, ArrayContextExt},
    backends::common::{
        Allocation, Backend, Buffer, CommandBuffer, Context, Encoder, Pending, kernel::TokenCopySampledKernel,
    },
    config::ModelMetadata,
    encodable_block::{DecoderDecodeInput, EncodingParameters, SamplingArguments, SamplingInputs},
    forward_pass::{cache_layers::CacheLayersSlice, kv_cache_layer::INVALID_POSITION, token_inputs::TokenInputs},
    language_model::grammar::CompiledGrammar,
    session::{
        config::DecodingConfig,
        helpers::Context as LlmContext,
        parameter::{ConfigResolvableValue, ResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::{TrieCreationConfig, TrieNode},
    utils::pointers::SendPtr,
};

#[derive(Debug, Clone)]
struct Task<'a> {
    token_ids: &'a [u64],
    token_subtrie_ranges: Option<&'a [[u32; 3]]>,
    token_positions: &'a [usize],
    token_bitmask: Option<&'a [u32]>,
    token_seeds: &'a [u64],
    active_row_count: usize,
    sampling_start: usize,
    sampling_length: usize,
    is_prefilling: bool,
}

struct ForwardPassResources<B: Backend> {
    token_inputs: TokenInputs<B>,
    logits: Option<Allocation<B>>,
    sampling_output: Option<Allocation<B>>,
    sampling_inputs: Option<SamplingInputs<B>>,
}

struct InFlightForwardPass<B: Backend> {
    resources: ForwardPassResources<B>,
    pending: Pending<B>,
}

pub struct LanguageModelGenerator<B: Backend> {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: LanguageModelGeneratorContext<B>,
    async_in_flight: Vec<Option<InFlightForwardPass<B>>>,
    async_token_ids: Option<Array<B>>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager<B>,
}

pub trait LanguageModelGeneratorTrait {
    fn prefill(
        &mut self,
        tokens: Vec<u64>,
        compiled_grammar: Option<&mut (dyn CompiledGrammar + 'static)>,
        sampling_method: SamplingMethod,
        prefix_offset: usize,
        sample_suffix: bool,
    ) -> Result<PrefillResult, Error>;

    fn generate(
        &mut self,
        compiled_grammar: Option<&mut (dyn CompiledGrammar + 'static)>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error>;

    fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    );
    fn async_generate(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: Box<dyn FnOnce(u64) + Send>,
    ) -> Result<(), Error>;
    fn finish_async(
        &mut self,
        pass_idx: usize,
    );

    fn clear_cache(&mut self);
    fn reset_state(&mut self);
    fn peak_memory_usage(&self) -> Option<usize>;

    fn tokens_len(&self) -> usize;
    fn tokens_push(
        &mut self,
        token: u64,
    );
    fn generate_suffix_length(&self) -> usize;
    fn async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize;

    fn get_slice(
        &self,
        range: Range<usize>,
    ) -> Option<Box<dyn Any>>;
    fn apply_slice(
        &mut self,
        slice: &dyn Any,
        range: Range<usize>,
    );

    fn build_llm_context(&self) -> Box<dyn Any>;
    fn reconfigure_from_context(
        &mut self,
        context: &dyn Any,
    );
}

impl<B: Backend> LanguageModelGeneratorTrait for LanguageModelGenerator<B> {
    fn prefill(
        &mut self,
        tokens: Vec<u64>,
        mut compiled_grammar: Option<&mut (dyn CompiledGrammar + 'static)>,
        sampling_method: SamplingMethod,
        prefix_offset: usize,
        sample_suffix: bool,
    ) -> Result<PrefillResult, Error> {
        assert!(!tokens.is_empty());

        self.tokens.extend(tokens.clone());

        let tokens_length = tokens.len();

        let prefill_step_size = self.decoding_config.prefill_step_size.resolve(&self.context.model_config);
        let prefill_steps = tokens_length.div_ceil(prefill_step_size);
        let prefill_size = prefill_steps * prefill_step_size;

        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = if sample_suffix {
            self.decoding_config.generate_suffix_length().saturating_sub(1).min(prefill_size - tokens_length)
        } else {
            prefill_size - tokens_length
        };
        let suffix_root = TrieNode::from_speculator(
            &tokens,
            &self.context.seed,
            compiled_grammar.as_deref_mut(),
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            suffix_length + 1,
        );
        let flat_trie = suffix_root.linearize();

        let has_grammar = compiled_grammar.is_some();

        let token_ids =
            tokens.iter().copied().take(tokens_length - 1).chain(flat_trie.token_ids()).chunks(prefill_step_size);

        let token_subtrie_ranges = repeat_n(None, tokens_length - 1)
            .chain(flat_trie.token_subtrie_ranges().map(Some))
            .chunks(prefill_step_size);

        let token_positions = (prefix_offset..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| prefix_offset + tokens_length - 1 + trie_position))
            .chunks(prefill_step_size);

        let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
        let token_bitmasks = repeat_n(None, tokens_length - 1).chain(flat_trie.token_masks()).chunks(prefill_step_size);

        let token_seeds = repeat_n(0, tokens_length - 1).chain(flat_trie.token_seeds()).chunks(prefill_step_size);

        let mut last_sampling_output: Option<Allocation<B>> = None;
        let mut last_sampling_length = 0;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for (
            step,
            (step_token_ids, step_token_subtrie_ranges, step_token_positions, step_token_bitmasks, step_token_seeds),
        ) in izip!(&token_ids, &token_subtrie_ranges, &token_positions, &token_bitmasks, &token_seeds).enumerate()
        {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;

            let step_token_ids = step_token_ids.collect::<Box<[u64]>>();
            let step_token_subtrie_ranges = step_token_subtrie_ranges.collect::<Box<[Option<[u32; 3]>]>>();
            let step_token_subtrie_ranges: Option<Box<[[u32; 3]]>> =
                if let Some(trie_start) = step_token_subtrie_ranges.iter().position(|e| e.is_some()) {
                    Some(
                        step_token_subtrie_ranges
                            .iter()
                            .enumerate()
                            .map(|(i, me)| {
                                if let Some([subtrie_start, subtrie_end, height]) = me {
                                    [
                                        trie_start as u32 + subtrie_start,
                                        trie_start as u32 + subtrie_end,
                                        trie_start as u32 + height,
                                    ]
                                } else {
                                    [i as u32, step_token_subtrie_ranges.len() as u32 - 1, i as u32]
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };
            let step_token_positions = step_token_positions.collect::<Box<[usize]>>();
            let step_token_seeds = step_token_seeds.collect::<Box<[u64]>>();

            let active_row_count = step_token_positions.len();
            let is_last_prefill_step = step == prefill_steps - 1;
            let should_sample_after_step = sample_suffix && is_last_prefill_step;

            // If we sample on the last prefill step, we only need logits/sampling
            // for tokens that are beyond the prompt prefix (i.e. starting at the
            // suffix-root token, which is the last prompt token).
            let (sampling_start, sampling_length) = if should_sample_after_step {
                let suffix_root_index_in_step = (tokens_length - 1).saturating_sub(tokens_start_index);
                let sampling_length = active_row_count.saturating_sub(suffix_root_index_in_step);
                debug_assert!(sampling_length > 0, "Expected at least one token to sample on the last prefill step");
                (suffix_root_index_in_step, sampling_length)
            } else {
                (0, 0)
            };

            let step_token_bitmask: Option<Box<[u32]>> = if has_grammar && sampling_length > 0 {
                Some(
                    step_token_bitmasks
                        .map(|mask| match mask {
                            Some(mask) => Either::Left(
                                mask.iter()
                                    .copied()
                                    .take(single_token_bitmask_size)
                                    .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                            ),
                            None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                        })
                        .flatten()
                        .collect::<Box<[u32]>>(),
                )
            } else {
                // Drain the chunk iterator to keep the other chunked iterators aligned.
                let _ = step_token_bitmasks.count();
                None
            };

            let should_capture = self.gpu_capture.should_capture_prefill(step == 0);

            if should_capture {
                let _ = self.gpu_capture.start_capture(&self.context.context, "prefill");
            }

            let _ = last_sampling_output.take();

            let task = Task {
                token_ids: &step_token_ids,
                token_subtrie_ranges: step_token_subtrie_ranges.as_deref(),
                token_positions: &step_token_positions,
                token_bitmask: step_token_bitmask.as_deref(),
                token_seeds: &step_token_seeds,
                active_row_count,
                sampling_start,
                sampling_length,
                is_prefilling: !should_sample_after_step,
            };

            let (sampling_output, run_time) = self.run_model(task, sampling_method)?;

            if should_capture {
                self.gpu_capture.stop_capture(&self.context.context, "prefill").map_err(|_| Error::CaptureFailed)?;
            }

            // Register the accepted prompt tokens from this step.
            let step_end_token_index = std::cmp::min(tokens_end_index, tokens_length);
            let mut tokens_processed_this_step = step_end_token_index - tokens_start_index;

            if step == prefill_steps - 1 && sample_suffix {
                tokens_processed_this_step = tokens_processed_this_step.saturating_sub(1);
            }

            if tokens_processed_this_step > 0 {
                self.update_cache_layers(&(0..tokens_processed_this_step).collect::<Vec<usize>>(), None, true)?;

                self.context.cache_layers.borrow_mut().register_accepted_tokens(tokens_processed_this_step);

                self.registered_prefix_len = prefix_offset + tokens_start_index + tokens_processed_this_step;
            }

            last_sampling_length = sampling_length;
            last_sampling_output = sampling_output;
            run_times.push(run_time);
        }

        let final_sampling_output = last_sampling_output;
        if !sample_suffix {
            self.sync_prefix();
            return Ok(PrefillResult {
                tokens: Vec::new(),
                forwardpass_durations: run_times,
            });
        }
        let sampled_tokens = self.read_sampling_output(
            final_sampling_output.as_ref().expect("sampling output must exist"),
            last_sampling_length,
        )?;

        let last_suffix_start = prefill_step_size * (prefill_steps - 1);
        let suffix_root_index = (tokens_length - last_suffix_start) - 1;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());

        self.update_cache_layers(
            &accepted_token_indices.into_iter().map(|p| suffix_root_index + p).collect::<Box<[usize]>>(),
            Some(last_suffix_start),
            false,
        )?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(PrefillResult {
            tokens: accepted_tokens,
            forwardpass_durations: run_times,
        })
    }

    fn generate(
        &mut self,
        mut compiled_grammar: Option<&mut (dyn CompiledGrammar + 'static)>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = self.decoding_config.generate_suffix_length();
        let suffix_root = TrieNode::from_speculator(
            &self.tokens,
            &self.context.seed,
            compiled_grammar.as_deref_mut(),
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            suffix_length,
        );

        let flat_trie = suffix_root.linearize();
        let active_row_count = flat_trie.len();

        let token_ids =
            flat_trie.token_ids().chain(repeat_n(0, suffix_length - active_row_count)).collect::<Box<[u64]>>();

        let token_subtrie_ranges = flat_trie
            .token_subtrie_ranges()
            .chain(repeat_n([u32::MAX, u32::MAX, u32::MAX], suffix_length - active_row_count))
            .collect::<Box<[[u32; 3]]>>();

        let token_bitmask: Option<Box<[u32]>> = compiled_grammar.is_some().then(|| {
            let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
            flat_trie
                .token_masks()
                .chain(repeat_n(None, suffix_length - active_row_count))
                .map(|mask| match mask {
                    Some(mask) => Either::Left(
                        mask.iter()
                            .copied()
                            .take(single_token_bitmask_size)
                            .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                    ),
                    None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                })
                .flatten()
                .collect::<Box<[u32]>>()
        });

        let start_position = self.tokens.len() - 1;
        let token_positions = flat_trie
            .token_positions()
            .map(|trie_position| start_position + trie_position)
            .chain(repeat_n(INVALID_POSITION, suffix_length - active_row_count))
            .collect::<Box<[usize]>>();

        let token_seeds =
            flat_trie.token_seeds().chain(repeat_n(0, suffix_length - active_row_count)).collect::<Box<[u64]>>();

        let task = Task {
            token_ids: &token_ids,
            token_subtrie_ranges: Some(&token_subtrie_ranges),
            token_positions: &token_positions,
            token_bitmask: token_bitmask.as_deref(),
            token_seeds: &token_seeds,
            active_row_count,
            sampling_start: 0,
            sampling_length: active_row_count,
            is_prefilling: false,
        };

        let sampling_length = task.sampling_length;
        let (sampling_output, run_time) = self.run_model(task, sampling_method)?;
        let sampled_tokens =
            self.read_sampling_output(sampling_output.as_ref().expect("sampling output must exist"), sampling_length)?;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());
        let speculator_proposed = active_row_count.saturating_sub(1);
        let speculator_accepted = accepted_tokens.len().saturating_sub(1);

        self.update_cache_layers(&accepted_token_indices, None, false)?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
            speculator_proposed,
            speculator_accepted,
        })
    }

    /// Prepares async buffers for generation.
    /// Must be called after prefill, before async_generate loop.
    fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    ) {
        let prefill_count = self.tokens.len();

        self.context.async_buffers.prepare_positions(prefill_count, tokens_to_generate);
        self.context.async_buffers.prepare_seeds(&self.context.seed, prefill_count, tokens_to_generate);
        self.context.async_buffers.reset_counter();
        self.async_in_flight = (0..self.context.async_buffers.batch_size).map(|_| None).collect();
        self.async_token_ids = None;
    }

    /// Submits a single async forward pass.
    /// Does NOT block (except when GPU capture is enabled for the first decode).
    ///
    /// - `pass_idx`: Index of this pass (0, 1, 2, ...)
    /// - `sampling_method`: Sampling configuration
    /// - `on_complete`: Callback receiving sampled token as u64
    fn async_generate(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: Box<dyn FnOnce(u64) + Send>,
    ) -> Result<(), Error> {
        assert_eq!(self.decoding_config.generate_suffix_length(), 1, "async_generate only supports suffix_length=1");

        // Extract values from async_buffers before mutable borrow
        let current_counter = self.context.async_buffers.counter.get();
        let is_continuation = current_counter > 0;
        let batch_size = self.context.async_buffers.batch_size;
        let slot = pass_idx % batch_size;

        let last_token = *self.tokens.last().ok_or(Error::PrefillFailed)?;
        let token_position = self.context.async_buffers.positions.as_slice::<i32>()[pass_idx] as usize;
        let token_seed = self.context.async_buffers.seeds.as_slice::<u64>()[pass_idx];

        let task = Task {
            token_ids: &[last_token],
            token_subtrie_ranges: None,
            token_positions: &[token_position],
            token_bitmask: None,
            token_seeds: &[0], // Ignored, using async buffer
            active_row_count: 1,
            sampling_start: 0,
            sampling_length: 1,
            is_prefilling: false,
        };

        let token_ids_array = if pass_idx > 0 {
            Some(self.async_token_ids.take().expect("previous async pass must provide token_ids array"))
        } else {
            None
        };
        let sampling_inputs = SamplingInputs::from_slices(self.context.context.as_ref(), &[token_seed], None, None);

        let is_first_decode = !is_continuation;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);
        if should_capture {
            let _ = self.gpu_capture.start_capture(&self.context.context, "decode");
        }

        let batch_dim = task.active_row_count;
        let sampling_start = task.sampling_start;
        let sampling_length = task.sampling_length;
        let token_inputs = self.build_token_inputs(task, token_ids_array, None);
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;
        if is_continuation {
            encoder.encode_wait_for_event(&self.context.async_buffers.event, current_counter);
        }
        let resources = Self::encode_forward_pass_on(
            &self.context,
            &mut encoder,
            token_inputs,
            batch_dim,
            sampling_start,
            sampling_length,
            false,
            Some(sampling_method),
            &EncodingParameters::new(),
            Some(sampling_inputs),
        )?;

        // Copy sampled token: sampling_output → token_ids (for next pass)
        let token_ids_shape = [1];
        let token_ids_data_type = crate::DataType::U64;
        let mut async_token_ids_allocation = self
            .context
            .context
            .create_array_uninitialized(&token_ids_shape, token_ids_data_type, "async_token_id")
            .into_allocation();
        let (async_token_ids_buffer, async_token_ids_range) = async_token_ids_allocation.as_buffer_range();
        let async_token_ptr = SendPtr(unsafe {
            (async_token_ids_buffer.cpu_ptr().as_ptr() as *const u64)
                .add(async_token_ids_range.start / std::mem::size_of::<u64>())
        });
        self.context.token_copy_sampled.encode(
            resources.sampling_output.as_ref().expect("Sampling output must exist"),
            &mut async_token_ids_allocation,
            &mut encoder,
        );
        self.async_token_ids = Some(unsafe {
            Array::from_allocation(async_token_ids_allocation, 0, &token_ids_shape, token_ids_data_type)
        });

        // Scatter + register for all transformer layers
        self.context.cache_layers.borrow_mut().update_after_acceptance(
            &[0],
            None,
            &mut encoder,
            &self.context.kv_cache_update,
        );
        self.context.cache_layers.borrow_mut().register_accepted_tokens(1);

        // Signal event for next pass
        let next_counter = current_counter + 1;
        encoder.encode_signal_event(&self.context.async_buffers.event, next_counter);
        self.context.async_buffers.counter.set(next_counter);

        // Add completion handler
        let handler = move |result: Result<&<B::CommandBuffer as CommandBuffer>::Completed, B::Error>| {
            result.expect("async decoding forward pass completed with error");
            let token = unsafe { *async_token_ptr.as_ptr() };
            on_complete(token);
        };

        encoder.add_completion_handler(handler);

        let pending = encoder.end_encoding().submit();

        if should_capture {
            let completed = pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
            drop(resources);
            drop(completed);
            self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        } else {
            assert!(self.async_in_flight[slot].is_none(), "async slot {slot} still holds an in-flight state");
            self.async_in_flight[slot] = Some(InFlightForwardPass {
                resources,
                pending,
            });
        }

        Ok(())
    }

    fn finish_async(
        &mut self,
        pass_idx: usize,
    ) {
        let slot = pass_idx % self.context.async_buffers.batch_size;
        if let Some(in_flight) = self.async_in_flight[slot].take() {
            let InFlightForwardPass {
                resources,
                pending,
            } = in_flight;
            drop((resources, pending));
        }
    }

    fn clear_cache(&mut self) {}

    fn reset_state(&mut self) {
        self.context.cache_layers.borrow_mut().clear(self.context.context.as_ref());
        self.tokens.clear();
        self.registered_prefix_len = 0;
        self.async_in_flight.clear();
        self.async_token_ids = None;
        self.gpu_capture.reset();

        let seed = self.decoding_config.sampling_seed.resolve();
        self.context.seed = PRng::new(seed);
        self.context.async_buffers.reset_counter();
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.context.context.peak_memory_usage()
    }

    fn tokens_len(&self) -> usize {
        self.tokens.len()
    }
    fn tokens_push(
        &mut self,
        token: u64,
    ) {
        self.tokens.push(token);
    }
    fn generate_suffix_length(&self) -> usize {
        self.decoding_config.generate_suffix_length()
    }
    fn async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize {
        self.decoding_config.async_batch_size.resolve::<B>(model_path, self.context.context.as_ref())
    }

    fn get_slice(
        &self,
        range: Range<usize>,
    ) -> Option<Box<dyn Any>> {
        self.context.cache_layers.borrow().slice(&self.context.context, range).map(|s| Box::new(s) as Box<dyn Any>)
    }
    fn apply_slice(
        &mut self,
        slice: &dyn Any,
        range: Range<usize>,
    ) {
        let slice = slice.downcast_ref::<CacheLayersSlice<B>>().unwrap();
        self.context.cache_layers.borrow_mut().apply_slice(slice, Some(range));
    }

    fn build_llm_context(&self) -> Box<dyn Any> {
        let cache_layers = self.context.cache_layers.borrow().clone(&self.context.context);
        let context = LlmContext::new(self.tokens.clone(), cache_layers, self.decoding_config.clone());
        Box::new(context)
    }
    fn reconfigure_from_context(
        &mut self,
        context: &dyn Any,
    ) {
        let ctx = context.downcast_ref::<LlmContext<B>>().unwrap();
        self.context.cache_layers.borrow_mut().copy_from(&ctx.cache_layers, self.context.context.as_ref());
        self.tokens = ctx.tokens.clone();
        self.registered_prefix_len = self.tokens.len().saturating_sub(1);
    }
}

impl<B: Backend> LanguageModelGenerator<B> {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
        model_metadata: &ModelMetadata,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = LanguageModelGeneratorContext::new(model_path, &decoding_config, model_metadata)?;

        Ok(Self {
            decoding_config,
            tokens: Vec::new(),
            context,
            async_in_flight: Vec::new(),
            async_token_ids: None,
            registered_prefix_len: 0,
            gpu_capture,
        })
    }

    fn run_model(
        &mut self,
        task: Task,
        sampling_method: SamplingMethod,
    ) -> Result<(Option<Allocation<B>>, f64), Error> {
        let run_start = Instant::now();
        let sample = !task.is_prefilling;
        let is_prefilling = task.is_prefilling;
        let batch_dim = task.active_row_count;
        let sampling_start = task.sampling_start;
        let sampling_length = task.sampling_length;
        let sampling_inputs = sample.then(|| {
            self.create_sampling_inputs(task.sampling_start, task.sampling_length, task.token_seeds, task.token_bitmask)
        });

        let is_first_decode = task.token_ids.len() == 1;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);
        let token_inputs = self.build_token_inputs(task, None, None);

        if should_capture {
            self.gpu_capture.start_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        }

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;
        let resources = Self::encode_forward_pass_on(
            &self.context,
            &mut encoder,
            token_inputs,
            batch_dim,
            sampling_start,
            sampling_length,
            is_prefilling,
            sample.then_some(sampling_method),
            &EncodingParameters::new(),
            sampling_inputs,
        )?;

        let pending = encoder.end_encoding().submit();

        let completed = pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        let ForwardPassResources {
            token_inputs,
            logits,
            sampling_output,
            sampling_inputs,
        } = resources;
        drop((token_inputs, logits, sampling_inputs));
        drop(completed);

        let run_time = run_start.elapsed().as_secs_f64();

        if should_capture {
            self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        }

        Ok((sampling_output, run_time))
    }

    fn encode_forward_pass_on(
        context: &LanguageModelGeneratorContext<B>,
        encoder: &mut Encoder<B>,
        token_inputs: TokenInputs<B>,
        batch_dim: usize,
        sampling_start: usize,
        sampling_length: usize,
        is_prefilling: bool,
        sampling_method: Option<SamplingMethod>,
        parameters: &EncodingParameters,
        sampling_inputs: Option<SamplingInputs<B>>,
    ) -> Result<ForwardPassResources<B>, Error> {
        let mut sampling_output = sampling_method.map(|_| {
            context
                .context
                .create_array_uninitialized(&[sampling_length], crate::DataType::U32, "sampling_output")
                .into_allocation()
        });
        let mut logits = None;
        if is_prefilling {
            let mut cache_layers = context.cache_layers.borrow_mut();
            let decoder_arguments = token_inputs.decoder_arguments(
                &context.model_shape,
                context.shared_buffers.as_ref(),
                Some(&mut *cache_layers),
                batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                None,
            );
            context
                .executables
                .encode_prefill(decoder_arguments, token_inputs.token_ids(), parameters, encoder)
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        } else {
            let mut cache_layers = context.cache_layers.borrow_mut();
            let decoder_arguments = token_inputs.decoder_arguments(
                &context.model_shape,
                context.shared_buffers.as_ref(),
                Some(&mut *cache_layers),
                batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                None,
            );
            let mut retained_logits = context
                .executables
                .encode_decode(
                    decoder_arguments,
                    DecoderDecodeInput::TokenIds(token_inputs.token_ids()),
                    None,
                    parameters,
                    encoder,
                )
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

            let sampling_inputs = sampling_inputs.as_ref().expect("Sampling requires sampling inputs");
            let sampling_result = context.gpu_sampler.encode(
                SamplingArguments {
                    logits: &mut retained_logits,
                    seeds: sampling_inputs.seeds.allocation(),
                    bitmask: sampling_inputs.bitmask.as_ref().map(Array::allocation),
                    output: sampling_output.as_mut().expect("Sampling requires output allocation"),
                    sampling_method: sampling_method.expect("Sampling requires method"),
                    batch_size: sampling_length,
                    vocab_size: context.model_config.model_config.vocab_size,
                },
                encoder,
            );
            sampling_result.map_err(|e| Error::EncodeFailed(Box::new(e)))?;
            logits = Some(retained_logits);
        }

        Ok(ForwardPassResources {
            token_inputs,
            logits,
            sampling_output,
            sampling_inputs,
        })
    }

    fn build_token_inputs(
        &self,
        task: Task<'_>,
        token_ids_array: Option<Array<B>>,
        token_positions_array: Option<Array<B>>,
    ) -> TokenInputs<B> {
        TokenInputs::new_llm(
            self.context.context.as_ref(),
            &self.context.model_shape,
            task.token_ids,
            task.token_subtrie_ranges,
            task.token_positions,
            token_ids_array,
            token_positions_array,
            task.sampling_start,
            task.sampling_length,
        )
    }

    fn create_sampling_inputs(
        &self,
        sampling_start: usize,
        sampling_length: usize,
        token_seeds: &[u64],
        token_bitmask: Option<&[u32]>,
    ) -> SamplingInputs<B> {
        let seeds = &token_seeds[sampling_start..sampling_start + sampling_length];
        let bitmask_row_len = token_bitmask.map(|_| self.context.model_shape.bitmask_shape(1)[1]);
        let bitmask = token_bitmask.map(|mask| {
            let row_len = bitmask_row_len.expect("bitmask row length must exist");
            let start = sampling_start * row_len;
            let end = start + sampling_length * row_len;
            &mask[start..end]
        });
        SamplingInputs::from_slices(self.context.context.as_ref(), seeds, bitmask, bitmask_row_len)
    }

    fn read_sampling_output(
        &self,
        sampling_output: &Allocation<B>,
        batch_size: usize,
    ) -> Result<Vec<u64>, Error> {
        let (buffer, range) = sampling_output.as_buffer_range();
        let values = unsafe {
            std::slice::from_raw_parts(
                (buffer.cpu_ptr().as_ptr() as *const u32).add(range.start / std::mem::size_of::<u32>()),
                batch_size,
            )
        };
        Ok(values.iter().map(|value| *value as u64).collect())
    }

    fn update_cache_layers(
        &mut self,
        accepted_token_indices: &[usize],
        suffix_start: Option<usize>,
        wait_until_completed: bool,
    ) -> Result<(), Error> {
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        {
            let mut cache_layers = self.context.cache_layers.borrow_mut();
            cache_layers.update_after_acceptance(
                accepted_token_indices,
                suffix_start,
                &mut encoder,
                &self.context.kv_cache_update,
            );
        }

        let pending = encoder.end_encoding().submit();

        if wait_until_completed {
            pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        }
        Ok(())
    }

    fn sync_prefix(&mut self) {
        if self.tokens.is_empty() {
            return;
        }

        let desired_prefix_len = self.tokens.len() - 1;
        if desired_prefix_len > self.registered_prefix_len {
            let number_of_accepted_tokens = desired_prefix_len - self.registered_prefix_len;
            self.context.cache_layers.borrow_mut().register_accepted_tokens(number_of_accepted_tokens);
            self.registered_prefix_len = desired_prefix_len;
        }
    }
}
