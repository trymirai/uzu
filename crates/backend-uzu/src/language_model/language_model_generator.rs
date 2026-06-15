use std::{
    any::Any,
    iter::repeat_n,
    mem::size_of,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

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
        Allocation, AllocationType, AsBufferRangeRef, Backend, CommandBuffer, Context, DenseBuffer, Encoder, Pending,
        kernel::TokenCopySampledKernel,
    },
    config::model::language_model::LanguageModelConfig,
    data_type::DataType,
    encodable_block::DecoderDecodeInput,
    forward_pass::{cache_layers::CacheLayersSlice, kv_cache_layer::INVALID_POSITION, token_inputs::TokenInputs},
    language_model::grammar::CompiledGrammar,
    session::{
        config::DecodingConfig,
        helpers::Context as LlmContext,
        parameter::{ConfigResolvableValue, ResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::{FlatTrie, TrieCreationConfig, TrieNode},
    utils::pointers::SendPtr,
};

#[derive(Debug, Clone)]
pub struct Task {
    token_ids: Box<[u64]>,
    token_subtrie_ranges: Option<Box<[[u32; 3]]>>,
    token_positions: Box<[usize]>,
    token_bitmask: Option<Box<[u32]>>,
    token_seeds: Box<[u64]>,
    active_row_count: usize,
    sampling_start: usize,
    sampling_length: usize,
    is_prefilling: bool,
}

struct ForwardPassResources<B: Backend> {
    token_inputs: TokenInputs<B>,
    logits: Option<Allocation<B>>,
    sampling_output: Option<Allocation<B>>,
    sampling_seeds: Option<Allocation<B>>,
    sampling_bitmask: Option<Allocation<B>>,
}

struct InFlightForwardPass<B: Backend> {
    resources: ForwardPassResources<B>,
    pending: Pending<B>,
}

pub struct RunModelResult<B: Backend> {
    pub sampling_output: Option<Allocation<B>>,
    pub cpu_run_time: f64,
    #[allow(unused)]
    pub gpu_run_time: Duration,
}

struct PrefillStepRun<B: Backend> {
    sampling_output: Option<Allocation<B>>,
    run_time: f64,
    sampling_rows_are_local: bool,
}

#[derive(Clone, Copy)]
struct PrefillStepSchedule {
    sampling_start: usize,
    sampling_length: usize,
    accepted_prompt_rows: usize,
}

impl PrefillStepSchedule {
    fn new(
        should_sample: bool,
        tokens_length: usize,
        tokens_start_index: usize,
        tokens_end_index: usize,
        active_row_count: usize,
    ) -> Self {
        let step_end_token_index = std::cmp::min(tokens_end_index, tokens_length);
        let mut accepted_prompt_rows = step_end_token_index - tokens_start_index;

        let (sampling_start, sampling_length) = if should_sample {
            let suffix_root_index_in_step = (tokens_length - 1).saturating_sub(tokens_start_index);
            let sampling_length = active_row_count.saturating_sub(suffix_root_index_in_step);
            debug_assert!(sampling_length > 0, "Expected at least one token to sample on the last prefill step");
            accepted_prompt_rows = accepted_prompt_rows.saturating_sub(1);
            (suffix_root_index_in_step, sampling_length)
        } else {
            (0, 0)
        };

        Self {
            sampling_start,
            sampling_length,
            accepted_prompt_rows,
        }
    }

    fn should_sample(self) -> bool {
        self.sampling_length > 0
    }

    fn split_sampling_start(
        self,
        prefill_skips_trailing_layers: bool,
    ) -> Option<usize> {
        (prefill_skips_trailing_layers && self.sampling_start > 0).then_some(self.sampling_start)
    }
}

fn rebase_sampling_subtrie_ranges(
    ranges: &[[u32; 3]],
    row_offset: usize,
) -> Box<[[u32; 3]]> {
    let offset = u32::try_from(row_offset).expect("prefill split point must fit in u32");
    ranges[row_offset..]
        .iter()
        .map(|[start, end, height]| {
            [
                start.checked_sub(offset).expect("sampling subtrie start must be after prompt rows"),
                end.checked_sub(offset).expect("sampling subtrie end must be after prompt rows"),
                height.checked_sub(offset).expect("sampling subtrie height must be after prompt rows"),
            ]
        })
        .collect()
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

    fn reset_state(&mut self);
    fn peak_memory_usage(&self) -> Option<usize>;
    fn activation_data_type(&self) -> DataType;

    fn tokens_len(&self) -> usize;
    fn tokens_push(
        &mut self,
        token: u64,
    );
    fn generate_suffix_length(&self) -> usize;
    fn async_batch_size(
        &self,
        model_path: &Path,
    ) -> Result<usize, Error>;

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

    fn get_context_length(&self) -> usize;
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
            self.context.model_shape.vocabulary_size,
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
        let mut final_sampling_rows_are_local = false;
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
                step_token_subtrie_ranges.iter().position(|e| e.is_some()).map(|trie_start| {
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
                        .collect()
                });
            let step_token_positions = step_token_positions.collect::<Box<[usize]>>();
            let step_token_seeds = step_token_seeds.collect::<Box<[u64]>>();

            let active_row_count = step_token_positions.len();
            let is_last_prefill_step = step == prefill_steps - 1;
            let schedule = PrefillStepSchedule::new(
                sample_suffix && is_last_prefill_step,
                tokens_length,
                tokens_start_index,
                tokens_end_index,
                active_row_count,
            );
            let sampling_length = schedule.sampling_length;

            let step_token_bitmask: Option<Box<[u32]>> = if has_grammar && sampling_length > 0 {
                Some(
                    step_token_bitmasks
                        .flat_map(|mask| match mask {
                            Some(mask) => Either::Left(
                                mask.iter()
                                    .copied()
                                    .take(single_token_bitmask_size)
                                    .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                            ),
                            None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                        })
                        .collect::<Box<[u32]>>(),
                )
            } else {
                // Drain the chunk iterator to keep the other chunked iterators aligned.
                let _ = step_token_bitmasks.count();
                None
            };

            let should_capture = self.gpu_capture.should_capture_prefill(step == 0);

            if should_capture {
                self.gpu_capture
                    .start_capture(&self.context.context, "prefill")
                    .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
            }

            let _ = last_sampling_output.take();

            let task = Task {
                token_ids: step_token_ids,
                token_subtrie_ranges: step_token_subtrie_ranges,
                token_positions: step_token_positions,
                token_bitmask: step_token_bitmask,
                token_seeds: step_token_seeds,
                active_row_count,
                sampling_start: schedule.sampling_start,
                sampling_length,
                is_prefilling: !schedule.should_sample(),
            };

            let step_run = self.run_prefill_step(
                task,
                schedule,
                prefix_offset + tokens_start_index,
                single_token_bitmask_size,
                sampling_method,
            )?;
            if should_capture {
                self.gpu_capture
                    .stop_capture(&self.context.context, "prefill")
                    .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
            }

            last_sampling_length = sampling_length;
            final_sampling_rows_are_local = step_run.sampling_rows_are_local;
            last_sampling_output = step_run.sampling_output;
            run_times.push(step_run.run_time);
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

        let (accepted_tokens, accepted_token_indices) = flat_trie.accept(&sampled_tokens, compiled_grammar);
        if final_sampling_rows_are_local {
            self.update_cache_layers(&accepted_token_indices, None, false)?;
        } else {
            self.accept_sampled_prefill_rows(accepted_token_indices, suffix_root_index, last_suffix_start)?;
        }

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
            self.context.model_shape.vocabulary_size,
            &TrieCreationConfig::default(),
            suffix_length,
        );
        let flat_trie = suffix_root.linearize();

        let task = self.get_generate_task(&flat_trie, compiled_grammar.is_some());

        let active_row_count = task.active_row_count;
        let sampling_length = task.sampling_length;
        let run_result = self.run_model(task, sampling_method)?;
        let sampled_tokens = self.read_sampling_output(
            run_result.sampling_output.as_ref().expect("sampling output must exist"),
            sampling_length,
        )?;

        let (accepted_tokens, accepted_token_indices) = flat_trie.accept(&sampled_tokens, compiled_grammar);
        let speculator_proposed = active_row_count.saturating_sub(1);
        let speculator_accepted = accepted_tokens.len().saturating_sub(1);

        self.update_cache_layers(&accepted_token_indices, None, false)?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_result.cpu_run_time,
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
            token_ids: Box::new([last_token]),
            token_subtrie_ranges: None,
            token_positions: Box::new([token_position]),
            token_bitmask: None,
            token_seeds: Box::new([0]), // Ignored, using async buffer
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
        let sampling_seed = [token_seed];
        let mut sampling_seeds = self
            .context
            .context
            .create_allocation(std::mem::size_of_val(&sampling_seed), AllocationType::Global)
            .expect("Failed to create sampling seed allocation");
        sampling_seeds.copyin(&sampling_seed);
        let repetition_penalty = Self::repetition_penalty_config(sampling_method);
        if pass_idx == 0
            && let Some((_, suffix_repetition_length)) = repetition_penalty
        {
            let capacity = self.context.async_buffers.repetition_context_ring_capacity;
            Self::copy_repetition_context_ring(
                &mut self.context.async_buffers.repetition_context_ring,
                &self.tokens,
                suffix_repetition_length,
                capacity,
            );
        }
        let sampling_context_ring =
            repetition_penalty.as_ref().map(|_| &self.context.async_buffers.repetition_context_ring);

        let is_first_decode = !is_continuation;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);
        if should_capture {
            self.gpu_capture
                .start_capture(&self.context.context, "decode")
                .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
        }

        let batch_dim = task.active_row_count;
        let sampling_start = task.sampling_start;
        let sampling_length = task.sampling_length;
        let token_inputs = self.build_token_inputs(task, token_ids_array);
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;
        let resources = Self::encode_forward_pass_on(
            &self.context,
            &mut encoder,
            token_inputs,
            batch_dim,
            sampling_start,
            sampling_length,
            false,
            Some(sampling_method),
            Some(sampling_seeds),
            None,
            sampling_context_ring,
        )?;

        // Copy sampled token: sampling_output → token_ids (for next pass)
        let token_ids_shape = [1];
        let token_ids_data_type = DataType::U64;
        let mut async_token_ids_allocation =
            self.context.context.create_array_uninitialized(&token_ids_shape, token_ids_data_type).into_allocation();
        let async_token_ids_buffer_range = async_token_ids_allocation.as_buffer_range_ref();
        let async_token_ids_range = async_token_ids_buffer_range.range();
        let async_token_ptr = SendPtr(unsafe {
            (async_token_ids_buffer_range.buffer().cpu_ptr().as_ptr() as *const u64)
                .add(async_token_ids_range.start / std::mem::size_of::<u64>())
        });
        if let Some((_, suffix_repetition_length)) = repetition_penalty {
            self.context.token_copy_sampled_context_ring.encode(
                resources.sampling_output.as_ref().expect("Sampling output must exist"),
                &mut async_token_ids_allocation,
                Some(&mut self.context.async_buffers.repetition_context_ring),
                Some(suffix_repetition_length as u32),
                &mut encoder,
            );
        } else {
            self.context.token_copy_sampled.encode(
                resources.sampling_output.as_ref().expect("Sampling output must exist"),
                &mut async_token_ids_allocation,
                None::<&mut Allocation<B>>,
                None,
                &mut encoder,
            );
        }
        self.async_token_ids = Some(unsafe {
            Array::from_allocation(async_token_ids_allocation, 0, &token_ids_shape, token_ids_data_type)
        });

        // Scatter + register for all transformer layers
        self.context.cache_layers.borrow_mut().update_after_acceptance(
            &[0],
            None,
            &self.context.context,
            &mut encoder,
            &self.context.kv_cache_update,
        );
        self.context.cache_layers.borrow_mut().register_accepted_tokens(1);

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
            self.gpu_capture
                .stop_capture(&self.context.context, "decode")
                .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
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

    fn activation_data_type(&self) -> DataType {
        self.context.model_shape.data_type
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
    ) -> Result<usize, Error> {
        self.decoding_config
            .async_batch_size
            .resolve::<B>(model_path, self.context.context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))
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
        self.context.cache_layers.borrow_mut().apply_slice(&self.context.context, slice, Some(range));
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

    fn get_context_length(&self) -> usize {
        self.context.get_context_length(&self.decoding_config)
    }
}

impl<B: Backend> LanguageModelGenerator<B> {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
        model_config: &LanguageModelConfig,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = LanguageModelGeneratorContext::new(model_path, &decoding_config, model_config)?;

        let mut generator = Self {
            decoding_config,
            tokens: Vec::new(),
            context,
            async_in_flight: Vec::new(),
            async_token_ids: None,
            registered_prefix_len: 0,
            gpu_capture,
        };
        generator.warmup();
        Ok(generator)
    }

    fn warmup(&mut self) {
        if let Err(error) = self.run_warmup() {
            eprintln!("uzu: model warmup skipped (kernels will compile on first use): {error}");
        }
        self.reset_state();
    }

    fn run_warmup(&mut self) -> Result<(), Error> {
        let context_length = self.context.get_context_length(&self.decoding_config);
        let first = 64.min(context_length.saturating_sub(1)).max(1);
        self.prefill(vec![0u64; first], None, SamplingMethod::Greedy, 0, true)?;
        self.generate(None, SamplingMethod::Greedy)?;
        for &len in &[65usize, 16] {
            let offset = self.tokens.len();
            if offset + len >= context_length {
                break;
            }
            self.prefill(vec![0u64; len], None, SamplingMethod::Greedy, offset, false)?;
        }
        Ok(())
    }

    fn repetition_penalty_config(sampling_method: SamplingMethod) -> Option<(f32, usize)> {
        match sampling_method {
            SamplingMethod::Stochastic {
                repetition_penalty: Some(repetition_penalty),
                suffix_repetition_length,
                ..
            } => Some((
                repetition_penalty,
                suffix_repetition_length.expect("suffix_repetition_length is required for repetition_penalty"),
            )),
            _ => None,
        }
    }

    fn copy_repetition_context_ring(
        context_ring: &mut Allocation<B>,
        tokens: &[u64],
        suffix_repetition_length: usize,
        capacity: usize,
    ) {
        let ring_length = tokens.len().min(suffix_repetition_length);
        let mut packed_ring = vec![0u32; 2 + capacity];
        packed_ring[1] = ring_length as u32;
        for (index, token) in tokens[tokens.len() - ring_length..].iter().enumerate() {
            packed_ring[2 + index] = *token as u32;
        }
        context_ring.copyin(&packed_ring);
    }

    pub fn get_generate_task(
        &self,
        flat_trie: &FlatTrie,
        has_grammar: bool,
    ) -> Task {
        let suffix_length = self.decoding_config.generate_suffix_length();
        let active_row_count = flat_trie.len();

        let token_ids = flat_trie.token_ids().chain(repeat_n(0, suffix_length - active_row_count)).collect();

        let token_subtrie_ranges = flat_trie
            .token_subtrie_ranges()
            .chain(repeat_n([u32::MAX, u32::MAX, u32::MAX], suffix_length - active_row_count))
            .collect();

        let token_bitmask: Option<Box<[u32]>> = has_grammar.then(|| {
            let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
            flat_trie
                .token_masks()
                .chain(repeat_n(None, suffix_length - active_row_count))
                .flat_map(|mask| match mask {
                    Some(mask) => Either::Left(
                        mask.iter()
                            .copied()
                            .take(single_token_bitmask_size)
                            .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                    ),
                    None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                })
                .collect::<Box<[u32]>>()
        });

        let start_position = self.tokens.len() - 1;
        let token_positions = flat_trie
            .token_positions()
            .map(|trie_position| start_position + trie_position)
            .chain(repeat_n(INVALID_POSITION, suffix_length - active_row_count))
            .collect();

        let token_seeds = flat_trie.token_seeds().chain(repeat_n(0, suffix_length - active_row_count)).collect();

        Task {
            token_ids,
            token_subtrie_ranges: Some(token_subtrie_ranges),
            token_positions,
            token_bitmask,
            token_seeds,
            active_row_count,
            sampling_start: 0,
            sampling_length: active_row_count,
            is_prefilling: false,
        }
    }

    pub fn run_model(
        &mut self,
        task: Task,
        sampling_method: SamplingMethod,
    ) -> Result<RunModelResult<B>, Error> {
        let run_start = Instant::now();
        let sample = !task.is_prefilling;
        let is_prefilling = task.is_prefilling;
        let batch_dim = task.active_row_count;
        let sampling_start = task.sampling_start;
        let sampling_length = task.sampling_length;
        let repetition_penalty = if sample {
            Self::repetition_penalty_config(sampling_method)
        } else {
            None
        };
        let sampling_context_ring = if let Some((_, suffix_repetition_length)) = repetition_penalty {
            let mut allocation = self
                .context
                .context
                .create_allocation((2 + suffix_repetition_length) * size_of::<u32>(), AllocationType::Global)
                .map_err(|e| Error::UnableToCreateContext(e.into()))?;
            Self::copy_repetition_context_ring(
                &mut allocation,
                &self.tokens,
                suffix_repetition_length,
                suffix_repetition_length,
            );
            Some(allocation)
        } else {
            None
        };
        let (sampling_seeds, sampling_bitmask) = if sample {
            let seeds = &task.token_seeds[task.sampling_start..task.sampling_start + task.sampling_length];
            let mut seed_allocation = self
                .context
                .context
                .create_allocation(std::mem::size_of_val(seeds), AllocationType::Global)
                .expect("Failed to create sampling seed allocation");
            seed_allocation.copyin(seeds);

            let bitmask_allocation = task.token_bitmask.as_deref().map(|mask| {
                let row_len = self.context.model_shape.bitmask_shape(1)[1];
                let start = task.sampling_start * row_len;
                let end = start + task.sampling_length * row_len;
                let bitmask = &mask[start..end];
                let mut allocation = self
                    .context
                    .context
                    .create_allocation(std::mem::size_of_val(bitmask), AllocationType::Global)
                    .expect("Failed to create sampling bitmask allocation");
                allocation.copyin(bitmask);
                allocation
            });

            (Some(seed_allocation), bitmask_allocation)
        } else {
            (None, None)
        };

        let is_first_decode = task.token_ids.len() == 1;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);
        let token_inputs = self.build_token_inputs(task, None);

        if should_capture {
            self.gpu_capture
                .start_capture(&self.context.context, "decode")
                .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
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
            sampling_seeds,
            sampling_bitmask,
            sampling_context_ring.as_ref(),
        )?;

        let pending = encoder.end_encoding().submit();

        let completed = pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        let gpu_run_time = completed.gpu_execution_time();
        let ForwardPassResources {
            token_inputs,
            logits,
            sampling_output,
            sampling_seeds,
            sampling_bitmask,
        } = resources;
        drop((token_inputs, logits, sampling_seeds, sampling_bitmask));
        drop(completed);

        let run_time = run_start.elapsed().as_secs_f64();

        if should_capture {
            self.gpu_capture
                .stop_capture(&self.context.context, "decode")
                .map_err(|error| Error::CaptureFailed(Box::new(error)))?;
        }

        Ok(RunModelResult {
            sampling_output,
            cpu_run_time: run_time,
            gpu_run_time,
        })
    }

    fn run_prefill_step(
        &mut self,
        task: Task,
        schedule: PrefillStepSchedule,
        prefix_start_token_index: usize,
        bitmask_row_len: usize,
        sampling_method: SamplingMethod,
    ) -> Result<PrefillStepRun<B>, Error> {
        let Some(prompt_row_count) =
            schedule.split_sampling_start(self.context.model_shape.prefill_skips_trailing_layers())
        else {
            let run_result = self.run_model(task, sampling_method)?;
            self.accept_prefilled_prompt_rows(
                schedule.accepted_prompt_rows,
                prefix_start_token_index + schedule.accepted_prompt_rows,
            )?;
            return Ok(PrefillStepRun {
                sampling_output: run_result.sampling_output,
                run_time: run_result.cpu_run_time,
                sampling_rows_are_local: false,
            });
        };

        let prefix_task = Task {
            token_ids: task.token_ids[..prompt_row_count].to_vec().into_boxed_slice(),
            token_subtrie_ranges: None,
            token_positions: task.token_positions[..prompt_row_count].to_vec().into_boxed_slice(),
            token_bitmask: None,
            token_seeds: task.token_seeds[..prompt_row_count].to_vec().into_boxed_slice(),
            active_row_count: prompt_row_count,
            sampling_start: 0,
            sampling_length: 0,
            is_prefilling: true,
        };
        let prefix_run = self.run_model(prefix_task, sampling_method)?;
        self.accept_prefilled_prompt_rows(prompt_row_count, prefix_start_token_index + prompt_row_count)?;

        let suffix_token_ids = task.token_ids[prompt_row_count..].to_vec().into_boxed_slice();
        let suffix_token_bitmask = task
            .token_bitmask
            .as_ref()
            .map(|mask| mask[prompt_row_count * bitmask_row_len..].to_vec().into_boxed_slice());
        let suffix_token_subtrie_ranges =
            task.token_subtrie_ranges.as_ref().map(|ranges| rebase_sampling_subtrie_ranges(ranges, prompt_row_count));
        let suffix_active_row_count = suffix_token_ids.len();
        let suffix_task = Task {
            token_ids: suffix_token_ids,
            token_subtrie_ranges: suffix_token_subtrie_ranges,
            token_positions: task.token_positions[prompt_row_count..].to_vec().into_boxed_slice(),
            token_bitmask: suffix_token_bitmask,
            token_seeds: task.token_seeds[prompt_row_count..].to_vec().into_boxed_slice(),
            active_row_count: suffix_active_row_count,
            sampling_start: 0,
            sampling_length: schedule.sampling_length,
            is_prefilling: false,
        };
        let suffix_run = self.run_model(suffix_task, sampling_method)?;

        Ok(PrefillStepRun {
            sampling_output: suffix_run.sampling_output,
            run_time: prefix_run.cpu_run_time + suffix_run.cpu_run_time,
            sampling_rows_are_local: true,
        })
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
        sampling_seeds: Option<Allocation<B>>,
        sampling_bitmask: Option<Allocation<B>>,
        sampling_context_ring: Option<&Allocation<B>>,
    ) -> Result<ForwardPassResources<B>, Error> {
        let mut sampling_output = None;
        let mut logits = None;
        if is_prefilling {
            let mut cache_layers = context.cache_layers.borrow_mut();
            cache_layers.prepare_for_forward_pass(&context.context, batch_dim);
            let decoder_arguments = token_inputs.decoder_arguments(
                Some(&mut *cache_layers),
                batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                None,
            );
            context
                .executables
                .encode_prefill(
                    decoder_arguments,
                    token_inputs.token_ids(),
                    context.model_shape.prefill_layer_count(),
                    encoder,
                )
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        } else {
            let mut cache_layers = context.cache_layers.borrow_mut();
            cache_layers.prepare_for_forward_pass(&context.context, batch_dim);
            let decoder_arguments = token_inputs.decoder_arguments(
                Some(&mut *cache_layers),
                batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                None,
            );
            let retained_logits = context
                .executables
                .encode_decode(decoder_arguments, DecoderDecodeInput::TokenIds(token_inputs.token_ids()), None, encoder)
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

            let sampling_seeds = sampling_seeds.as_ref().expect("Sampling requires seeds");
            let output = context.gpu_sampler.encode(
                &retained_logits,
                Some(sampling_seeds),
                sampling_bitmask.as_ref(),
                sampling_context_ring,
                Some(token_inputs.token_ids()),
                sampling_method.expect("Sampling requires method"),
                sampling_length,
                encoder,
            );
            sampling_output = Some(output.map_err(|e| Error::EncodeFailed(Box::new(e)))?);
            logits = Some(retained_logits);
        }

        Ok(ForwardPassResources {
            token_inputs,
            logits,
            sampling_output,
            sampling_seeds,
            sampling_bitmask,
        })
    }

    fn build_token_inputs(
        &self,
        task: Task,
        token_ids_array: Option<Array<B>>,
    ) -> TokenInputs<B> {
        TokenInputs::new_llm(
            self.context.context.as_ref(),
            &self.context.model_shape,
            &task.token_ids,
            task.token_subtrie_ranges.as_deref(),
            &task.token_positions,
            token_ids_array,
            task.sampling_start,
            task.sampling_length,
        )
    }

    fn read_sampling_output(
        &self,
        sampling_output: &Allocation<B>,
        batch_size: usize,
    ) -> Result<Vec<u64>, Error> {
        let values = sampling_output.copyout::<u32>();
        Ok(values[..batch_size].iter().map(|value| u64::from(*value)).collect())
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
                &self.context.context,
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

    fn accept_prefilled_prompt_rows(
        &mut self,
        row_count: usize,
        registered_prefix_len: usize,
    ) -> Result<(), Error> {
        if row_count == 0 {
            return Ok(());
        }

        self.update_cache_layers(&(0..row_count).collect::<Vec<_>>(), None, true)?;
        self.context.cache_layers.borrow_mut().register_accepted_tokens(row_count);
        self.registered_prefix_len = registered_prefix_len;
        Ok(())
    }

    fn accept_sampled_prefill_rows(
        &mut self,
        accepted_token_indices: Vec<usize>,
        suffix_root_index: usize,
        suffix_start: usize,
    ) -> Result<(), Error> {
        let accepted_rows =
            accepted_token_indices.into_iter().map(|row| suffix_root_index + row).collect::<Box<[usize]>>();
        self.update_cache_layers(&accepted_rows, Some(suffix_start), false)
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

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::PrefillStepSchedule;

    #[uzu_test]
    fn cache_only_prefill_step_accepts_prompt_rows_without_sampling() {
        let schedule = PrefillStepSchedule::new(false, 10, 8, 16, 8);

        assert_eq!(schedule.sampling_start, 0);
        assert_eq!(schedule.sampling_length, 0);
        assert_eq!(schedule.accepted_prompt_rows, 2);
    }

    #[uzu_test]
    fn sampling_prefill_step_keeps_suffix_root_for_sampling() {
        let schedule = PrefillStepSchedule::new(true, 10, 8, 16, 8);

        assert_eq!(schedule.sampling_start, 1);
        assert_eq!(schedule.sampling_length, 7);
        assert_eq!(schedule.accepted_prompt_rows, 1);
    }

    #[uzu_test]
    fn sampling_prefill_step_on_chunk_boundary_keeps_final_prompt_row_for_sampling() {
        let schedule = PrefillStepSchedule::new(true, 16, 8, 16, 8);

        assert_eq!(schedule.sampling_start, 7);
        assert_eq!(schedule.sampling_length, 1);
        assert_eq!(schedule.accepted_prompt_rows, 7);
    }
}

#[cfg(test)]
#[path = "../../unit/language_model/language_model_generator_bench.rs"]
mod bench_tests;
