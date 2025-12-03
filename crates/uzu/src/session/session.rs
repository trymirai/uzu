use std::{
    fs::File,
    io::BufReader,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::Instant,
};

fn nanos_to_secs(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

use objc2::rc::autoreleasepool;
use tokenizers::Tokenizer;

use crate::{
    backends::metal::forward_pass::cache_layers::CacheLayer,
    config::{ModelMetadata, decoder_layer::MixerConfig},
    generator::{
        generator::Generator,
        result::{GenerateResult, PrefillResult},
    },
    session::{
        config::{DecodingConfig, RunConfig},
        helpers::{
            Context, InputProcessor, InputProcessorDefault, OutputParser,
            is_directory_fits_ram,
        },
        parameter::{ConfigResolvableValue, ContextMode, SamplingMethod},
        types::{
            Error, FinishReason, Input, Output, RunStats, Stats, StepStats,
            TotalStats,
        },
    },
};

struct RunContext {
    eos_tokens: Vec<u64>,
    context_length: usize,
    prefix_len_before: usize,
    input_tokens_len: usize,
    tokens_limit: usize,
    prefill_result: PrefillResult,
    prefill_duration: f64,
    prefill_suffix_length: usize,
    run_start: Instant,
}

pub struct Session {
    pub model_path: PathBuf,
    pub model_metadata: ModelMetadata,

    tokenizer: Tokenizer,
    input_processor: Box<dyn InputProcessor>,
    output_parser: OutputParser,
    generator: Option<Generator>,
    static_context: Option<Context>,
}

impl Session {
    pub fn new(
        model_path: PathBuf,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        if !is_directory_fits_ram(&model_path) {
            return Err(Error::NotEnoughMemory);
        }

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file =
            File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata = serde_json::from_reader(
            BufReader::new(config_file),
        )
        .map_err(|err| {
            eprintln!("Failed to parse config.json: {err}");
            Error::UnableToLoadConfig
        })?;

        let is_ssm = model_metadata
            .clone()
            .model_config
            .decoder_config
            .layer_configs
            .unwrap_or(Box::new([]))
            .iter()
            .any(|layer| matches!(layer.mixer_config, MixerConfig::Mamba(_)));
        if is_ssm {
            match decoding_config.context_mode {
                ContextMode::None => {},
                ContextMode::Static {
                    ..
                } => {
                    return Err(Error::UnsupportedContextModeForModel);
                },
                ContextMode::Dynamic => {
                    return Err(Error::UnsupportedContextModeForModel);
                },
            }

            if decoding_config.speculator_config.number_of_speculated_tokens > 0
            {
                return Err(Error::UnsupportedSpeculatorConfigForModel);
            }
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|_| Error::UnableToLoadTokenizer)?;

        let input_processor = InputProcessorDefault::new(
            model_metadata.model_config.message_processor_config.clone(),
        );

        let output_parser = OutputParser::new(
            model_metadata
                .model_config
                .message_processor_config
                .output_parser_regex
                .clone(),
        )?;

        let generator = Generator::new(&model_path, decoding_config)
            .map_err(Error::from)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            input_processor: Box::new(input_processor),
            output_parser,
            generator: Some(generator),
            static_context: None,
        })
    }

    pub fn run<F>(
        &mut self,
        input: Input,
        config: RunConfig,
        progress: Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        autoreleasepool(|_| {
            let context_mode = self
                .generator
                .as_ref()
                .ok_or(Error::GeneratorNotLoaded)?
                .decoding_config
                .context_mode
                .clone();

            match &context_mode {
                ContextMode::None => {
                    let generator = self
                        .generator
                        .as_mut()
                        .ok_or(Error::GeneratorNotLoaded)?;
                    generator.reset_state();
                },
                ContextMode::Static {
                    input,
                } => {
                    if self.static_context.is_none() {
                        let mut prefill_config = config.clone();
                        prefill_config.tokens_limit = 0;
                        let (_out, run_context) =
                            self.extend(input.clone(), None, prefill_config)?;
                        self.static_context = Some(run_context);
                    }
                    let tmp = self.static_context.take();
                    if let Some(ref run_context) = tmp {
                        self.reconfigure_generator(Some(run_context))?;
                    }
                    self.static_context = tmp;
                },
                ContextMode::Dynamic => {},
            }

            let output = self.run_internal(input, config, progress)?;

            match context_mode {
                ContextMode::None
                | ContextMode::Static {
                    ..
                } => {
                    let generator = self
                        .generator
                        .as_mut()
                        .ok_or(Error::GeneratorNotLoaded)?;
                    generator.reset_state();
                },
                ContextMode::Dynamic => {},
            }
            Ok(output)
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        autoreleasepool(|_| generator.reset_state());
        Ok(())
    }
}

impl Session {
    fn extend(
        &mut self,
        input: Input,
        context: Option<&Context>,
        config: RunConfig,
    ) -> Result<(Output, Context), Error> {
        self.reconfigure_generator(context)?;
        let output =
            self.run_internal(input, config, None::<fn(Output) -> bool>)?;
        let new_context = self.build_context_from_generator()?;
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok((output, new_context))
    }

    fn run_internal<F>(
        &mut self,
        input: Input,
        config: RunConfig,
        progress: Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;

        let run_start = Instant::now();
        let text =
            self.input_processor.process(&input, config.enable_thinking)?;
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();

        let context_length = generator
            .decoding_config
            .context_length
            .resolve(&self.model_metadata.model_config);
        if tokens.len() >= context_length {
            return Err(Error::ContextLengthExceeded);
        }

        let eos_tokens: Vec<u64> = self
            .model_metadata
            .model_config
            .generation_config
            .stop_token_ids
            .iter()
            .map(|&x| x as u64)
            .collect();

        let sampling_method =
            config.sampling_policy.resolve(&self.model_metadata.model_config);

        let prefill_start = Instant::now();
        let prefix_offset = generator.tokens.len();
        let prefix_len_before = prefix_offset.saturating_sub(1);

        let sample_suffix = config.tokens_limit > 0;
        let prefill_result = generator.prefill(
            tokens.clone(),
            sampling_method,
            prefix_offset,
            sample_suffix,
        )?;
        let prefill_tokens = prefill_result.tokens.clone();
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        generator.clear_cache();

        let prefill_suffix_length = generator
            .decoding_config
            .prefill_step_size
            .resolve(&self.model_metadata.model_config);

        let run_context = RunContext {
            eos_tokens,
            context_length,
            prefix_len_before,
            input_tokens_len: tokens.len(),
            tokens_limit: config.tokens_limit as usize,
            prefill_result: prefill_result.clone(),
            prefill_duration,
            prefill_suffix_length,
            run_start,
        };

        let prefill_finish_reason =
            Self::check_finish_reason(&run_context, generator, &prefill_tokens);
        let prefill_output = Self::build_output(
            &self.model_metadata,
            &self.tokenizer,
            &self.output_parser,
            &run_context,
            generator,
            &[],
            &[],
            prefill_finish_reason.clone(),
        )?;

        let prefill_should_continue = if let Some(ref p) = progress {
            p(prefill_output.clone())
        } else {
            true
        };

        if !prefill_should_continue || prefill_finish_reason.is_some() {
            if prefill_should_continue {
                return Ok(prefill_output);
            } else {
                return Ok(prefill_output
                    .clone_with_finish_reason(Some(FinishReason::Cancelled)));
            }
        }

        let can_use_async = generator.decoding_config.generate_suffix_length()
            == 1
            && !generator.has_attention_layers();

        let generate_output = if can_use_async {
            let batch_size = generator
                .decoding_config
                .async_batch_size
                .resolve(&self.model_path);
            let (results, durations, finish_reason) = Self::run_async_batch(
                &self.model_metadata,
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                generator,
                sampling_method,
                &progress,
                batch_size,
            )?;
            Self::build_output(
                &self.model_metadata,
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                generator,
                &results,
                &durations,
                Some(finish_reason),
            )?
        } else {
            Self::run_sync_generate(
                &self.model_metadata,
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                generator,
                sampling_method,
                &progress,
            )?
        };

        generator.clear_cache();
        Ok(generate_output
            .clone_with_duration(run_start.elapsed().as_secs_f64()))
    }

    fn reconfigure_generator(
        &mut self,
        context: Option<&Context>,
    ) -> Result<(), Error> {
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        generator.reset_state();
        if let Some(run_context) = context {
            let mut generator_state =
                generator.context.cache_layers.borrow_mut();
            for (run_context_layer, gen_layer) in run_context
                .cache_layers
                .data
                .iter()
                .zip(generator_state.data.iter_mut())
            {
                match (run_context_layer, gen_layer) {
                    (
                        CacheLayer::Transformer(src),
                        CacheLayer::Transformer(dst),
                    ) => {
                        let copy_rows = src.prefix_segment_length();
                        if copy_rows > 0 {
                            {
                                let mut dst_keys = dst.keys.borrow_mut();
                                let src_keys = src.keys.borrow();
                                dst_keys.copy_slice(
                                    &src_keys,
                                    1,
                                    0..copy_rows,
                                    0,
                                );
                            }
                            {
                                let mut dst_values = dst.values.borrow_mut();
                                let src_values = src.values.borrow();
                                dst_values.copy_slice(
                                    &src_values,
                                    1,
                                    0..copy_rows,
                                    0,
                                );
                            }
                        }
                        dst.state = src.state.clone();
                        dst.prefix_token_positions =
                            src.prefix_token_positions.clone();
                    },
                    (
                        CacheLayer::StateSpace(src),
                        CacheLayer::StateSpace(dst),
                    ) => {
                        {
                            let mut dst_conv = dst.conv_state.borrow_mut();
                            let src_conv = src.conv_state.borrow();
                            dst_conv.copy_from_array(&src_conv);
                        }
                        {
                            let mut dst_ssm = dst.ssm_state.borrow_mut();
                            let src_ssm = src.ssm_state.borrow();
                            dst_ssm.copy_from_array(&src_ssm);
                        }
                    },
                    _ => panic!(
                        "Layer type mismatch when reconfiguring generator cache"
                    ),
                }
            }
            drop(generator_state);

            generator.tokens = run_context.tokens.clone();
        }
        Ok(())
    }

    fn build_context_from_generator(&self) -> Result<Context, Error> {
        let generator =
            self.generator.as_ref().ok_or(Error::GeneratorNotLoaded)?;
        let cache_layers = generator
            .context
            .cache_layers
            .borrow()
            .clone(&generator.context.mtl_context);
        let context = Context::new(
            generator.tokens.clone(),
            cache_layers,
            generator.decoding_config.clone(),
        );
        Ok(context)
    }
}

impl Session {
    fn decode_generated_tokens(
        tokenizer: &Tokenizer,
        generator: &Generator,
        run_context: &RunContext,
    ) -> Result<String, Error> {
        let start_idx =
            run_context.prefix_len_before + run_context.input_tokens_len;
        let generated_tokens: Vec<u32> =
            generator.tokens[start_idx..].iter().map(|&v| v as u32).collect();
        tokenizer
            .decode(&generated_tokens, true)
            .map_err(|_| Error::UnableToDecodeText)
    }

    fn check_finish_reason(
        run_context: &RunContext,
        generator: &Generator,
        new_tokens: &[u64],
    ) -> Option<FinishReason> {
        let start_idx =
            run_context.prefix_len_before + run_context.input_tokens_len;
        let total_new_tokens = generator.tokens[start_idx..].len();
        let has_eos =
            new_tokens.iter().any(|t| run_context.eos_tokens.contains(t));
        let context_limit =
            generator.tokens.len() >= run_context.context_length;

        if has_eos {
            Some(FinishReason::Stop)
        } else if total_new_tokens >= run_context.tokens_limit {
            Some(FinishReason::Length)
        } else if context_limit {
            Some(FinishReason::ContextLimitReached)
        } else {
            None
        }
    }

    fn build_output(
        model_metadata: &ModelMetadata,
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        generator: &Generator,
        generate_results: &[GenerateResult],
        generate_durations: &[f64],
        finish_reason: Option<FinishReason>,
    ) -> Result<Output, Error> {
        let text =
            Self::decode_generated_tokens(tokenizer, generator, run_context)?;
        let parsed = output_parser.parse(text);
        let start_idx =
            run_context.prefix_len_before + run_context.input_tokens_len;
        let output_tokens = generator.tokens[start_idx..].len();

        Ok(Output {
            text: parsed,
            stats: Self::build_stats(
                model_metadata,
                run_context.prefill_result.clone(),
                run_context.prefill_duration,
                run_context.prefill_suffix_length,
                generate_results.to_vec(),
                generate_durations.to_vec(),
                generator.decoding_config.generate_suffix_length(),
                run_context.run_start.elapsed().as_secs_f64(),
                run_context.input_tokens_len,
                output_tokens,
            ),
            finish_reason,
        })
    }

    fn run_async_batch<F>(
        model_metadata: &ModelMetadata,
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        generator: &mut Generator,
        sampling_method: SamplingMethod,
        progress: &Option<F>,
        batch_size: usize,
    ) -> Result<(Vec<GenerateResult>, Vec<f64>, FinishReason), Error>
    where
        F: Fn(Output) -> bool,
    {
        let remaining_by_limit = run_context
            .tokens_limit
            .saturating_sub(run_context.prefill_result.tokens.len());
        let remaining_by_context =
            run_context.context_length.saturating_sub(generator.tokens.len());
        let tokens_to_generate = remaining_by_limit.min(remaining_by_context);

        generator.prepare_async(tokens_to_generate);

        let (sender, receiver) = mpsc::channel::<(usize, u64, u64)>();
        let start_time = Instant::now();
        let last_nanos = Arc::new(AtomicU64::new(0));

        let mut results: Vec<GenerateResult> =
            Vec::with_capacity(tokens_to_generate);
        let mut durations: Vec<f64> = Vec::with_capacity(tokens_to_generate);
        let mut finish_reason = FinishReason::Length;
        let mut next_to_submit = 0;
        let mut in_flight = 0;

        while in_flight > 0 || next_to_submit < tokens_to_generate {
            let batch_end =
                std::cmp::min(next_to_submit + batch_size, tokens_to_generate);
            for idx in next_to_submit..batch_end {
                let batch_sender = sender.clone();
                let last_callback_nanos = last_nanos.clone();
                let batch_start_time = start_time;
                generator.async_generate(
                    idx,
                    sampling_method,
                    move |token| {
                        let now_nanos =
                            batch_start_time.elapsed().as_nanos() as u64;
                        let prev_nanos = last_callback_nanos
                            .swap(now_nanos, Ordering::SeqCst);
                        let _ = batch_sender.send((
                            idx,
                            token,
                            now_nanos.saturating_sub(prev_nanos),
                        ));
                    },
                )?;
                in_flight += 1;
            }
            let batch_submitted = batch_end - next_to_submit;
            next_to_submit = batch_end;

            for _ in 0..batch_submitted {
                let (_, token, duration_nanos) =
                    receiver.recv().map_err(|_| Error::SamplingFailed)?;
                in_flight -= 1;

                let duration = nanos_to_secs(duration_nanos);
                generator.tokens.push(token);
                results.push(GenerateResult {
                    tokens: vec![token],
                    forwardpass_duration: duration,
                });
                durations.push(duration);

                let should_stop = if run_context.eos_tokens.contains(&token) {
                    finish_reason = FinishReason::Stop;
                    true
                } else if generator.tokens.len() >= run_context.context_length {
                    finish_reason = FinishReason::ContextLimitReached;
                    true
                } else if let Some(progress_fn) = progress {
                    let output = Self::build_output(
                        model_metadata,
                        tokenizer,
                        output_parser,
                        run_context,
                        generator,
                        &results,
                        &durations,
                        None,
                    )?;
                    if !progress_fn(output) {
                        finish_reason = FinishReason::Cancelled;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if should_stop {
                    while in_flight > 0 {
                        let _ = receiver.recv();
                        in_flight -= 1;
                    }
                    return Ok((results, durations, finish_reason));
                }
            }
        }

        Ok((results, durations, finish_reason))
    }

    fn run_sync_generate<F>(
        model_metadata: &ModelMetadata,
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        generator: &mut Generator,
        sampling_method: SamplingMethod,
        progress: &Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let mut results: Vec<GenerateResult> = Vec::new();
        let mut durations: Vec<f64> = Vec::new();

        loop {
            let start = Instant::now();
            let result = generator.generate(sampling_method)?;
            let new_tokens = result.tokens.clone();
            let duration = start.elapsed().as_secs_f64();

            results.push(result);
            durations.push(duration);

            let finish_reason =
                Self::check_finish_reason(run_context, generator, &new_tokens);
            let output = Self::build_output(
                model_metadata,
                tokenizer,
                output_parser,
                run_context,
                generator,
                &results,
                &durations,
                finish_reason.clone(),
            )?;

            let should_continue = if let Some(progress_fn) = progress {
                progress_fn(output.clone())
            } else {
                true
            };

            if !should_continue || finish_reason.is_some() {
                if should_continue {
                    return Ok(output);
                } else {
                    return Ok(output.clone_with_finish_reason(Some(
                        FinishReason::Cancelled,
                    )));
                }
            }
        }
    }
}

impl Session {
    fn build_stats(
        model_metadata: &ModelMetadata,
        prefill_result: PrefillResult,
        prefill_duration: f64,
        prefill_suffix_length: usize,
        generate_results: Vec<GenerateResult>,
        generate_durations: Vec<f64>,
        generate_suffix_length: usize,
        total_duration: f64,
        tokens_count_input: usize,
        tokens_count_output: usize,
    ) -> Stats {
        let prefill_stats = {
            let tokens_count = prefill_result.tokens.len();
            let tokens_per_second = tokens_count as f64 / prefill_duration;

            // TODO: Unify after removing MPS blocks
            let processed_tokens = if model_metadata.quantization.is_some() {
                let result = tokens_count_input + tokens_count;
                result
            } else {
                let number_of_prefill_steps =
                    (tokens_count_input as f32 / prefill_suffix_length as f32)
                        .ceil() as usize;
                let result = prefill_suffix_length * number_of_prefill_steps
                    + tokens_count;
                result
            };
            let processed_tokens_per_second =
                processed_tokens as f64 / prefill_duration;

            let model_run_count = prefill_result.forwardpass_durations.len();
            let model_run_average_duration =
                prefill_result.forwardpass_durations.iter().sum::<f64>()
                    / model_run_count as f64;

            StepStats {
                duration: prefill_duration,
                suffix_length: prefill_suffix_length as u64,
                tokens_count: tokens_count as u64,
                tokens_per_second,
                processed_tokens_per_second,
                model_run: RunStats {
                    count: model_run_count as u64,
                    average_duration: model_run_average_duration,
                },
                run: None,
            }
        };

        let generate_stats: Option<StepStats>;
        if generate_results.len() != 0 {
            let duration = generate_durations.iter().sum::<f64>();
            let tokens_count = generate_results
                .iter()
                .flat_map(|result| result.tokens.clone())
                .collect::<Vec<u64>>()
                .len();
            let tokens_per_second: f64 = tokens_count as f64 / duration;

            let model_run_count = generate_results.len();
            let model_run_average_duration = generate_results
                .iter()
                .map(|result| result.forwardpass_duration)
                .sum::<f64>()
                / model_run_count as f64;

            let run_count = generate_durations.len();
            let run_average_duration =
                generate_durations.iter().sum::<f64>() / run_count as f64;

            generate_stats = Some(StepStats {
                duration,
                suffix_length: generate_suffix_length as u64,
                tokens_count: tokens_count as u64,
                tokens_per_second,
                processed_tokens_per_second: tokens_per_second,
                model_run: RunStats {
                    count: model_run_count as u64,
                    average_duration: model_run_average_duration,
                },
                run: Some(RunStats {
                    count: run_count as u64,
                    average_duration: run_average_duration,
                }),
            });
        } else {
            generate_stats = None;
        }

        let total_stats = TotalStats {
            duration: total_duration,
            tokens_count_input: tokens_count_input as u64,
            tokens_count_output: tokens_count_output as u64,
        };

        let stats = Stats {
            prefill_stats,
            generate_stats,
            total_stats,
        };

        stats
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        autoreleasepool(|_| {
            self.generator = None;
        });
    }
}
