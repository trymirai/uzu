use std::{fs::File, io::BufReader, path::PathBuf, time::Instant};

use objc2::rc::autoreleasepool;
use tokenizers::Tokenizer;

use crate::{
    config::ModelMetadata,
    generator::{
        generator::Generator,
        result::{GenerateResult, PrefillResult},
    },
    session::{
        config::{DecodingConfig, RunConfig},
        helpers::{
            Context, InputProcessor, InputProcessorDefault,
            is_directory_fits_ram,
        },
        parameter::ConfigResolvableValue,
        types::{
            Error, FinishReason, Input, Output, RunStats, Stats, StepStats,
            TotalStats,
        },
    },
};

pub struct Session {
    pub model_path: PathBuf,
    pub model_metadata: ModelMetadata,

    tokenizer: Tokenizer,
    input_processor: Box<dyn InputProcessor>,
    generator: Option<Generator>,
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
        let model_metadata: ModelMetadata =
            serde_json::from_reader(BufReader::new(config_file))
                .map_err(|_| Error::UnableToLoadConfig)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|_| Error::UnableToLoadTokenizer)?;

        let input_processor = InputProcessorDefault::new(
            model_metadata.model_config.message_processor_config.clone(),
        );

        let generator = Generator::new(&model_path, decoding_config)
            .map_err(Error::from)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            input_processor: Box::new(input_processor),
            generator: Some(generator),
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
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        generator.reset_state();
        let output = self.run_internal(input, config, progress)?;
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok(output)
    }

    pub fn extend(
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

    pub fn run_with_context(
        &mut self,
        input: Input,
        context: Option<&Context>,
        config: RunConfig,
    ) -> Result<Output, Error> {
        self.reconfigure_generator(context)?;
        let output =
            self.run_internal(input, config, None::<fn(Output) -> bool>)?;
        let generator =
            self.generator.as_mut().ok_or(Error::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok(output)
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

        let prefix_len_before = generator.prefix_len();

        let eos_tokens: Vec<u64> = self
            .model_metadata
            .model_config
            .generation_config
            .stop_token_ids
            .iter()
            .map(|&x| x as u64)
            .collect();

        let finish_reason = |generator: &Generator,
                             tokens_new: Vec<u64>|
         -> Option<FinishReason> {
            let start_idx = prefix_len_before + tokens.len();
            let total_tokens = generator.tokens.len();
            let total_new_tokens = generator.tokens[start_idx..].len();
            let has_eos_token =
                tokens_new.iter().any(|token| eos_tokens.contains(token));
            let context_limit_reached = total_tokens >= context_length;

            if has_eos_token {
                Some(FinishReason::Stop)
            } else if total_new_tokens >= config.tokens_limit as usize {
                Some(FinishReason::Length)
            } else if context_limit_reached {
                Some(FinishReason::ContextLimitReached)
            } else {
                None
            }
        };

        let build_generated_text = |generator: &Generator,
                                    tokenizer: &Tokenizer|
         -> Result<String, Error> {
            let start_idx = prefix_len_before + tokens.len();
            let generated_tokens: Vec<u32> = generator.tokens[start_idx..]
                .to_vec()
                .iter()
                .map(|value| *value as u32)
                .collect();
            let generated_text = tokenizer
                .decode(&generated_tokens, true)
                .map_err(|_| Error::UnableToDecodeText)?;
            Ok(generated_text)
        };

        let sampling_method =
            config.sampling_policy.resolve(&self.model_metadata.model_config);

        let prefill_start = Instant::now();
        let prefix_offset = generator.tokens.len();

        let prefill_result =
            generator.prefill(tokens.clone(), sampling_method, prefix_offset);
        let prefill_tokens = prefill_result.tokens.clone();
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        generator.clear_cache();

        let prefill_finish_reason =
            finish_reason(generator, prefill_tokens.clone());
        let prefill_generated_text =
            build_generated_text(generator, &self.tokenizer)?;

        let prefill_suffix_length = generator
            .decoding_config
            .prefill_step_size
            .resolve(&self.model_metadata.model_config);
        let prefill_output = Output {
            text: prefill_generated_text,
            stats: Self::build_stats(
                prefill_result.clone(),
                prefill_duration,
                prefill_suffix_length,
                Vec::new(),
                Vec::new(),
                generator.decoding_config.generate_suffix_length(),
                run_start.elapsed().as_secs_f64(),
                tokens.len(),
                generator.tokens[prefix_len_before + tokens.len()..].len(),
            ),
            finish_reason: prefill_finish_reason.clone(),
        };

        let prefill_should_continue = if let Some(progress) = &progress {
            progress(prefill_output.clone())
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

        let mut generate_results: Vec<GenerateResult> = Vec::new();
        let mut generate_durations: Vec<f64> = Vec::new();
        let generate_output = loop {
            let generate_start = Instant::now();
            let generate_result = generator.generate(sampling_method);
            let generate_tokens = generate_result.tokens.clone();
            let generate_duration = generate_start.elapsed().as_secs_f64();
            generate_results.push(generate_result);
            generate_durations.push(generate_duration);

            let generate_finish_reason =
                finish_reason(generator, generate_tokens);
            let generate_generated_text =
                build_generated_text(generator, &self.tokenizer)?;

            let generate_output = Output {
                text: generate_generated_text,
                stats: Self::build_stats(
                    prefill_result.clone(),
                    prefill_duration,
                    prefill_suffix_length,
                    generate_results.clone(),
                    generate_durations.clone(),
                    generator.decoding_config.generate_suffix_length(),
                    run_start.elapsed().as_secs_f64(),
                    tokens.len(),
                    generator.tokens[prefix_len_before + tokens.len()..].len(),
                ),
                finish_reason: generate_finish_reason.clone(),
            };

            let generate_should_continue = if let Some(progress) = &progress {
                progress(generate_output.clone())
            } else {
                true
            };

            if !generate_should_continue || generate_finish_reason.is_some() {
                if generate_should_continue {
                    break generate_output;
                } else {
                    break generate_output.clone_with_finish_reason(Some(
                        FinishReason::Cancelled,
                    ));
                }
            }
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
        if let Some(ctx) = context {
            let mut generator_cache = generator.context.kv_cache.borrow_mut();
            for (ctx_layer, gen_layer) in
                ctx.kv_cache.data.iter().zip(generator_cache.data.iter_mut())
            {
                let copy_rows = ctx_layer.effective_prefix_length();
                if copy_rows > 0 {
                    gen_layer.keys.borrow_mut().copy_slice(
                        &ctx_layer.keys.borrow(),
                        1,
                        0..copy_rows,
                        0,
                    );
                    gen_layer.values.borrow_mut().copy_slice(
                        &ctx_layer.values.borrow(),
                        1,
                        0..copy_rows,
                        0,
                    );
                }
                gen_layer.state = ctx_layer.state.clone();
                gen_layer.prefix_token_positions =
                    ctx_layer.prefix_token_positions.clone();
            }
            drop(generator_cache);

            generator.tokens = ctx.tokens.clone();
        }
        Ok(())
    }

    fn build_context_from_generator(&self) -> Result<Context, Error> {
        let generator =
            self.generator.as_ref().ok_or(Error::GeneratorNotLoaded)?;
        let prefix_len = generator.prefix_len();
        let kv_cache = generator
            .context
            .kv_cache
            .borrow()
            .clone_with_prefix_len(&generator.context.mtl_context, prefix_len);
        let context = Context::new(
            generator.tokens.clone(),
            kv_cache,
            generator.decoding_config.clone(),
        );
        Ok(context)
    }
}

impl Session {
    fn build_stats(
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

            let model_run_count = prefill_result.forwardpass_durations.len();
            let model_run_average_duration =
                prefill_result.forwardpass_durations.iter().sum::<f64>()
                    / model_run_count as f64;

            StepStats {
                duration: prefill_duration,
                suffix_length: prefill_suffix_length as u64,
                tokens_count: tokens_count as u64,
                tokens_per_second,
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
