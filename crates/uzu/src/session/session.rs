#[cfg(target_os = "ios")]
use std::path::Path;
use std::{path::PathBuf, time::Instant};

use objc2::rc::autoreleasepool;
use tokenizers::Tokenizer;

use super::{
    session_config::SessionConfig,
    session_context::SessionContext,
    session_input::{
        SessionInput, SessionInputProcessor, SessionInputProcessorDefault,
    },
    session_output::{
        SessionOutput, SessionOutputFinishReason, SessionOutputRunStats,
        SessionOutputStats, SessionOutputStepStats, SessionOutputTotalStats,
    },
    session_run_config::SessionRunConfig,
};
use crate::{
    generator::{
        config::GeneratorConfigProvider,
        generator::Generator,
        result::{GenerateResult, PrefillResult},
    },
    session::{
        session_error::SessionError,
        session_tokenizer_config::SessionTokenizerConfig,
    },
};

pub struct Session {
    model_path: PathBuf,
    tokenizer: Tokenizer,
    tokenizer_config: SessionTokenizerConfig,
    input_processor: Box<dyn SessionInputProcessor>,
    generator: Option<Generator>,
}

impl Session {
    #[cfg(target_os = "ios")]
    fn directory_size(path: &Path) -> std::io::Result<u64> {
        let mut size = 0u64;
        for entry_result in std::fs::read_dir(path)? {
            let entry = entry_result?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                size += Self::directory_size(&entry.path())?;
            } else {
                size += metadata.len();
            }
        }
        Ok(size)
    }

    #[cfg(target_os = "ios")]
    fn assert_model_fits_ram(model_path: &Path) -> Result<(), SessionError> {
        use sysinfo::System;

        let model_size_bytes = Self::directory_size(model_path).unwrap_or(0);

        let mut sys = System::new();
        sys.refresh_memory();

        let allowed_bytes = sys.total_memory() * 60 / 100;

        if model_size_bytes > allowed_bytes {
            Err(SessionError::UnsupportedModel)
        } else {
            Ok(())
        }
    }

    pub fn new(model_path: PathBuf) -> Result<Self, SessionError> {
        #[cfg(target_os = "ios")]
        Self::assert_model_fits_ram(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|_| SessionError::UnableToLoadTokenizer)?;

        let tokenizer_config =
            SessionTokenizerConfig::load_and_add_special_tokens_to_tokenizer(
                model_path.clone(),
                &mut tokenizer,
            )
            .ok_or(SessionError::UnableToLoadTokenizerConfig)?;

        let input_processor =
            SessionInputProcessorDefault::new(tokenizer_config.clone());

        Ok(Self {
            model_path,
            tokenizer,
            tokenizer_config,
            input_processor: Box::new(input_processor),
            generator: None,
        })
    }

    pub fn load(
        &mut self,
        generator_config_provider: Box<dyn GeneratorConfigProvider>,
    ) -> Result<(), SessionError> {
        let generator_config =
            generator_config_provider.generator_config(&self.tokenizer);
        let generator = Generator::new(&self.model_path, generator_config)
            .map_err(SessionError::from)?;

        self.generator = Some(generator);
        Ok(())
    }

    pub fn load_with_session_config(
        &mut self,
        config: SessionConfig,
    ) -> Result<(), SessionError> {
        self.load(Box::new(config))
    }

    pub fn extend(
        &mut self,
        input: SessionInput,
        context: Option<&SessionContext>,
        config: SessionRunConfig,
    ) -> Result<(SessionOutput, SessionContext), SessionError> {
        self.reconfigure_generator(context)?;
        let output = self.run_internal(
            input,
            config,
            None::<fn(SessionOutput) -> bool>,
        )?;
        let new_context = self.build_context_from_generator()?;
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok((output, new_context))
    }

    pub fn run<F>(
        &mut self,
        input: SessionInput,
        config: SessionRunConfig,
        progress: Option<F>,
    ) -> Result<SessionOutput, SessionError>
    where
        F: Fn(SessionOutput) -> bool,
    {
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;
        generator.reset_state();
        let output = self.run_internal(input, config, progress)?;
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok(output)
    }

    pub fn run_with_context(
        &mut self,
        input: SessionInput,
        context: Option<&SessionContext>,
        config: SessionRunConfig,
    ) -> Result<SessionOutput, SessionError> {
        self.reconfigure_generator(context)?;
        let output = self.run_internal(
            input,
            config,
            None::<fn(SessionOutput) -> bool>,
        )?;
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;
        generator.reset_state();
        Ok(output)
    }

    fn run_internal<F>(
        &mut self,
        input: SessionInput,
        config: SessionRunConfig,
        progress: Option<F>,
    ) -> Result<SessionOutput, SessionError>
    where
        F: Fn(SessionOutput) -> bool,
    {
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;

        let run_start = Instant::now();
        let text = self.input_processor.process(&input);
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();

        let prefix_len_before = generator.prefix_len();

        let eos_tokens = self
            .tokenizer_config
            .eos_tokens
            .iter()
            .map(|token| {
                self.tokenizer.token_to_id(token.as_str()).unwrap() as u64
            })
            .collect::<Vec<_>>();

        let finish_reason = |generator: &Generator,
                             tokens_new: Vec<u64>|
         -> Option<SessionOutputFinishReason> {
            let start_idx = prefix_len_before + tokens.len();
            let total_new_tokens = generator.tokens[start_idx..].len();
            let has_eos_token =
                tokens_new.iter().any(|token| eos_tokens.contains(token));

            if has_eos_token {
                Some(SessionOutputFinishReason::Stop)
            } else if total_new_tokens >= config.tokens_limit as usize {
                Some(SessionOutputFinishReason::Length)
            } else {
                None
            }
        };

        let build_generated_text =
            |generator: &Generator, tokenizer: &Tokenizer| -> String {
                let start_idx = prefix_len_before + tokens.len();
                let generated_tokens: Vec<u32> = generator.tokens[start_idx..]
                    .to_vec()
                    .iter()
                    .map(|value| *value as u32)
                    .collect();
                tokenizer.decode(&generated_tokens, true).unwrap()
            };

        let sampling_config = config
            .sampling_config
            .unwrap_or(self.tokenizer_config.sampling_config);

        let prefill_start = Instant::now();
        let prefix_offset = generator.tokens.len();

        let prefill_result =
            generator.prefill(tokens.clone(), sampling_config, prefix_offset);
        let prefill_tokens = prefill_result.tokens.clone();
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        generator.clear_cache();

        let prefill_finish_reason =
            finish_reason(generator, prefill_tokens.clone());
        let prefill_generated_text =
            build_generated_text(generator, &self.tokenizer);

        let prefill_output = SessionOutput {
            text: prefill_generated_text,
            stats: Self::build_stats(
                prefill_result.clone(),
                prefill_duration,
                generator.config.prefill_step_size,
                Vec::new(),
                Vec::new(),
                generator.config.generate_suffix_length(),
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
                return Ok(prefill_output.clone_with_finish_reason(Some(
                    SessionOutputFinishReason::Cancelled,
                )));
            }
        }

        let mut generate_results: Vec<GenerateResult> = Vec::new();
        let mut generate_durations: Vec<f64> = Vec::new();
        let generate_output = loop {
            let generate_start = Instant::now();
            let generate_result = generator.generate(sampling_config);
            let generate_tokens = generate_result.tokens.clone();
            let generate_duration = generate_start.elapsed().as_secs_f64();
            generate_results.push(generate_result);
            generate_durations.push(generate_duration);

            let generate_finish_reason =
                finish_reason(generator, generate_tokens);
            let generate_generated_text =
                build_generated_text(generator, &self.tokenizer);

            let generate_output = SessionOutput {
                text: generate_generated_text,
                stats: Self::build_stats(
                    prefill_result.clone(),
                    prefill_duration,
                    generator.config.prefill_step_size,
                    generate_results.clone(),
                    generate_durations.clone(),
                    generator.config.generate_suffix_length(),
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
                        SessionOutputFinishReason::Cancelled,
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
        context: Option<&SessionContext>,
    ) -> Result<(), SessionError> {
        let generator =
            self.generator.as_mut().ok_or(SessionError::GeneratorNotLoaded)?;
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

    fn build_context_from_generator(
        &self
    ) -> Result<SessionContext, SessionError> {
        let generator =
            self.generator.as_ref().ok_or(SessionError::GeneratorNotLoaded)?;
        let prefix_len = generator.prefix_len();
        let kv_cache = generator
            .context
            .kv_cache
            .borrow()
            .clone_with_prefix_len(&generator.context.mtl_context, prefix_len);
        let context = SessionContext::new(
            generator.tokens.clone(),
            kv_cache,
            generator.config.clone(),
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
    ) -> SessionOutputStats {
        let prefill_stats = {
            let tokens_count = prefill_result.tokens.len();
            let tokens_per_second = tokens_count as f64 / prefill_duration;

            let model_run_count = prefill_result.forwardpass_durations.len();
            let model_run_average_duration =
                prefill_result.forwardpass_durations.iter().sum::<f64>()
                    / model_run_count as f64;

            SessionOutputStepStats {
                duration: prefill_duration,
                suffix_length: prefill_suffix_length as u64,
                tokens_count: tokens_count as u64,
                tokens_per_second,
                model_run: SessionOutputRunStats {
                    count: model_run_count as u64,
                    average_duration: model_run_average_duration,
                },
                run: None,
            }
        };

        let generate_stats: Option<SessionOutputStepStats>;
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

            generate_stats = Some(SessionOutputStepStats {
                duration,
                suffix_length: generate_suffix_length as u64,
                tokens_count: tokens_count as u64,
                tokens_per_second,
                model_run: SessionOutputRunStats {
                    count: model_run_count as u64,
                    average_duration: model_run_average_duration,
                },
                run: Some(SessionOutputRunStats {
                    count: run_count as u64,
                    average_duration: run_average_duration,
                }),
            });
        } else {
            generate_stats = None;
        }

        let total_stats = SessionOutputTotalStats {
            duration: total_duration,
            tokens_count_input: tokens_count_input as u64,
            tokens_count_output: tokens_count_output as u64,
        };

        let stats = SessionOutputStats {
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
