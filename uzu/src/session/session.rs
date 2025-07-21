use std::{
    path::PathBuf,
    rc::Rc,
    time::Instant,
};

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
    generator: Generator,
}

impl Session {
    pub fn new(model_path: PathBuf) -> Result<Self, SessionError> {
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

        let generator_config = SessionConfig::default().generator_config(&tokenizer);
        let generator = Generator::new(&model_path, generator_config)
            .map_err(SessionError::from)?;

        Ok(Self {
            model_path,
            tokenizer,
            tokenizer_config,
            input_processor: Box::new(input_processor),
            generator,
        })
    }

    pub fn load_with_session_config(
        &mut self,
        config: SessionConfig,
    ) -> Result<(), SessionError> {
        let generator_config = config.generator_config(&self.tokenizer);
        self.generator = Generator::new(&self.model_path, generator_config)
            .map_err(SessionError::from)?;
        Ok(())
    }

    pub fn extend(
        &mut self,
        input: SessionInput,
        context: Option<Rc<SessionContext>>,
        config: SessionRunConfig,
    ) -> (SessionOutput, Rc<SessionContext>) {
        self.reconfigure_generator(context);
        let output = self.run_internal(input, config);
        let new_context = self.build_context_from_generator();
        self.generator.reset_state();
        (output, new_context)
    }

    pub fn run(
        &mut self,
        input: SessionInput,
        config: SessionRunConfig,
        _callback: Option<impl Fn(SessionOutput) -> bool>,
    ) -> SessionOutput {
        // Old API - no context support, direct run
        self.generator.reset_state();
        let output = self.run_internal(input, config);
        self.generator.reset_state();
        output
    }

    pub fn run_with_context(
        &mut self,
        input: SessionInput,
        context: Option<Rc<SessionContext>>,
        config: SessionRunConfig,
    ) -> SessionOutput {
        self.reconfigure_generator(context);
        let output = self.run_internal(input, config);
        self.generator.reset_state();
        output
    }

    fn run_internal(
        &mut self,
        input: SessionInput,
        config: SessionRunConfig,
    ) -> SessionOutput {
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


        let prefix_len_before = self.generator.prefix_len();

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
        let prefix_offset = self.generator.tokens.len();

        let prefill_result =
            self.generator.prefill(tokens.clone(), sampling_config, prefix_offset);
        let prefill_tokens = prefill_result.tokens.clone();
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        self.generator.clear_cache();

        let prefill_finish_reason =
            finish_reason(&self.generator, prefill_tokens.clone());
        let prefill_generated_text =
            build_generated_text(&self.generator, &self.tokenizer);

        let mut output = SessionOutput {
            text: prefill_generated_text,
            stats: Self::build_stats(
                prefill_result.clone(),
                prefill_duration,
                self.generator.config.prefill_step_size,
                Vec::new(),
                Vec::new(),
                self.generator.config.generate_suffix_length(),
                run_start.elapsed().as_secs_f64(),
                tokens.len(),
                self.generator.tokens[prefix_len_before + tokens.len()..].len(),
            ),
            finish_reason: prefill_finish_reason.clone(),
        };

        if prefill_finish_reason.is_some() {
            return output;
        }

        let mut generate_results: Vec<GenerateResult> = Vec::new();
        let mut generate_durations: Vec<f64> = Vec::new();
        loop {
            let generate_start = Instant::now();
            let generate_result = self.generator.generate(sampling_config);
            let generate_tokens = generate_result.tokens.clone();
            let generate_duration = generate_start.elapsed().as_secs_f64();
            generate_results.push(generate_result);
            generate_durations.push(generate_duration);

            let generate_finish_reason =
                finish_reason(&self.generator, generate_tokens);
            let generate_generated_text =
                build_generated_text(&self.generator, &self.tokenizer);
            
            output = SessionOutput {
                text: generate_generated_text,
                stats: Self::build_stats(
                    prefill_result.clone(),
                    prefill_duration,
                    self.generator.config.prefill_step_size,
                    generate_results.clone(),
                    generate_durations.clone(),
                    self.generator.config.generate_suffix_length(),
                    run_start.elapsed().as_secs_f64(),
                    tokens.len(),
                    self.generator.tokens[prefix_len_before + tokens.len()..].len(),
                ),
                finish_reason: generate_finish_reason.clone(),
            };

            if generate_finish_reason.is_some() {
                break;
            }
        }
        self.generator.clear_cache();
        output.clone_with_duration(run_start.elapsed().as_secs_f64())
    }

    fn reconfigure_generator(
        &mut self,
        context: Option<Rc<SessionContext>>,
    ) {
        self.generator.reset_state();
        if let Some(ctx) = context {
            // Copy context data into the generator's already properly-sized KV cache
            let mut generator_cache = self.generator.context.kv_cache.borrow_mut();
            for (ctx_layer, gen_layer) in ctx.kv_cache.data.iter().zip(generator_cache.data.iter_mut()) {
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
                gen_layer.prefix_token_positions = ctx_layer.prefix_token_positions.clone();
            }
            drop(generator_cache);
            
            self.generator.tokens = ctx.tokens.clone();
        }
    }

    fn build_context_from_generator(&self) -> Rc<SessionContext> {
        let prefix_len = self.generator.prefix_len();
        let kv_cache = self.generator.context.kv_cache.borrow().clone_and_slice(
            &self.generator.context.mtl_context,
            prefix_len,
        );
        let context = SessionContext::new(
            self.generator.tokens.clone(),
            kv_cache,
            self.generator.config.clone(),
        );
        Rc::new(context)
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
            // empty
        });
    }
}
