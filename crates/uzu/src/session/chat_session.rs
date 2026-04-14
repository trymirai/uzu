use std::{
    any::Any,
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

use tokenizers::Tokenizer;
use xgrammar::TokenizerInfo;

use crate::{
    KvDebugSnapshot, TargetHiddenSnapshot, TraceDebugSnapshot,
    backends::{common::Backend, select_backend},
    config::{MixerConfig, ModelMetadata},
    language_model::{
        LanguageModelGenerator, LanguageModelGeneratorTrait,
        grammar::CompiledGrammar,
        result::{GenerateResult, PrefillResult},
    },
    session::{
        config::{DecodingConfig, RunConfig},
        helpers::{InputProcessor, InputProcessorDefault, OutputParser, is_directory_fits_ram},
        parameter::{ConfigResolvableValue, ContextMode},
        types::{Error, FinishReason, Input, Output, RunStats, Stats, StepStats, TotalStats},
    },
    utils::env_utils::EnvVar,
};

fn nanos_to_secs(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

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

pub struct ChatSession {
    pub model_path: PathBuf,
    pub model_metadata: ModelMetadata,

    tokenizer: Tokenizer,
    tokenizer_info: TokenizerInfo,
    input_processor: Box<dyn InputProcessor>,
    output_parser: OutputParser,
    decoding_config: DecodingConfig,
    llm: Option<Box<dyn LanguageModelGeneratorTrait>>,
    static_context: Option<Box<dyn Any>>,
}

impl ChatSession {
    pub fn new(
        model_path: PathBuf,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        select_backend!(Self::new_with_backend::<B>(model_path, decoding_config), Error::UnableToOpenAnyBackend)
    }

    pub fn new_with_backend<B: Backend>(
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
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata = serde_json::from_reader(BufReader::new(config_file)).map_err(|err| {
            eprintln!("Failed to parse config.json: {err}");
            Error::UnableToLoadConfig
        })?;

        let (has_non_attention_mixer, has_mamba_mixer) = model_metadata
            .model_config
            .as_language_model()
            .and_then(|lm| lm.decoder_config().ok())
            .and_then(|dc| dc.layer_configs)
            .map(|layers| {
                let has_non_attention =
                    layers.iter().any(|layer| !matches!(layer.mixer_config, MixerConfig::Attention(_)));
                let has_mamba = layers.iter().any(|layer| matches!(layer.mixer_config, MixerConfig::Mamba(_)));
                (has_non_attention, has_mamba)
            })
            .unwrap_or((false, false));
        if has_non_attention_mixer {
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
        }

        if has_mamba_mixer && decoding_config.speculator_config.number_of_speculated_tokens > 0 {
            return Err(Error::UnsupportedSpeculatorConfigForModel);
        }

        let language_model_config = model_metadata.model_config.as_language_model().ok_or(Error::UnableToLoadConfig)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| Error::UnableToLoadTokenizer)?;

        let stop_token_ids: Vec<i32> =
            language_model_config.generation_config.stop_token_ids.iter().map(|&x| x as i32).collect();
        let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, Some(&stop_token_ids))
            .map_err(|error_message| Error::GrammarError(error_message))?;

        let input_processor = InputProcessorDefault::new(language_model_config.message_processor_config.clone());

        let output_parser =
            OutputParser::new(language_model_config.message_processor_config.output_parser_regex.clone())?;

        let llm = Box::new(
            LanguageModelGenerator::<B>::new(&model_path, decoding_config.clone(), &model_metadata)
                .map_err(Error::from)?,
        );

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            tokenizer_info,
            input_processor: Box::new(input_processor),
            output_parser,
            decoding_config,
            llm: Some(llm),
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
        self.run_with_forced_token_path(input, config, None, progress)
    }

    pub fn run_forced_token_path<F>(
        &mut self,
        input: Input,
        config: RunConfig,
        forced_token_path: &[u64],
        progress: Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        assert!(!forced_token_path.is_empty(), "forced token path must not be empty");
        self.run_with_forced_token_path(input, config, Some(forced_token_path), progress)
    }

    pub fn run_forced_token_path_once(
        &mut self,
        input: Input,
        config: RunConfig,
        forced_token_path: &[u64],
    ) -> Result<Output, Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "single forced generate only supports ContextMode::None",
        );
        assert!(!forced_token_path.is_empty(), "forced token path must not be empty");

        let language_model_config =
            self.model_metadata.model_config.as_language_model().ok_or(Error::UnableToLoadConfig)?;
        let text = self.input_processor.process(&input, config.enable_thinking, config.tokens_limit > 0)?;
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let context_length = self.decoding_config.context_length.resolve(language_model_config);
        if tokens.len() >= context_length {
            return Err(Error::ContextLengthExceeded);
        }

        let prefix_offset = language_model_generator.tokens_len();
        let prefix_len_before = prefix_offset;
        let eos_tokens: Vec<u64> =
            language_model_config.generation_config.stop_token_ids.iter().map(|&x| x as u64).collect();
        let sampling_method = config.sampling_policy.resolve(language_model_config);
        let mut compiled_grammar: Option<CompiledGrammar> = if let Some(ref grammar_config) = config.grammar_config {
            Some(CompiledGrammar::from_config(grammar_config, None, &self.tokenizer_info)?)
        } else {
            None
        };

        let prefill_start = Instant::now();
        let prefill_result = language_model_generator.prefill(
            tokens.clone(),
            compiled_grammar.as_mut(),
            sampling_method,
            prefix_offset,
            false,
        )?;
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        language_model_generator.clear_cache();

        let run_context = RunContext {
            eos_tokens,
            context_length,
            prefix_len_before,
            input_tokens_len: tokens.len(),
            tokens_limit: config.tokens_limit as usize,
            prefill_result: prefill_result.clone(),
            prefill_duration,
            prefill_suffix_length: self.decoding_config.prefill_step_size.resolve(language_model_config),
            run_start: Instant::now(),
        };

        let generate_start = Instant::now();
        let generate_result = language_model_generator.generate_from_token_path(
            forced_token_path,
            compiled_grammar.as_mut(),
            sampling_method,
        )?;
        let generate_duration = generate_start.elapsed().as_secs_f64();
        let generate_tokens = generate_result.tokens.clone();
        let grammar_terminated = compiled_grammar.as_ref().map(|g| g.is_terminated()).unwrap_or(false);
        let finish_reason = if grammar_terminated {
            Some(FinishReason::Stop)
        } else {
            Self::check_finish_reason(&run_context, language_model_generator.as_ref(), &generate_tokens)
        };
        let output = Self::build_output(
            &self.tokenizer,
            &self.output_parser,
            &run_context,
            language_model_generator.as_ref(),
            &[generate_result],
            &[generate_duration],
            finish_reason,
        )?;

        language_model_generator.reset_state();
        Ok(output)
    }

    fn run_with_forced_token_path<F>(
        &mut self,
        input: Input,
        config: RunConfig,
        forced_token_path: Option<&[u64]>,
        progress: Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        match &self.decoding_config.context_mode {
            ContextMode::None => {
                let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
                language_model_generator.reset_state();
            },
            ContextMode::Static {
                input,
            } => {
                if self.static_context.is_none() {
                    let mut prefill_config = config.clone();
                    prefill_config.tokens_limit = 0;
                    let (_out, ctx) = self.extend(input.clone(), None, prefill_config)?;
                    self.static_context = Some(ctx);
                }
                let tmp = self.static_context.take();
                if let Some(ref ctx) = tmp {
                    self.reconfigure_language_model_generator(Some(ctx))?;
                }
                self.static_context = tmp;
            },
            ContextMode::Dynamic => {},
        }

        let output = self.run_internal(input, config, progress, forced_token_path)?;

        match &self.decoding_config.context_mode {
            ContextMode::None
            | ContextMode::Static {
                ..
            } => {
                let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
                language_model_generator.reset_state();
            },
            ContextMode::Dynamic => {},
        }
        Ok(output)
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();
        Ok(())
    }

    pub fn run_capture_kv_debug(
        &mut self,
        input: Input,
        config: RunConfig,
    ) -> Result<(Output, KvDebugSnapshot), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "KV debug capture only supports ContextMode::None",
        );

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let output = self.run_internal(input, config, None::<fn(Output) -> bool>, None)?;
        let snapshot = self.kv_debug_snapshot()?;

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, snapshot))
    }

    pub fn run_capture_prefill_kv_debug(
        &mut self,
        input: Input,
        config: RunConfig,
    ) -> Result<(Output, KvDebugSnapshot), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "KV debug capture only supports ContextMode::None",
        );

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let output = self.run_internal(input, config, Some(|_: Output| false), None)?;
        let snapshot = self.kv_debug_snapshot()?;

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, snapshot))
    }

    pub fn run_capture_first_generate_kv_debug(
        &mut self,
        input: Input,
        config: RunConfig,
    ) -> Result<(Output, KvDebugSnapshot), Error> {
        self.run_capture_generated_kv_debug(input, config, 1)
    }

    pub fn run_capture_generated_kv_debug(
        &mut self,
        input: Input,
        config: RunConfig,
        generated_tokens: usize,
    ) -> Result<(Output, KvDebugSnapshot), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "KV debug capture only supports ContextMode::None",
        );
        assert!(generated_tokens > 0, "generated KV debug capture requires at least one generated token");

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let generated_so_far = std::cell::Cell::new(0usize);
        let output = self.run_internal(
            input,
            config,
            Some(|_: Output| {
                let generated = generated_so_far.get() + 1;
                generated_so_far.set(generated);
                generated < generated_tokens
            }),
            None,
        )?;
        let snapshot = self.kv_debug_snapshot()?;

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, snapshot))
    }

    pub fn run_capture_generated_trace_debug(
        &mut self,
        input: Input,
        config: RunConfig,
        generated_tokens: usize,
    ) -> Result<(Output, TraceDebugSnapshot), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "trace debug capture only supports ContextMode::None",
        );
        assert!(generated_tokens > 0, "generated trace debug capture requires at least one generated token");

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let generated_so_far = std::cell::Cell::new(0usize);
        let output = self.run_internal(
            input,
            config,
            Some(|_: Output| {
                let generated = generated_so_far.get() + 1;
                generated_so_far.set(generated);
                generated < generated_tokens
            }),
            None,
        )?;
        let snapshot = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?.trace_debug_snapshot();

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, snapshot))
    }

    #[cfg(feature = "tracing")]
    pub fn run_capture_target_hidden(
        &mut self,
        input: Input,
        config: RunConfig,
        sample_count: usize,
    ) -> Result<(Output, TargetHiddenSnapshot), Error> {
        let (output, snapshot) = self.run_capture_generated_trace_debug(input, config, 1)?;
        Ok((output, TargetHiddenSnapshot::from_trace_snapshot(&snapshot, sample_count)))
    }

    #[cfg(feature = "tracing")]
    pub fn run_capture_first_generate_step(
        &mut self,
        input: Input,
        config: RunConfig,
        sample_count: usize,
    ) -> Result<(Output, TargetHiddenSnapshot, Box<[u64]>), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "first-step capture only supports ContextMode::None",
        );

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let output = self.run_internal(input, config, Some(|_: Output| false), None)?;
        let trace_snapshot = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?.trace_debug_snapshot();

        let generated_token_count = output.stats.total_stats.tokens_count_output as usize;
        let language_model_generator = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        let tokens = language_model_generator.tokens();
        assert!(generated_token_count <= tokens.len(), "generated token count must fit inside generator tokens");
        let generated_token_ids = tokens[tokens.len() - generated_token_count..].to_vec().into_boxed_slice();

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, TargetHiddenSnapshot::from_trace_snapshot(&trace_snapshot, sample_count), generated_token_ids))
    }

    pub fn run_capture_generated_token_ids(
        &mut self,
        input: Input,
        config: RunConfig,
    ) -> Result<(Output, Box<[u64]>), Error> {
        assert!(
            matches!(self.decoding_config.context_mode, ContextMode::None),
            "token capture only supports ContextMode::None",
        );

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        let output = self.run_internal(input, config, None::<fn(Output) -> bool>, None)?;
        let generated_token_count = output.stats.total_stats.tokens_count_output as usize;
        let language_model_generator = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        let tokens = language_model_generator.tokens();
        assert!(generated_token_count <= tokens.len(), "generated token count must fit inside generator tokens");
        let generated_token_ids = tokens[tokens.len() - generated_token_count..].to_vec().into_boxed_slice();

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();

        Ok((output, generated_token_ids))
    }

    fn extend(
        &mut self,
        input: Input,
        context: Option<&Box<dyn Any>>,
        config: RunConfig,
    ) -> Result<(Output, Box<dyn Any>), Error> {
        self.reconfigure_language_model_generator(context)?;
        let output = self.run_internal(input, config, None::<fn(Output) -> bool>, None)?;
        let new_context = self.build_context_from_language_model_generator()?;
        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();
        Ok((output, new_context))
    }

    fn run_internal<F>(
        &mut self,
        input: Input,
        config: RunConfig,
        progress: Option<F>,
        forced_token_path: Option<&[u64]>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let run_start = Instant::now();
        let text = self.input_processor.process(&input, config.enable_thinking, config.tokens_limit > 0)?;
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();

        let language_model_config =
            self.model_metadata.model_config.as_language_model().ok_or(Error::UnableToLoadConfig)?;

        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        let context_length = self.decoding_config.context_length.resolve(language_model_config);
        if tokens.len() >= context_length {
            return Err(Error::ContextLengthExceeded);
        }

        let prefix_offset = language_model_generator.tokens_len();
        let prefix_len_before = prefix_offset;

        let eos_tokens: Vec<u64> =
            language_model_config.generation_config.stop_token_ids.iter().map(|&x| x as u64).collect();

        let sampling_method = config.sampling_policy.resolve(language_model_config);

        let mut compiled_grammar: Option<CompiledGrammar> = if let Some(ref grammar_config) = config.grammar_config {
            Some(CompiledGrammar::from_config(grammar_config, None, &self.tokenizer_info)?)
        } else {
            None
        };

        let prefill_start = Instant::now();
        let sample_suffix = forced_token_path.is_none() && config.tokens_limit > 0;
        let prefill_result = language_model_generator.prefill(
            tokens.clone(),
            compiled_grammar.as_mut(),
            sampling_method,
            prefix_offset,
            sample_suffix,
        )?;
        let prefill_tokens = prefill_result.tokens.clone();
        let prefill_duration = prefill_start.elapsed().as_secs_f64();
        language_model_generator.clear_cache();

        let prefill_suffix_length = self.decoding_config.prefill_step_size.resolve(language_model_config);

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

        let grammar_terminated = compiled_grammar.as_ref().map(|g| g.is_terminated()).unwrap_or(false);

        let prefill_finish_reason = if grammar_terminated {
            Some(FinishReason::Stop)
        } else {
            Self::check_finish_reason(&run_context, language_model_generator.as_ref(), &prefill_tokens)
        };
        let prefill_output = Self::build_output(
            &self.tokenizer,
            &self.output_parser,
            &run_context,
            language_model_generator.as_ref(),
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
                return Ok(prefill_output.clone_with_finish_reason(Some(FinishReason::Cancelled)));
            }
        }

        let can_use_async = forced_token_path.is_none()
            && language_model_generator.generate_suffix_length() == 1
            && compiled_grammar.is_none()
            && !language_model_generator.uses_materialized_transformer_state()
            && !EnvVar::DisableAsyncGeneration.is_enabled();

        let generate_output = if let Some(forced_token_path) = forced_token_path {
            Self::run_sync_generate_forced_token_path(
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                language_model_generator.as_mut(),
                compiled_grammar.as_mut(),
                sampling_method,
                forced_token_path,
                &progress,
            )?
        } else if can_use_async {
            let batch_size = language_model_generator.async_batch_size(&self.model_path);
            Self::run_async_batch(
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                language_model_generator.as_mut(),
                sampling_method,
                &progress,
                batch_size,
            )?
        } else {
            Self::run_sync_generate(
                &self.tokenizer,
                &self.output_parser,
                &run_context,
                language_model_generator.as_mut(),
                compiled_grammar.as_mut(),
                sampling_method,
                &progress,
            )?
        };

        language_model_generator.clear_cache();
        Ok(generate_output.clone_with_duration(run_start.elapsed().as_secs_f64()))
    }

    fn run_sync_generate<F>(
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        language_model_generator: &mut dyn LanguageModelGeneratorTrait,
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: super::parameter::SamplingMethod,
        progress: &Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let mut generate_results: Vec<GenerateResult> = Vec::new();
        let mut generate_durations: Vec<f64> = Vec::new();
        let mut compiled_grammar_mut = compiled_grammar;

        loop {
            let generate_start = Instant::now();
            let generate_result =
                language_model_generator.generate(compiled_grammar_mut.as_deref_mut(), sampling_method)?;
            let generate_tokens = generate_result.tokens.clone();
            let generate_duration = generate_start.elapsed().as_secs_f64();
            generate_results.push(generate_result);
            generate_durations.push(generate_duration);

            let grammar_terminated = compiled_grammar_mut.as_ref().map(|g| g.is_terminated()).unwrap_or(false);

            let generate_finish_reason = if grammar_terminated {
                Some(FinishReason::Stop)
            } else {
                Self::check_finish_reason(run_context, language_model_generator, &generate_tokens)
            };
            let generate_output = Self::build_output(
                tokenizer,
                output_parser,
                run_context,
                language_model_generator,
                &generate_results,
                &generate_durations,
                generate_finish_reason.clone(),
            )?;

            let generate_should_continue = if let Some(progress_fn) = progress {
                progress_fn(generate_output.clone())
            } else {
                true
            };

            if !generate_should_continue || generate_finish_reason.is_some() {
                if generate_should_continue {
                    return Ok(generate_output);
                } else {
                    return Ok(generate_output.clone_with_finish_reason(Some(FinishReason::Cancelled)));
                }
            }
        }
    }

    fn run_sync_generate_forced_token_path<F>(
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        language_model_generator: &mut dyn LanguageModelGeneratorTrait,
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: super::parameter::SamplingMethod,
        forced_token_path: &[u64],
        progress: &Option<F>,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let mut generate_results: Vec<GenerateResult> = Vec::new();
        let mut generate_durations: Vec<f64> = Vec::new();
        let mut compiled_grammar_mut = compiled_grammar;
        let mut path_offset = 0;

        loop {
            let remaining_path = &forced_token_path[path_offset..];
            if remaining_path.is_empty() {
                return Ok(Self::build_output(
                    tokenizer,
                    output_parser,
                    run_context,
                    language_model_generator,
                    &generate_results,
                    &generate_durations,
                    None,
                )?);
            }

            let generate_start = Instant::now();
            let generate_result = language_model_generator.generate_from_token_path(
                remaining_path,
                compiled_grammar_mut.as_deref_mut(),
                sampling_method,
            )?;
            let generate_tokens = generate_result.tokens.clone();
            let generate_duration = generate_start.elapsed().as_secs_f64();
            path_offset += generate_tokens.len();
            generate_results.push(generate_result);
            generate_durations.push(generate_duration);

            let grammar_terminated = compiled_grammar_mut.as_ref().map(|g| g.is_terminated()).unwrap_or(false);

            let generate_finish_reason = if grammar_terminated {
                Some(FinishReason::Stop)
            } else {
                Self::check_finish_reason(run_context, language_model_generator, &generate_tokens)
            };
            let generate_output = Self::build_output(
                tokenizer,
                output_parser,
                run_context,
                language_model_generator,
                &generate_results,
                &generate_durations,
                generate_finish_reason.clone(),
            )?;

            let generate_should_continue = if let Some(progress_fn) = progress {
                progress_fn(generate_output.clone())
            } else {
                true
            };

            if !generate_should_continue || generate_finish_reason.is_some() {
                if generate_should_continue {
                    return Ok(generate_output);
                } else {
                    return Ok(generate_output.clone_with_finish_reason(Some(FinishReason::Cancelled)));
                }
            }
        }
    }

    fn run_async_batch<F>(
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        llm: &mut dyn LanguageModelGeneratorTrait,
        sampling_method: super::parameter::SamplingMethod,
        progress: &Option<F>,
        batch_size: usize,
    ) -> Result<Output, Error>
    where
        F: Fn(Output) -> bool,
    {
        let remaining_by_limit = run_context.tokens_limit.saturating_sub(run_context.prefill_result.tokens.len());
        let remaining_by_context = run_context.context_length.saturating_sub(llm.tokens_len());
        let tokens_to_generate = remaining_by_limit.min(remaining_by_context);

        llm.prepare_async(tokens_to_generate);

        let (sender, receiver) = mpsc::channel::<(usize, u64, u64)>();
        let start_time = Instant::now();
        let last_nanos = Arc::new(AtomicU64::new(0));

        let mut results: Vec<GenerateResult> = Vec::with_capacity(tokens_to_generate);
        let mut durations: Vec<f64> = Vec::with_capacity(tokens_to_generate);
        let mut finish_reason = FinishReason::Length;
        let mut next_to_submit = 0;
        let mut in_flight = 0;

        while in_flight > 0 || next_to_submit < tokens_to_generate {
            let batch_end = std::cmp::min(next_to_submit + batch_size, tokens_to_generate);
            let batch_submitted = batch_end - next_to_submit;
            let accepted_before_batch = results.len();

            let cache_slice = llm.get_slice(0..batch_submitted);

            for idx in next_to_submit..batch_end {
                let batch_sender = sender.clone();
                let last_callback_nanos = last_nanos.clone();
                let batch_start_time = start_time;
                llm.async_generate(
                    idx,
                    sampling_method,
                    Box::new(move |token| {
                        let now_nanos = batch_start_time.elapsed().as_nanos() as u64;
                        let prev_nanos = last_callback_nanos.swap(now_nanos, Ordering::SeqCst);
                        let _ = batch_sender.send((idx, token, now_nanos.saturating_sub(prev_nanos)));
                    }),
                )?;
                in_flight += 1;
            }
            next_to_submit = batch_end;

            for _ in 0..batch_submitted {
                let (_, token, duration_nanos) = receiver.recv().map_err(|_| Error::SamplingFailed)?;
                in_flight -= 1;

                let duration = nanos_to_secs(duration_nanos);
                llm.tokens_push(token);
                results.push(GenerateResult {
                    tokens: vec![token],
                    forwardpass_duration: duration,
                    speculator_proposed: 0,
                    speculator_accepted: 0,
                });
                durations.push(duration);

                let should_stop = if run_context.eos_tokens.contains(&token) {
                    finish_reason = FinishReason::Stop;
                    true
                } else if llm.tokens_len() >= run_context.context_length {
                    finish_reason = FinishReason::ContextLimitReached;
                    true
                } else if let Some(progress_fn) = progress {
                    let output =
                        Self::build_output(tokenizer, output_parser, run_context, llm, &results, &durations, None)?;
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
                    if let Some(slice) = cache_slice {
                        let accepted_in_batch = results.len().saturating_sub(accepted_before_batch);
                        if accepted_in_batch < batch_submitted {
                            llm.apply_slice(slice.as_ref(), accepted_in_batch..batch_submitted);
                        }
                    }
                    return Self::build_output(
                        tokenizer,
                        output_parser,
                        run_context,
                        llm,
                        &results,
                        &durations,
                        Some(finish_reason),
                    );
                }
            }
        }

        Self::build_output(tokenizer, output_parser, run_context, llm, &results, &durations, Some(finish_reason))
    }

    fn reconfigure_language_model_generator(
        &mut self,
        context: Option<&Box<dyn Any>>,
    ) -> Result<(), Error> {
        let language_model_generator = self.llm.as_mut().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        language_model_generator.reset_state();
        if let Some(ctx) = context {
            language_model_generator.reconfigure_from_context(ctx.as_ref());
        }
        Ok(())
    }

    fn build_context_from_language_model_generator(&self) -> Result<Box<dyn Any>, Error> {
        let language_model_generator = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        Ok(language_model_generator.build_llm_context())
    }

    fn check_finish_reason(
        run_context: &RunContext,
        language_model_generator: &dyn LanguageModelGeneratorTrait,
        new_tokens: &[u64],
    ) -> Option<FinishReason> {
        let start_idx = run_context.prefix_len_before + run_context.input_tokens_len;
        let total_new_tokens = language_model_generator.tokens_len().saturating_sub(start_idx);
        let has_eos = new_tokens.iter().any(|t| run_context.eos_tokens.contains(t));
        let context_limit = language_model_generator.tokens_len() >= run_context.context_length;

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

    fn decode_generated_tokens(
        tokenizer: &Tokenizer,
        prefill_tokens: &[u64],
        generate_results: &[GenerateResult],
    ) -> Result<String, Error> {
        let generated_tokens: Vec<u32> = prefill_tokens
            .iter()
            .chain(generate_results.iter().flat_map(|r| r.tokens.iter()))
            .map(|&v| v as u32)
            .collect();
        tokenizer.decode(&generated_tokens, true).map_err(|_| Error::UnableToDecodeText)
    }

    fn build_output(
        tokenizer: &Tokenizer,
        output_parser: &OutputParser,
        run_context: &RunContext,
        language_model_generator: &dyn LanguageModelGeneratorTrait,
        generate_results: &[GenerateResult],
        generate_durations: &[f64],
        finish_reason: Option<FinishReason>,
    ) -> Result<Output, Error> {
        let text = Self::decode_generated_tokens(tokenizer, &run_context.prefill_result.tokens, generate_results)?;
        let parsed = output_parser.parse(text);
        let start_idx = run_context.prefix_len_before + run_context.input_tokens_len;
        let output_tokens = language_model_generator.tokens_len().saturating_sub(start_idx);

        Ok(Output {
            text: parsed,
            stats: Self::build_stats(
                run_context.prefill_result.clone(),
                run_context.prefill_duration,
                run_context.prefill_suffix_length,
                generate_results.to_vec(),
                generate_durations.to_vec(),
                language_model_generator.generate_suffix_length(),
                run_context.run_start.elapsed().as_secs_f64(),
                run_context.input_tokens_len,
                output_tokens,
            ),
            finish_reason,
        })
    }

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

            let processed_tokens = tokens_count_input + tokens_count;
            let processed_tokens_per_second = processed_tokens as f64 / prefill_duration;

            let model_run_count = prefill_result.forwardpass_durations.len();
            let model_run_average_duration =
                prefill_result.forwardpass_durations.iter().sum::<f64>() / model_run_count as f64;

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
                speculator_proposed: 0,
                speculator_accepted: 0,
            }
        };

        let generate_stats: Option<StepStats>;
        if generate_results.len() != 0 {
            let duration = generate_durations.iter().sum::<f64>();
            let tokens_count: usize = generate_results.iter().map(|result| result.tokens.len()).sum();
            let tokens_per_second: f64 = tokens_count as f64 / duration;

            let model_run_count = generate_results.len();
            let model_run_average_duration =
                generate_results.iter().map(|result| result.forwardpass_duration).sum::<f64>() / model_run_count as f64;

            let run_count = generate_durations.len();
            let run_average_duration = generate_durations.iter().sum::<f64>() / run_count as f64;

            let speculator_proposed: usize = generate_results.iter().map(|result| result.speculator_proposed).sum();
            let speculator_accepted: usize = generate_results.iter().map(|result| result.speculator_accepted).sum();

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
                speculator_proposed: speculator_proposed as u64,
                speculator_accepted: speculator_accepted as u64,
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

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.llm.as_ref().unwrap().peak_memory_usage()
    }

    pub fn kv_storage_bytes(&self) -> u64 {
        self.llm.as_ref().expect("language model generator must exist").kv_storage_bytes() as u64
    }

    pub fn kv_debug_snapshot(&self) -> Result<KvDebugSnapshot, Error> {
        let language_model_generator = self.llm.as_ref().ok_or(Error::LanguageModelGeneratorNotLoaded)?;
        Ok(language_model_generator.kv_debug_snapshot())
    }
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        self.llm = None
    }
}
