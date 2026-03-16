use super::*;

pub(super) type PreInjectionEncodeCallback<'a, B> = dyn FnMut(
        &TokenDecoderRunner<B>,
        &ForwardPassState<B>,
        &mut <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), Error>
    + 'a;

pub(super) enum EmbeddingInjection {
    None,
    AddPreloaded {
        post_scale: Option<f32>,
    },
    OverrideFirstRowInternal,
}

pub(super) struct TextSamplingState {
    rng: StdRng,
    method: SamplingMethod,
}

impl TextSamplingState {
    pub(super) fn from_config(
        seed: u64,
        config: &TextSamplingConfig,
    ) -> Self {
        Self::with_params(seed, config.temperature, config.top_p)
    }

    pub(super) fn with_params(
        seed: u64,
        temperature: f32,
        top_p: f32,
    ) -> Self {
        let method = if temperature <= 0.0 || top_p <= 0.0 {
            SamplingMethod::Greedy
        } else {
            SamplingMethod::AdvancedStochastic {
                temperature: Some(temperature),
                top_k: None,
                top_p: Some(top_p),
                min_p: None,
                processing_order: SamplingProcessingOrder::FiltersThenTemperature,
            }
        };
        Self {
            rng: StdRng::seed_from_u64(seed),
            method,
        }
    }

    pub(super) fn method(&self) -> SamplingMethod {
        self.method
    }

    pub(super) fn next_seed(&mut self) -> u64 {
        self.rng.random::<u64>()
    }
}

pub(super) fn text_decoder_prefill_step_size(
    config: &TextDecoderRuntimeConfig,
    context_length: usize,
) -> usize {
    config.prefill_step_size.min(context_length.max(1)).max(1)
}

pub(super) fn normalize_rendered_prompt(
    mut rendered: String,
    template: &str,
    drop_initial_newline: bool,
) -> String {
    if drop_initial_newline && rendered.starts_with('\n') {
        rendered.remove(0);
    }
    if template.ends_with('\n') && rendered.ends_with('\n') {
        rendered.pop();
    }
    rendered
}

impl AdaptiveChunkController {
    pub(super) fn new(config: &TtsRunConfig) -> Self {
        Self {
            ema_ms_per_frame: None,
            current_chunk_frames: config.min_chunk_frames.max(1),
        }
    }

    pub(super) fn target_frames(
        &self,
        config: &TtsRunConfig,
    ) -> usize {
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        match config.chunk_policy {
            TtsChunkPolicy::Fixed => min_frames,
            TtsChunkPolicy::Adaptive => {
                let Some(ema_ms_per_frame) = self.ema_ms_per_frame else {
                    return min_frames;
                };
                let raw = (config.target_emit_latency_ms as f64 / ema_ms_per_frame).round();
                let candidate = raw.max(min_frames as f64).min(max_frames as f64) as usize;
                if self.current_chunk_frames == 0 {
                    return candidate;
                }
                if candidate <= self.current_chunk_frames {
                    return self.current_chunk_frames;
                }
                let change = ((candidate as f64 - self.current_chunk_frames as f64).abs()
                    / self.current_chunk_frames as f64)
                    .max(0.0);
                if change < DEFAULT_CHUNK_HYSTERESIS_FRACTION {
                    self.current_chunk_frames
                } else {
                    candidate
                }
            },
        }
    }

    pub(super) fn observe(
        &mut self,
        frames: usize,
        decode_elapsed: std::time::Duration,
        next_chunk_frames: usize,
    ) {
        if frames == 0 {
            return;
        }
        let ms_per_frame = (decode_elapsed.as_secs_f64() * 1000.0) / frames as f64;
        self.ema_ms_per_frame = Some(match self.ema_ms_per_frame {
            Some(previous) => previous * (1.0 - DEFAULT_CHUNK_EMA_ALPHA) + ms_per_frame * DEFAULT_CHUNK_EMA_ALPHA,
            None => ms_per_frame,
        });
        self.current_chunk_frames = next_chunk_frames.max(1);
    }

    pub(super) fn adapt_up_for_realtime(
        &mut self,
        config: &TtsRunConfig,
        generated_frames: usize,
        sample_rate: u32,
        decode_elapsed: std::time::Duration,
        emitted_audio_frames: usize,
    ) {
        if generated_frames == 0 || emitted_audio_frames == 0 || sample_rate == 0 {
            return;
        }
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let produced_audio_ms = (emitted_audio_frames as f64) * 1000.0 / f64::from(sample_rate);
        if produced_audio_ms <= 0.0 {
            return;
        }
        let realtime_ratio = decode_ms / produced_audio_ms;
        if realtime_ratio <= 1.0 {
            return;
        }
        let scaled = ((generated_frames as f64) * realtime_ratio * 1.1).ceil() as usize;
        let clamped = scaled.clamp(min_frames, max_frames);
        self.current_chunk_frames = self.current_chunk_frames.max(clamped);
    }

    pub(super) fn promote_to_max_chunk(
        &mut self,
        config: &TtsRunConfig,
    ) {
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        self.current_chunk_frames = max_frames;
    }
}

pub(super) fn next_startup_target_frames(
    current_target_frames: usize,
    startup_cap_frames: usize,
) -> usize {
    let startup_cap_frames = startup_cap_frames.max(1);
    current_target_frames.max(1).saturating_mul(2).min(startup_cap_frames)
}

pub(super) fn audio_decode_streaming_mode(config: &TtsRunConfig) -> AudioDecodeStreamingMode {
    match config.vocoder_streaming_mode {
        TtsVocoderStreamingMode::IncrementalStateful => AudioDecodeStreamingMode::IncrementalStateful,
        TtsVocoderStreamingMode::PrefixFallback => AudioDecodeStreamingMode::PrefixFallback,
    }
}

pub(super) struct StreamingTokenAccumulator {
    by_codebook: Vec<Vec<u32>>,
}

impl StreamingTokenAccumulator {
    pub(super) fn new(num_codebooks: usize) -> Result<Self, Error> {
        if num_codebooks == 0 {
            return Err(Error::UnableToLoadConfig);
        }
        Ok(Self {
            by_codebook: vec![Vec::new(); num_codebooks],
        })
    }

    pub(super) fn push_frame(
        &mut self,
        frame_codes: &[u32],
    ) -> Result<(), Error> {
        if frame_codes.len() != self.by_codebook.len() {
            return Err(Error::GenerateFailed);
        }
        for (codebook, &token) in self.by_codebook.iter_mut().zip(frame_codes.iter()) {
            codebook.push(token);
        }
        Ok(())
    }

    pub(super) fn frames(&self) -> usize {
        self.by_codebook.first().map_or(0, Vec::len)
    }

    pub(super) fn to_grid_range(
        &self,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<AudioTokenGrid, Error> {
        let frames = self.frames();
        if frame_start > frame_end || frame_end > frames {
            return Err(Error::GenerateFailed);
        }
        let range_frames = frame_end - frame_start;
        let mut tokens = Vec::with_capacity(self.by_codebook.len() * range_frames);
        for codebook in &self.by_codebook {
            if codebook.len() != frames {
                return Err(Error::GenerateFailed);
            }
            tokens.extend_from_slice(&codebook[frame_start..frame_end]);
        }

        AudioTokenGrid::new(
            tokens.into_boxed_slice(),
            1,
            self.by_codebook.len(),
            range_frames,
            vec![range_frames].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .map_err(Error::from)
    }
}

pub(super) fn slice_codebook_major_grid_range(
    grid: &AudioTokenGrid,
    frame_start: usize,
    frame_end: usize,
) -> Result<AudioTokenGrid, Error> {
    if frame_start > frame_end || frame_end > grid.frames() {
        return Err(Error::GenerateFailed);
    }
    let range_frames = frame_end.saturating_sub(frame_start);
    let packed = grid.to_packing(AudioTokenPacking::CodebookMajor);
    let batch_size = packed.batch_size();
    let codebooks = packed.codebooks();
    let frames = packed.frames();
    if frames != grid.frames() {
        return Err(Error::GenerateFailed);
    }

    let mut tokens = Vec::with_capacity(batch_size * codebooks * range_frames);
    let row_stride = frames;
    for batch in 0..batch_size {
        for codebook in 0..codebooks {
            let row_index = batch
                .checked_mul(codebooks)
                .and_then(|value| value.checked_add(codebook))
                .ok_or(Error::GenerateFailed)?;
            let row_start = row_index.checked_mul(row_stride).ok_or(Error::GenerateFailed)?;
            let src_start = row_start.checked_add(frame_start).ok_or(Error::GenerateFailed)?;
            let src_end = row_start.checked_add(frame_end).ok_or(Error::GenerateFailed)?;
            tokens.extend_from_slice(&packed.tokens()[src_start..src_end]);
        }
    }

    let mut lengths = Vec::with_capacity(batch_size);
    for &length in packed.lengths() {
        lengths.push(length.saturating_sub(frame_start).min(range_frames));
    }

    AudioTokenGrid::new(
        tokens.into_boxed_slice(),
        batch_size,
        codebooks,
        range_frames,
        lengths.into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .map_err(Error::from)
}

pub(super) fn accumulate_audio_decode_step_stats(
    decode_calls: &mut usize,
    input_frames: &mut usize,
    decoded_window_frames: &mut usize,
    max_decoded_window_frames: &mut usize,
    step: Option<AudioDecodeStepStats>,
) {
    let Some(step) = step else {
        return;
    };
    if step.input_frames == 0 && step.decoded_window_frames == 0 {
        return;
    }
    *decode_calls = decode_calls.saturating_add(1);
    *input_frames = input_frames.saturating_add(step.input_frames);
    *decoded_window_frames = decoded_window_frames.saturating_add(step.decoded_window_frames);
    *max_decoded_window_frames = (*max_decoded_window_frames).max(step.decoded_window_frames);
}

pub(super) struct StreamingSynthesisState<'a, F: FnMut(&AudioPcmBatch)> {
    pub(super) on_chunk: &'a mut F,
    pub(super) pending_chunk: Option<PendingStreamingChunk>,
    pub(super) callback_seconds: f64,
    pub(super) audio_decode_seconds_in_loop: f64,
    pub(super) audio_decode_seconds: f64,
    pub(super) output_samples: Vec<f32>,
    pub(super) output_frames: usize,
    pub(super) output_sample_rate: u32,
    pub(super) output_channels: usize,
    pub(super) emitted_chunks: usize,
    pub(super) first_emit_pending: bool,
    pub(super) first_chunk_seconds: Option<f64>,
    pub(super) first_chunk_frames: usize,
    pub(super) stream_start: Instant,
    pub(super) startup_target_frames: usize,
    pub(super) startup_cap_frames: usize,
    pub(super) chunk_controller: AdaptiveChunkController,
    pub(super) audio_decode_calls: usize,
    pub(super) audio_input_frames: usize,
    pub(super) audio_decoded_window_frames: usize,
    pub(super) audio_max_decoded_window_frames: usize,
}

impl<'a, F: FnMut(&AudioPcmBatch)> StreamingSynthesisState<'a, F> {
    pub(super) fn maybe_flush_pending(
        &mut self,
        force: bool,
        count_in_loop: bool,
        config: &TtsRunConfig,
    ) -> Result<(), Error> {
        let should_flush =
            self.pending_chunk.as_ref().map(|pending| force || pending.chunk.is_ready()).unwrap_or(false);
        if !should_flush {
            return Ok(());
        }

        let pending = self.pending_chunk.take().ok_or(Error::GenerateFailed)?;
        accumulate_audio_decode_step_stats(
            &mut self.audio_decode_calls,
            &mut self.audio_input_frames,
            &mut self.audio_decoded_window_frames,
            &mut self.audio_max_decoded_window_frames,
            pending.chunk.step_stats(),
        );
        let (partial_pcm, resolve_decode_duration) = pending.chunk.resolve_with_decode_duration()?;
        let decode_elapsed = pending.submission_decode_duration.saturating_add(resolve_decode_duration);
        if count_in_loop {
            self.audio_decode_seconds_in_loop += decode_elapsed.as_secs_f64();
        }
        self.audio_decode_seconds += decode_elapsed.as_secs_f64();

        let partial_sample_rate = partial_pcm.sample_rate();
        let callback_start = Instant::now();
        let emitted_frames = if partial_pcm.lengths().len() == 1 {
            partial_pcm.lengths()[0]
        } else {
            return Err(Error::GenerateFailed);
        };
        if emitted_frames > 0 {
            (self.on_chunk)(&partial_pcm);
        }
        self.callback_seconds += callback_start.elapsed().as_secs_f64();
        self.chunk_controller.observe(pending.ready_frames, decode_elapsed, pending.next_chunk_frames);

        if emitted_frames > 0 {
            self.chunk_controller.adapt_up_for_realtime(
                config,
                pending.ready_frames,
                partial_sample_rate,
                decode_elapsed,
                emitted_frames,
            );
            self.output_samples.extend_from_slice(partial_pcm.samples());
            self.output_frames = self.output_frames.saturating_add(emitted_frames);
            self.output_sample_rate = partial_pcm.sample_rate();
            self.output_channels = partial_pcm.channels();
            self.emitted_chunks = self.emitted_chunks.saturating_add(1);
            if self.first_emit_pending {
                self.first_emit_pending = false;
                self.first_chunk_seconds = Some(self.stream_start.elapsed().as_secs_f64());
                self.first_chunk_frames = emitted_frames;
                self.chunk_controller.promote_to_max_chunk(config);
            }
        } else if self.first_emit_pending {
            self.startup_target_frames =
                next_startup_target_frames(self.startup_target_frames, self.startup_cap_frames);
        }

        Ok(())
    }
}

impl SemanticDecoderBackend for StubTextDecoderRuntime {
    fn default_seed(&self) -> u64 {
        self.default_seed
    }

    fn generate_semantic_tokens(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        generate_stub_semantic_grid(self, text_tokens, codec_cardinality, seed, max_semantic_frames)
    }

    fn generate_semantic_tokens_with_callback(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<(), Error>,
    ) -> Result<AudioTokenGrid, Error> {
        let grid = generate_stub_semantic_grid(self, text_tokens, codec_cardinality, seed, max_semantic_frames)?;
        for frame in 0..grid.frames() {
            let mut frame_codes = Vec::with_capacity(self.num_codebooks);
            for codebook in 0..self.num_codebooks {
                frame_codes.push(grid.get(0, codebook, frame));
            }
            on_frame(&frame_codes)?;
        }
        Ok(grid)
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        RunnerInstrumentation::default()
    }
}

impl<B: Backend + Send + Sync> TtsSession<B> {
    pub fn new(model_path: PathBuf) -> Result<Self, Error> {
        Self::new_with_options(model_path, TtsSessionOptions::default())
    }

    pub fn new_with_options(
        model_path: PathBuf,
        options: TtsSessionOptions,
    ) -> Result<Self, Error> {
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(std::io::BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        Self::from_model_metadata_with_options(model_path, model_metadata, options)
    }

    pub fn last_execution_stats(&self) -> Option<TtsExecutionStats> {
        self.last_execution_stats.borrow().clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    fn from_model_metadata_with_options(
        model_path: PathBuf,
        model_metadata: ModelMetadata,
        options: TtsSessionOptions,
    ) -> Result<Self, Error> {
        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| Error::UnableToLoadTokenizer)?;

        let loaded_runtime = load_tts_runtime(&model_path, &model_metadata, &options)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            audio: loaded_runtime.audio,
            audio_decoder: loaded_runtime.audio_decoder,
            message_processor_config: loaded_runtime.message_processor_config,
            text_decoder: RefCell::new(loaded_runtime.text_decoder),
            last_execution_stats: RefCell::new(None),
        })
    }

    pub(super) fn render_prompt(
        &self,
        input: &Input,
    ) -> Result<String, Error> {
        let messages = input
            .get_messages()
            .into_iter()
            .map(|message| message.resolve(&self.message_processor_config))
            .collect::<Vec<_>>();

        let template_name = "tts_prompt_template";
        let mut environment = Environment::new();
        environment
            .add_template(template_name, self.message_processor_config.prompt_template.as_str())
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;
        let template = environment.get_template(template_name).map_err(|_| Error::UnableToLoadPromptTemplate)?;

        let result = template
            .render(context!(
                messages => messages
            ))
            .map_err(|_| Error::UnableToRenderPromptTemplate)?;

        Ok(normalize_rendered_prompt(
            result,
            self.message_processor_config.prompt_template.as_str(),
            self.message_processor_config.drop_initial_newline,
        ))
    }
}

pub(super) fn semantic_token_to_code(
    semantic_token: u64,
    semantic_begin: i64,
    semantic_end: i64,
    token_upper_bound: usize,
) -> u32 {
    if semantic_begin > semantic_end || token_upper_bound == 0 {
        return 0;
    }

    let semantic = semantic_token as i64;
    if semantic < semantic_begin || semantic > semantic_end {
        return 0;
    }

    let relative = usize::try_from(semantic - semantic_begin).unwrap_or(0);
    let clamped = relative.min(token_upper_bound.saturating_sub(1));
    u32::try_from(clamped).unwrap_or(0)
}

pub(super) fn build_semantic_sampling_mask_row(
    vocab_size: usize,
    semantic_begin: i64,
    semantic_end: i64,
    im_end: i64,
) -> Result<Box<[u32]>, Error> {
    if vocab_size == 0 || semantic_begin > semantic_end {
        return Err(Error::UnableToLoadConfig);
    }

    let max_token_id = i64::try_from(vocab_size.saturating_sub(1)).map_err(|_| Error::UnableToLoadConfig)?;
    if semantic_begin < 0 || semantic_end < 0 || semantic_end > max_token_id || im_end < 0 || im_end > max_token_id {
        return Err(Error::UnableToLoadConfig);
    }

    let row_words = vocab_size.div_ceil(32);
    let mut mask = vec![0_u32; row_words];
    for token_index in semantic_begin..=semantic_end {
        let token = usize::try_from(token_index).map_err(|_| Error::UnableToLoadConfig)?;
        let word = token / 32;
        let bit = token % 32;
        mask[word] |= 1_u32 << bit;
    }
    let im_end_token = usize::try_from(im_end).map_err(|_| Error::UnableToLoadConfig)?;
    let word = im_end_token / 32;
    let bit = im_end_token % 32;
    mask[word] |= 1_u32 << bit;
    Ok(mask.into_boxed_slice())
}

pub(super) fn clear_token_in_sampling_mask(
    mask: &mut [u32],
    token: i64,
) -> Result<(), Error> {
    if token < 0 {
        return Err(Error::UnableToLoadConfig);
    }
    let token = usize::try_from(token).map_err(|_| Error::UnableToLoadConfig)?;
    let word = token / 32;
    let bit = token % 32;
    if word >= mask.len() {
        return Err(Error::UnableToLoadConfig);
    }
    mask[word] &= !(1_u32 << bit);
    Ok(())
}

pub(super) fn expand_token_mask_for_sampling_row(
    row_mask: &[u32],
    token_count: usize,
) -> Result<Box<[u32]>, Error> {
    if token_count == 0 || row_mask.is_empty() {
        return Err(Error::GenerateFailed);
    }
    if token_count == 1 {
        return Ok(row_mask.to_vec().into_boxed_slice());
    }

    let row_words = row_mask.len();
    let total_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
    let mut expanded = vec![u32::MAX; total_words];
    let offset = (token_count - 1).checked_mul(row_words).ok_or(Error::GenerateFailed)?;
    expanded[offset..offset + row_words].copy_from_slice(row_mask);
    Ok(expanded.into_boxed_slice())
}
