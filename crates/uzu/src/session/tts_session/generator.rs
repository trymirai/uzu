use super::*;

impl<B: Backend + Send + Sync> TtsSession<B> {
    pub fn synthesize(
        &self,
        input: Input,
    ) -> Result<AudioPcmBatch, Error> {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_with_seed(input, seed)
    }

    pub fn synthesize_with_seed(
        &self,
        input: Input,
        seed: u64,
    ) -> Result<AudioPcmBatch, Error> {
        self.synthesize_with_seed_and_config(input, seed, &TtsRunConfig::default())
    }

    pub fn synthesize_with_config(
        &self,
        input: Input,
        config: &TtsRunConfig,
    ) -> Result<AudioPcmBatch, Error> {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_with_seed_and_config(input, seed, config)
    }

    pub fn synthesize_with_seed_and_config(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
    ) -> Result<AudioPcmBatch, Error> {
        config.validate().map_err(|reason| Error::InvalidTtsRunConfig(reason.to_string()))?;

        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        let semantic_start = Instant::now();
        let semantic_tokens = self.generate_semantic_tokens(&text_tokens, seed, config.max_semantic_frames)?;
        let semantic_decode_seconds = semantic_start.elapsed().as_secs_f64();
        let instrumentation = self.take_text_decoder_instrumentation();

        let audio_start = Instant::now();
        let mut audio_decode_calls = 0usize;
        let mut audio_input_frames = 0usize;
        let mut audio_decoded_window_frames = 0usize;
        let mut audio_max_decoded_window_frames = 0usize;
        let pcm = match config.non_streaming_mode {
            crate::session::config::TtsNonStreamingMode::FullDecode => {
                audio_decode_calls = usize::from(semantic_tokens.frames() > 0);
                audio_input_frames = semantic_tokens.frames();
                audio_decoded_window_frames = semantic_tokens.frames();
                audio_max_decoded_window_frames = semantic_tokens.frames();
                self.audio_decoder.decode(&semantic_tokens)?
            },
            crate::session::config::TtsNonStreamingMode::ChunkedIfNeeded => {
                config.validate_stream_decode().map_err(|reason| Error::InvalidTtsRunConfig(reason.to_string()))?;
                let total_frames = semantic_tokens.frames();
                let chunked_threshold = config.max_stream_workspace_frames.max(config.max_chunk_frames.max(1));
                if total_frames < chunked_threshold {
                    audio_decode_calls = usize::from(total_frames > 0);
                    audio_input_frames = total_frames;
                    audio_decoded_window_frames = total_frames;
                    audio_max_decoded_window_frames = total_frames;
                    self.audio_decoder.decode(&semantic_tokens)?
                } else {
                    let chunk_frames = config.max_chunk_frames.max(config.min_chunk_frames.max(1));
                    let mut stream = self.audio_decoder.begin_stream(
                        semantic_tokens.batch_size(),
                        semantic_tokens.codebooks(),
                        audio_decode_streaming_mode(config),
                        config.max_stream_workspace_frames,
                    )?;

                    let mut all_samples = Vec::<f32>::new();
                    let mut accumulated_lengths = vec![0usize; semantic_tokens.batch_size()];
                    let mut sample_rate = self.audio_decoder.sample_rate();
                    let mut channels = 1usize;

                    let mut frame_start = 0usize;
                    while frame_start < total_frames {
                        let frame_end = (frame_start + chunk_frames).min(total_frames);
                        let delta_grid = slice_codebook_major_grid_range(&semantic_tokens, frame_start, frame_end)?;
                        let partial_pcm = stream.decode_step(&delta_grid, frame_end == total_frames)?;
                        accumulate_audio_decode_step_stats(
                            &mut audio_decode_calls,
                            &mut audio_input_frames,
                            &mut audio_decoded_window_frames,
                            &mut audio_max_decoded_window_frames,
                            stream.last_step_stats(),
                        );
                        if partial_pcm.lengths().len() != accumulated_lengths.len() {
                            return Err(Error::GenerateFailed);
                        }
                        for (acc, &len) in accumulated_lengths.iter_mut().zip(partial_pcm.lengths().iter()) {
                            *acc = acc.saturating_add(len);
                        }
                        if !partial_pcm.samples().is_empty() {
                            all_samples.extend_from_slice(partial_pcm.samples());
                        }
                        sample_rate = partial_pcm.sample_rate();
                        channels = partial_pcm.channels();
                        frame_start = frame_end;
                    }
                    stream.finish()?;
                    AudioPcmBatch::new(
                        all_samples.into_boxed_slice(),
                        sample_rate,
                        channels,
                        accumulated_lengths.into_boxed_slice(),
                    )
                    .map_err(Error::from)?
                }
            },
        };
        let audio_decode_seconds = audio_start.elapsed().as_secs_f64();

        self.record_last_execution_stats(TtsExecutionStats {
            semantic_decode_seconds,
            audio_decode_seconds,
            callback_seconds: 0.0,
            first_chunk_seconds: 0.0,
            command_buffers_submitted: instrumentation.command_buffers_submitted,
            host_waits: instrumentation.host_waits,
            semantic_frames: semantic_tokens.frames(),
            first_chunk_frames: 0,
            emitted_chunks: usize::from(config.streaming_enabled),
            audio_decode_calls,
            audio_input_frames,
            audio_decoded_window_frames,
            audio_max_decoded_window_frames,
        });

        Ok(pcm)
    }

    pub fn generate_semantic_tokens_with_seed(
        &self,
        input: Input,
        seed: u64,
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_with_seed_and_config(input, seed, &TtsRunConfig::default())
    }

    pub fn generate_semantic_tokens_with_seed_and_config(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
    ) -> Result<AudioTokenGrid, Error> {
        config.validate().map_err(|reason| Error::InvalidTtsRunConfig(reason.to_string()))?;
        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        self.generate_semantic_tokens(&text_tokens, seed, config.max_semantic_frames)
    }

    pub fn synthesize_streaming<F>(
        &self,
        input: Input,
        chunk_frames: usize,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_streaming_with_seed(input, seed, chunk_frames, on_chunk)
    }

    pub fn synthesize_streaming_with_seed<F>(
        &self,
        input: Input,
        seed: u64,
        chunk_frames: usize,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let config = TtsRunConfig::fixed_chunk_frames(chunk_frames);
        self.synthesize_streaming_with_seed_and_config(input, seed, &config, on_chunk)
    }

    pub fn synthesize_streaming_with_config<F>(
        &self,
        input: Input,
        config: &TtsRunConfig,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_streaming_with_seed_and_config(input, seed, config, on_chunk)
    }

    pub fn synthesize_streaming_with_seed_and_config<F>(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
        mut on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        if !config.streaming_enabled {
            let pcm = self.synthesize_with_seed_and_config(input, seed, config)?;
            on_chunk(&pcm);
            return Ok(pcm);
        }
        config.validate_stream_decode().map_err(|reason| Error::InvalidTtsRunConfig(reason.to_string()))?;

        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        let mut streamed_tokens = StreamingTokenAccumulator::new(self.audio_decoder.num_codebooks())?;
        let mut audio_stream = self.audio_decoder.begin_stream(
            1,
            self.audio_decoder.num_codebooks(),
            audio_decode_streaming_mode(config),
            config.max_stream_workspace_frames,
        )?;
        let mut last_decoded_frames = 0usize;
        let startup_cap_frames = config.max_chunk_frames.max(config.min_chunk_frames.max(1));
        let initial_chunk_frames = config.initial_chunk_frames.max(1).min(startup_cap_frames);

        let mut ss = StreamingSynthesisState {
            on_chunk: &mut on_chunk,
            pending_chunk: None,
            callback_seconds: 0.0,
            audio_decode_seconds_in_loop: 0.0,
            audio_decode_seconds: 0.0,
            output_samples: Vec::new(),
            output_frames: 0,
            output_sample_rate: self.audio_decoder.sample_rate(),
            output_channels: 1,
            emitted_chunks: 0,
            first_emit_pending: true,
            first_chunk_seconds: None,
            first_chunk_frames: 0,
            stream_start: Instant::now(),
            startup_target_frames: initial_chunk_frames,
            startup_cap_frames,
            chunk_controller: AdaptiveChunkController::new(config),
            audio_decode_calls: 0,
            audio_input_frames: 0,
            audio_decoded_window_frames: 0,
            audio_max_decoded_window_frames: 0,
        };

        let semantic_start = Instant::now();
        let semantic_tokens = self.generate_semantic_tokens_with_callback(
            &text_tokens,
            seed,
            config.max_semantic_frames,
            &mut |codes| {
                ss.maybe_flush_pending(false, true, config)?;
                streamed_tokens.push_frame(codes)?;
                let ready_frames = streamed_tokens.frames().saturating_sub(last_decoded_frames);
                let adaptive_target_frames = ss.chunk_controller.target_frames(config);
                let target_frames = if ss.first_emit_pending {
                    ss.startup_target_frames
                } else {
                    adaptive_target_frames
                };
                if ready_frames >= target_frames && ss.pending_chunk.is_none() {
                    let partial_grid = streamed_tokens.to_grid_range(last_decoded_frames, streamed_tokens.frames())?;
                    last_decoded_frames = streamed_tokens.frames();
                    let pending_next_chunk_frames = if ss.first_emit_pending {
                        adaptive_target_frames
                    } else {
                        target_frames
                    };
                    let decode_start = Instant::now();
                    let pending_audio_chunk = audio_stream.decode_step_pending(&partial_grid, false)?;
                    ss.pending_chunk = Some(PendingStreamingChunk {
                        submission_decode_duration: decode_start.elapsed(),
                        ready_frames,
                        next_chunk_frames: pending_next_chunk_frames,
                        chunk: pending_audio_chunk,
                    });
                    ss.maybe_flush_pending(false, true, config)?;
                }
                Ok(())
            },
        )?;
        let semantic_loop_seconds = semantic_start.elapsed().as_secs_f64();

        ss.maybe_flush_pending(true, false, config)?;

        if last_decoded_frames < semantic_tokens.frames() {
            let final_decode_start = Instant::now();
            let final_delta_grid = streamed_tokens.to_grid_range(last_decoded_frames, semantic_tokens.frames())?;
            let final_pcm = audio_stream.decode_step(&final_delta_grid, true)?;
            accumulate_audio_decode_step_stats(
                &mut ss.audio_decode_calls,
                &mut ss.audio_input_frames,
                &mut ss.audio_decoded_window_frames,
                &mut ss.audio_max_decoded_window_frames,
                audio_stream.last_step_stats(),
            );
            ss.audio_decode_seconds += final_decode_start.elapsed().as_secs_f64();
            let callback_start = Instant::now();
            let emitted_frames = if final_pcm.lengths().len() == 1 {
                final_pcm.lengths()[0]
            } else {
                return Err(Error::GenerateFailed);
            };
            if emitted_frames > 0 {
                (ss.on_chunk)(&final_pcm);
            }
            ss.callback_seconds += callback_start.elapsed().as_secs_f64();
            if emitted_frames > 0 {
                ss.output_samples.extend_from_slice(final_pcm.samples());
                ss.output_frames = ss.output_frames.saturating_add(emitted_frames);
                ss.output_sample_rate = final_pcm.sample_rate();
                ss.output_channels = final_pcm.channels();
                ss.emitted_chunks += 1;
                if ss.first_emit_pending {
                    ss.first_chunk_seconds = Some(ss.stream_start.elapsed().as_secs_f64());
                    ss.first_chunk_frames = emitted_frames;
                }
            }
        }
        let semantic_decode_seconds =
            (semantic_loop_seconds - ss.audio_decode_seconds_in_loop - ss.callback_seconds).max(0.0);
        audio_stream.finish()?;

        let stats = TtsExecutionStats {
            semantic_decode_seconds,
            audio_decode_seconds: ss.audio_decode_seconds,
            callback_seconds: ss.callback_seconds,
            first_chunk_seconds: ss.first_chunk_seconds.unwrap_or(0.0),
            command_buffers_submitted: 0,
            host_waits: 0,
            semantic_frames: semantic_tokens.frames(),
            first_chunk_frames: ss.first_chunk_frames,
            emitted_chunks: ss.emitted_chunks,
            audio_decode_calls: ss.audio_decode_calls,
            audio_input_frames: ss.audio_input_frames,
            audio_decoded_window_frames: ss.audio_decoded_window_frames,
            audio_max_decoded_window_frames: ss.audio_max_decoded_window_frames,
        };

        let full_pcm = AudioPcmBatch::new(
            ss.output_samples.into_boxed_slice(),
            ss.output_sample_rate,
            ss.output_channels,
            vec![ss.output_frames].into_boxed_slice(),
        )
        .map_err(Error::from)?;

        let instrumentation = self.take_text_decoder_instrumentation();
        self.record_last_execution_stats(TtsExecutionStats {
            command_buffers_submitted: instrumentation.command_buffers_submitted,
            host_waits: instrumentation.host_waits,
            ..stats
        });

        Ok(full_pcm)
    }

    fn generate_semantic_tokens(
        &self,
        text_tokens: &[u64],
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality = self.audio_decoder.codec_cardinality();
        decoder.generate_semantic_tokens(text_tokens, codec_cardinality, seed, max_semantic_frames)
    }

    fn generate_semantic_tokens_with_callback<F>(
        &self,
        text_tokens: &[u64],
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut F,
    ) -> Result<AudioTokenGrid, Error>
    where
        F: FnMut(&[u32]) -> Result<(), Error>,
    {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality = self.audio_decoder.codec_cardinality();
        decoder.generate_semantic_tokens_with_callback(
            text_tokens,
            codec_cardinality,
            seed,
            max_semantic_frames,
            on_frame,
        )
    }

    fn take_text_decoder_instrumentation(&self) -> RunnerInstrumentation {
        self.text_decoder.borrow_mut().take_instrumentation()
    }

    fn record_last_execution_stats(
        &self,
        stats: TtsExecutionStats,
    ) {
        self.last_execution_stats.borrow_mut().replace(stats);
    }
}
