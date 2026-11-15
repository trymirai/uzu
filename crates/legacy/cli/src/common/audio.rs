use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tokenizers::Tokenizer;
use uzu_engine::audio::lfm3_audio::pipeline::{
    AudioChatTurn, GenerationMode, Lfm3AudioGenerateOptions, Lfm3AudioPipeline, PreparedAudioPrompt,
};
use uzu_engine::audio::lfm3_audio::AudioSamplingConfig;
use uzu_engine::audio::liquid_codec_metal::MetalLiquidCodec;
use uzu_engine::backends::metal::Metal;
use uzu_engine::engine::language_model::LanguageModel;
use uzu_engine::vision::{VlmSampler, VlmSamplingConfig, VlmState};

pub const DEFAULT_ASR_SYSTEM_PROMPT: &str = "Perform ASR in English.";
pub const DEFAULT_TTS_SYSTEM_PROMPT: &str = "Perform TTS in American English using a female voice.";

pub struct AudioInputTurn {
    pub role: String,
    pub text: String,
    pub wav_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Default)]
pub struct AudioGenerateOptions {
    pub audio_temperature: Option<f32>,
    pub audio_top_k: Option<usize>,
    pub audio_repetition_penalty: Option<f32>,
    pub audio_repetition_window: Option<usize>,
    pub assistant_prefill: Option<String>,
}

pub struct AudioOutput {
    pub text: String,
    pub raw_text: String,
    pub reasoning: Option<String>,
    pub tokens: Vec<u64>,
    pub frames: Vec<Vec<i64>>,
    pub wav_bytes: Option<Vec<u8>>,
    pub sample_rate: u32,
    pub prep_ms: f64,
    pub prefill_ms: f64,
    pub depth_ms: f64,
    pub backbone_ms: f64,
    pub codec_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTurnSummary {
    pub role: String,
    pub text: String,
    pub audio_hash: Option<u64>,
}

fn hash_audio_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

pub struct AudioRuntime {
    pub pipeline: Lfm3AudioPipeline,
    pub codec: Option<MetalLiquidCodec>,
    pub state: VlmState,
    session_id: Option<String>,
    cached_turns: Vec<AudioTurnSummary>,
    last_clean_eos: bool,
    live_asr_session_id: Option<String>,
}

impl AudioRuntime {
    pub async fn load(model_dir: &Path, lm: &LanguageModel<Metal>) -> Result<Self> {
        let context = lm.context();
        let pipeline = Lfm3AudioPipeline::load_from_model_dir(model_dir, context)
            .await
            .context("Failed to load LFM3-Audio pipeline")?;
        let tokenizer_dir = model_dir.join("audio_tokenizer");
        let codec = if tokenizer_dir.join("model.safetensors").is_file() {
            Some(
                MetalLiquidCodec::load_from_model_dir(&tokenizer_dir)
                    .context("Failed to load MetalLiquidCodec")?,
            )
        } else {
            None
        };
        let state = pipeline.acquire_state(lm).context("Failed to allocate audio VlmState")?;
        Ok(Self {
            pipeline,
            codec,
            state,
            session_id: None,
            cached_turns: Vec::new(),
            last_clean_eos: false,
            live_asr_session_id: None,
        })
    }

    pub fn is_active_session(&self, requested_session_id: Option<&str>) -> bool {
        requested_session_id.is_some()
            && self.session_id.as_deref() == requested_session_id
            && self.state.context_length() > 0
            && self.last_clean_eos
            && !self.cached_turns.is_empty()
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn generate_chat(
        &mut self,
        lm: &LanguageModel<Metal>,
        turns: &[AudioInputTurn],
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
    ) -> Result<AudioOutput> {
        self.generate_chat_streaming(lm, turns, mode, decode_audio, max_tokens, temperature, seed, |_| {})
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn generate_chat_streaming(
        &mut self,
        lm: &LanguageModel<Metal>,
        turns: &[AudioInputTurn],
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        on_text_delta: impl FnMut(&str),
    ) -> Result<AudioOutput> {
        self.generate_chat_with_session(
            lm,
            None,
            "cli_audio".to_string(),
            turns,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            on_text_delta,
        )
        .map(|(out, _)| out)
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn generate_chat_with_session(
        &mut self,
        lm: &LanguageModel<Metal>,
        requested_session_id: Option<&str>,
        fallback_session_id: String,
        turns: &[AudioInputTurn],
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        on_text_delta: impl FnMut(&str),
    ) -> Result<(AudioOutput, bool)> {
        self.generate_chat_with_options_streaming(
            lm,
            requested_session_id,
            fallback_session_id,
            turns,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            &AudioGenerateOptions::default(),
            on_text_delta,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_chat_with_options_streaming(
        &mut self,
        lm: &LanguageModel<Metal>,
        requested_session_id: Option<&str>,
        fallback_session_id: String,
        turns: &[AudioInputTurn],
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        options: &AudioGenerateOptions,
        on_text_delta: impl FnMut(&str),
        on_audio_chunk: impl FnMut(&[u8]),
    ) -> Result<(AudioOutput, bool)> {
        let tokenizer = lm.tokenizer();
        let t_prep = Instant::now();
        self.live_asr_session_id = None;

        let incoming_summary: Vec<AudioTurnSummary> = turns
            .iter()
            .map(|t| AudioTurnSummary {
                role: t.role.clone(),
                text: t.text.clone(),
                audio_hash: t.wav_bytes.as_deref().map(hash_audio_bytes),
            })
            .collect();

        let can_continue_state = self.state.context_length() > 0
            && self.last_clean_eos
            && !self.cached_turns.is_empty();
        let matches_prefix = can_continue_state
            && incoming_summary.len() > self.cached_turns.len()
            && incoming_summary[..self.cached_turns.len()] == self.cached_turns[..];
        let matches_session_delta = can_continue_state
            && requested_session_id.is_some()
            && self.session_id.as_deref() == requested_session_id
            && !matches_prefix
            && !turns.is_empty()
            && turns.iter().all(|t| t.role != "system");

        let (delta_start, include_bos, kv_reused) = if matches_prefix {
            (self.cached_turns.len(), false, true)
        } else if matches_session_delta {
            (0, false, true)
        } else {
            self.state.reset().context("Failed to reset audio VlmState")?;
            self.cached_turns.clear();
            self.last_clean_eos = false;
            self.session_id = Some(
                requested_session_id
                    .map(str::to_string)
                    .unwrap_or(fallback_session_id),
            );
            (0, true, false)
        };

        let delta_turns = &turns[delta_start..];
        let decoded_wavs: Vec<Option<Vec<f32>>> = delta_turns
            .iter()
            .map(|t| t.wav_bytes.as_deref().map(decode_wav_16khz_mono).transpose())
            .collect::<Result<_>>()?;
        let chat_turns: Vec<AudioChatTurn<'_>> = delta_turns
            .iter()
            .zip(decoded_wavs.iter())
            .map(|(t, wav)| AudioChatTurn {
                role: &t.role,
                text: &t.text,
                waveform: wav.as_deref(),
            })
            .collect();
        let prefill_str = options.assistant_prefill.as_deref();
        let mut prepared = self
            .pipeline
            .prepare_chat_with_prefill(tokenizer, lm, &chat_turns, include_bos, prefill_str)?;

        let mut actual_kv_reused = kv_reused;
        let max_ctx = lm.recommended_context_length().unwrap_or(8192) as usize;
        if actual_kv_reused && (self.state.context_length() as usize + prepared.prompt_tokens.len() + max_tokens > max_ctx)
        {
            self.state.reset().context("Failed to reset audio VlmState")?;
            self.cached_turns.clear();
            self.last_clean_eos = false;
            actual_kv_reused = false;
            let full_wavs: Vec<Option<Vec<f32>>> = turns
                .iter()
                .map(|t| t.wav_bytes.as_deref().map(decode_wav_16khz_mono).transpose())
                .collect::<Result<_>>()?;
            let full_turns: Vec<AudioChatTurn<'_>> = turns
                .iter()
                .zip(full_wavs.iter())
                .map(|(t, wav)| AudioChatTurn {
                    role: &t.role,
                    text: &t.text,
                    waveform: wav.as_deref(),
                })
                .collect();
            prepared = self
                .pipeline
                .prepare_chat_with_prefill(tokenizer, lm, &full_turns, true, prefill_str)?;
        }

        let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
        let out = self.run_prepared_with_options(
            lm,
            prepared,
            prep_ms,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            options,
            on_text_delta,
            on_audio_chunk,
        )?;

        if matches_session_delta && actual_kv_reused {
            self.cached_turns.extend(incoming_summary);
        } else {
            self.cached_turns = incoming_summary;
        }
        let eos_ids = &self.pipeline.config.eos_token_ids;
        self.last_clean_eos = out.tokens.last().is_some_and(|t| eos_ids.contains(t));
        if self.last_clean_eos {
            self.cached_turns.push(AudioTurnSummary {
                role: "assistant".to_string(),
                text: out.text.clone(),
                audio_hash: None,
            });
        }
        Ok((out, actual_kv_reused))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_omni_chat_with_options_streaming(
        &mut self,
        lm: &LanguageModel<Metal>,
        vision: &uzu_engine::vision::Lfm3VlPipeline,
        messages: &[uzu_engine::omni::OmniMessage],
        tools_preamble: Option<&str>,
        enable_thinking: bool,
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        options: &AudioGenerateOptions,
        on_text_delta: impl FnMut(&str),
        on_audio_chunk: impl FnMut(&[u8]),
    ) -> Result<AudioOutput> {
        self.state.reset().context("Failed to reset audio VlmState")?;
        self.session_id = None;
        self.live_asr_session_id = None;
        self.cached_turns.clear();
        self.last_clean_eos = false;

        let t_prep = std::time::Instant::now();
        let tokenizer = lm.tokenizer();
        let prepared = uzu_engine::omni::prepare_conversation_parts(
            vision,
            &self.pipeline,
            tokenizer,
            lm,
            messages,
            enable_thinking,
            tools_preamble,
            true,
            options.assistant_prefill.as_deref(),
        )?;
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
        self.run_prepared_with_options(
            lm,
            prepared,
            prep_ms,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            options,
            on_text_delta,
            on_audio_chunk,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        lm: &LanguageModel<Metal>,
        wav_bytes: Option<&[u8]>,
        system_prompt: Option<&str>,
        prompt: &str,
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
    ) -> Result<AudioOutput> {
        self.generate_with_options(
            lm,
            wav_bytes,
            system_prompt,
            prompt,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            &AudioGenerateOptions::default(),
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_options(
        &mut self,
        lm: &LanguageModel<Metal>,
        wav_bytes: Option<&[u8]>,
        system_prompt: Option<&str>,
        prompt: &str,
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        options: &AudioGenerateOptions,
        on_audio_chunk: impl FnMut(&[u8]),
    ) -> Result<AudioOutput> {
        self.state.reset().context("Failed to reset audio VlmState")?;
        self.session_id = None;
        self.live_asr_session_id = None;
        self.cached_turns.clear();
        self.last_clean_eos = false;
        let tokenizer = lm.tokenizer();
        let t_prep = Instant::now();
        let prepared = match wav_bytes {
            Some(bytes) => {
                let waveform = decode_wav_16khz_mono(bytes)?;
                self.pipeline
                    .prepare_transcribe_with_system(tokenizer, lm, system_prompt, prompt, &waveform)?
            },
            None => self.pipeline.prepare_prompt_with_system(tokenizer, system_prompt, prompt)?,
        };
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
        self.run_prepared_with_options(
            lm,
            prepared,
            prep_ms,
            mode,
            decode_audio,
            max_tokens,
            temperature,
            seed,
            options,
            |_| {},
            on_audio_chunk,
        )
    }

    pub fn live_asr_start(&mut self, lm: &LanguageModel<Metal>, session_id: String) -> Result<()> {
        self.state.reset().context("Failed to reset audio VlmState for LiveASR")?;
        self.session_id = None;
        self.cached_turns.clear();
        self.last_clean_eos = false;
        self.pipeline
            .live_asr_prefill_prompt(lm.tokenizer(), lm, &mut self.state)?;
        self.live_asr_session_id = Some(session_id);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn live_asr_chunk(
        &mut self,
        lm: &LanguageModel<Metal>,
        session_id: &str,
        audio_bytes: &[u8],
        format: Option<&str>,
        sample_rate: Option<u32>,
        is_final: bool,
        max_new_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
    ) -> Result<(String, u32)> {
        if self.live_asr_session_id.as_deref() != Some(session_id) {
            self.live_asr_start(lm, session_id.to_string())?;
        }
        let max_ctx = lm.recommended_context_length().unwrap_or(8192);
        if self.state.context_length() + (max_new_tokens as u32) + 64 >= max_ctx {
            self.live_asr_start(lm, session_id.to_string())?;
        }
        let waveform = if audio_bytes.is_empty() && is_final {
            Vec::new()
        } else {
            decode_audio_bytes_16khz_mono(audio_bytes, format, sample_rate)?
        };
        let sampler = VlmSampler::new(VlmSamplingConfig {
            temperature,
            seed: seed.map(|s| s as u64),
            ..Default::default()
        });
        let tokens = self.pipeline.live_asr_step(
            lm.tokenizer(),
            lm,
            &mut self.state,
            &waveform,
            is_final,
            max_new_tokens,
            sampler,
        )?;
        let text = decode_text_tokens(lm.tokenizer(), &self.pipeline, &tokens);
        if is_final {
            self.live_asr_session_id = None;
        }
        Ok((text, self.state.context_length()))
    }

    pub fn live_asr_stop(&mut self, session_id: &str) -> Result<()> {
        if self.live_asr_session_id.as_deref() == Some(session_id) {
            self.live_asr_session_id = None;
            self.state.reset().context("Failed to reset audio VlmState")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prepared_with_options(
        &mut self,
        lm: &LanguageModel<Metal>,
        mut prepared: PreparedAudioPrompt,
        prep_ms: f64,
        mode: Option<GenerationMode>,
        decode_audio: bool,
        max_tokens: usize,
        temperature: Option<f32>,
        seed: Option<i64>,
        options: &AudioGenerateOptions,
        mut on_text_delta: impl FnMut(&str),
        mut on_audio_chunk: impl FnMut(&[u8]),
    ) -> Result<AudioOutput> {
        let tokenizer = lm.tokenizer();

        let sampler = VlmSampler::new(VlmSamplingConfig {
            temperature,
            seed: seed.map(|s| s as u64),
            ..Default::default()
        });

        let has_custom_audio_sampling = options.audio_temperature.is_some()
            || options.audio_top_k.is_some()
            || options.audio_repetition_penalty.is_some()
            || options.audio_repetition_window.is_some();
        let audio_sampling = has_custom_audio_sampling.then(|| {
            let base = self.pipeline.config.audio_sampling;
            AudioSamplingConfig {
                temperature: options.audio_temperature.unwrap_or(base.temperature),
                top_k: options.audio_top_k.unwrap_or(base.top_k),
                repetition_penalty: options.audio_repetition_penalty.unwrap_or(base.repetition_penalty),
                repetition_window: options.audio_repetition_window.unwrap_or(base.repetition_window),
            }
        });
        let gen_opts = Lfm3AudioGenerateOptions {
            audio_sampling,
            ..Default::default()
        };

        let tool_start_id = tokenizer.token_to_id("<|tool_call_start|>").map(u64::from);
        let tool_end_id = tokenizer.token_to_id("<|tool_call_end|>").map(u64::from);
        let xml_tool_start_id = tokenizer.token_to_id("<tool_call>").map(u64::from);
        let xml_tool_end_id = tokenizer.token_to_id("</tool_call>").map(u64::from);
        let mut in_tool_call = false;
        let mut streamed_ids: Vec<u32> = Vec::new();
        let mut emitted_len = 0usize;

        let mut codec_ms = 0.0;
        let mut sample_rate = 24_000u32;
        let mut stream_state = if decode_audio {
            if let Some(codec) = self.codec.as_ref() {
                sample_rate = codec.config().sampling_rate as u32;
                Some(codec.new_streaming_state()?)
            } else {
                None
            }
        } else {
            None
        };
        let mut codec_ref = if decode_audio { self.codec.as_mut() } else { None };
        let k = self.pipeline.config.num_codebooks;
        let mut pending_frames: Vec<Vec<i64>> = Vec::new();
        let mut all_samples: Vec<f32> = Vec::new();
        let mut stream_decode_err: Option<anyhow::Error> = None;

        let generated = self.pipeline.generate_with_options_and_callbacks(
            tokenizer,
            lm,
            &mut self.state,
            &mut prepared,
            sampler,
            mode,
            max_tokens,
            None,
            &gen_opts,
            |token_id| {
                if Some(token_id) == tool_start_id || Some(token_id) == xml_tool_start_id {
                    in_tool_call = true;
                    return;
                }
                if Some(token_id) == tool_end_id || Some(token_id) == xml_tool_end_id {
                    in_tool_call = false;
                    return;
                }
                if in_tool_call {
                    return;
                }
                streamed_ids.push(token_id as u32);
                if let Ok(decoded) = tokenizer.decode(&streamed_ids, true)
                    && decoded.len() > emitted_len && !decoded.ends_with('\u{FFFD}') {
                        on_text_delta(&decoded[emitted_len..]);
                        emitted_len = decoded.len();
                    }
            },
            |frame| {
                if !decode_audio || stream_decode_err.is_some() {
                    return;
                }
                pending_frames.push(frame.to_vec());
                if pending_frames.len() >= 12
                    && let (Some(codec), Some(st)) = (codec_ref.as_deref_mut(), stream_state.as_mut()) {
                        let t_c = Instant::now();
                        let codes: Vec<Vec<i64>> = (0..k)
                            .map(|cb| pending_frames.iter().map(|f| f[cb]).collect())
                            .collect();
                        pending_frames.clear();
                        match codec.decode_streaming(&codes, st, false) {
                            Ok(samples) => {
                                codec_ms += t_c.elapsed().as_secs_f64() * 1000.0;
                                if !samples.is_empty() {
                                    all_samples.extend_from_slice(&samples);
                                    let wav_chunk = encode_wav_pcm16(&samples, sample_rate);
                                    on_audio_chunk(&wav_chunk);
                                }
                            },
                            Err(e) => {
                                stream_decode_err = Some(e);
                            },
                        }
                    }
            },
        )?;
        if let Some(err) = stream_decode_err {
            return Err(err);
        }

        let raw_decoded = decode_text_tokens(tokenizer, &self.pipeline, &generated.tokens);
        let raw_text = decode_text_tokens_with_specials(tokenizer, &self.pipeline, &generated.tokens);
        let (reasoning, text) = split_thinking_text(&raw_decoded, options.assistant_prefill.as_deref());
        if !raw_text.contains("<|tool_call_start|>") && !raw_text.contains("<tool_call>") && text.len() > emitted_len {
            on_text_delta(&text[emitted_len..]);
        }

        let wav_bytes = if decode_audio && !generated.frames.is_empty() {
            let codec = codec_ref.context("audio_tokenizer weights not loaded; cannot synthesize waveform")?;
            let st = stream_state.as_mut().context("streaming codec state missing")?;
            let t_c = Instant::now();
            let rem_codes: Vec<Vec<i64>> = if pending_frames.is_empty() {
                Vec::new()
            } else {
                (0..k)
                    .map(|cb| pending_frames.iter().map(|f| f[cb]).collect())
                    .collect()
            };
            let tail_samples = codec
                .decode_streaming(&rem_codes, st, true)
                .context("LiquidCodec streaming finalize failed")?;
            codec_ms += t_c.elapsed().as_secs_f64() * 1000.0;
            if !tail_samples.is_empty() {
                all_samples.extend_from_slice(&tail_samples);
                let wav_chunk = encode_wav_pcm16(&tail_samples, sample_rate);
                on_audio_chunk(&wav_chunk);
            }
            Some(encode_wav_pcm16(&all_samples, sample_rate))
        } else {
            None
        };

        Ok(AudioOutput {
            text,
            raw_text,
            reasoning,
            tokens: generated.tokens,
            frames: generated.frames,
            wav_bytes,
            sample_rate,
            prep_ms,
            prefill_ms: generated.prefill_ms,
            depth_ms: generated.depth_ms,
            backbone_ms: generated.backbone_ms,
            codec_ms,
        })
    }
}

fn split_thinking_text(raw: &str, assistant_prefill: Option<&str>) -> (Option<String>, String) {
    if let Some((before, after)) = raw.split_once("</think>") {
        let thinking = before
            .trim_start()
            .strip_prefix("<think>")
            .unwrap_or(before)
            .trim()
            .to_string();
        let answer = after.trim().to_string();
        let reasoning = (!thinking.is_empty()).then_some(thinking);
        return (reasoning, answer);
    }
    if assistant_prefill == Some("<think>") {
        let trimmed = raw.trim().to_string();
        return ((!trimmed.is_empty()).then_some(trimmed), String::new());
    }
    (None, raw.to_string())
}

pub fn decode_text_tokens(tokenizer: &Tokenizer, pipeline: &Lfm3AudioPipeline, tokens: &[u64]) -> String {
    decode_text_tokens_inner(tokenizer, pipeline, tokens, true)
}

pub fn decode_text_tokens_with_specials(tokenizer: &Tokenizer, pipeline: &Lfm3AudioPipeline, tokens: &[u64]) -> String {
    decode_text_tokens_inner(tokenizer, pipeline, tokens, false)
}

fn decode_text_tokens_inner(
    tokenizer: &Tokenizer,
    pipeline: &Lfm3AudioPipeline,
    tokens: &[u64],
    skip_special_tokens: bool,
) -> String {
    let audio_out = pipeline.config.audio_out_token_id;
    let text_end = pipeline.config.text_end_token_id;
    let eos = &pipeline.config.eos_token_ids;
    let text_ids: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|&id| id != audio_out && id != text_end && !eos.contains(&id))
        .map(|id| id as u32)
        .collect();
    if text_ids.is_empty() {
        return String::new();
    }
    tokenizer.decode(&text_ids, skip_special_tokens).unwrap_or_default()
}

/// Decode WAV or raw PCM16 bytes into 16 kHz mono f32 samples.
pub fn decode_audio_bytes_16khz_mono(
    bytes: &[u8],
    format: Option<&str>,
    sample_rate: Option<u32>,
) -> Result<Vec<f32>> {
    match format.map(|f| f.trim().to_ascii_lowercase()).as_deref() {
        Some("pcm16") | Some("pcm") => {
            if bytes.len() < 2 {
                return Ok(Vec::new());
            }
            let mono: Vec<f32> = bytes
                .as_chunks::<2>().0.iter()
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
            Ok(resample_to_16khz(mono, sample_rate.unwrap_or(16_000)))
        },
        _ => decode_wav_16khz_mono(bytes),
    }
}

fn resample_to_16khz(mono: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if sample_rate == 16_000 || mono.is_empty() || sample_rate == 0 {
        return mono;
    }
    let out_len = ((mono.len() as u64 * 16_000) / sample_rate as u64).max(1) as usize;
    let ratio = sample_rate as f64 / 16_000.0;
    let mut resampled = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = (src as usize).min(mono.len() - 1);
        let next = (idx + 1).min(mono.len() - 1);
        let frac = (src - idx as f64) as f32;
        resampled.push(mono[idx] * (1.0 - frac) + mono[next] * frac);
    }
    resampled
}

/// Decode a RIFF/WAVE file (PCM16 or IEEE float32) into 16 kHz mono f32 samples.
pub fn decode_wav_16khz_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("unsupported audio format: expected RIFF/WAVE");
    }
    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if payload_end - payload_start < 16 {
                bail!("invalid WAV fmt chunk");
            }
            let fmt = &bytes[payload_start..payload_end];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if id == b"data" {
            data_slice = Some(&bytes[payload_start..payload_end]);
            break;
        }
        pos = payload_start + size + (size & 1);
    }

    let data = data_slice.context("WAV file missing data chunk")?;
    if channels == 0 || sample_rate == 0 {
        bail!("WAV file missing valid fmt chunk");
    }

    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => data
            .as_chunks::<2>().0.iter()
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => data
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => bail!("unsupported WAV encoding: format={audio_format}, bits={bits_per_sample} (expected PCM16 or float32)"),
    };

    let ch = channels as usize;
    let mono: Vec<f32> = if ch == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    if sample_rate == 16_000 || mono.is_empty() {
        return Ok(mono);
    }

    let out_len = ((mono.len() as u64 * 16_000) / sample_rate as u64).max(1) as usize;
    let ratio = sample_rate as f64 / 16_000.0;
    let mut resampled = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = (src as usize).min(mono.len() - 1);
        let next = (idx + 1).min(mono.len() - 1);
        let frac = (src - idx as f64) as f32;
        resampled.push(mono[idx] * (1.0 - frac) + mono[next] * frac);
    }
    Ok(resampled)
}

/// Encode mono f32 samples (`[-1.0, 1.0]`) into a standard PCM16 RIFF/WAVE buffer.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}
