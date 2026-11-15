//! LFM3-Audio end-to-end path (Metal): log-mel STFT front-end on CPU,
//! conv downsampler, encoder trunk (`Decoder<Metal>`), and projector all
//! chained on Metal GPU, audio-feature injection into `<|audio|>` placeholders,
//! and the text/speech slot generation loop (see
//! `modeling_lfm3_audio.py::Lfm3AudioForConditionalGeneration._generate` /
//! mlx `stream_slots`).
//!
//! Speech frames are produced on Metal GPU by the MaskGIT depth decoder
//! (`MetalDepth`) from the backbone hidden state, embedded through the shared
//! `codebook_embedding` table and fed back as a one-row feature injection.
//! Generated frames decode to 24 kHz waveforms on Metal GPU through
//! `MetalLiquidCodec`, outside this module.

use std::collections::HashSet;
use std::path::Path;
use std::range::Range;

use futures::executor::block_on;
use half::bf16;
use tokenizers::Tokenizer;

use crate::backends::common::{
    Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
};
use crate::backends::metal::Metal;
use crate::audio::lfm3_audio::features::MelFilterBank;
use crate::audio::lfm3_audio::features_metal::MetalAudioFeatures;
use crate::audio::lfm3_audio::{AudioSamplingConfig, DepthConfig, DepthDecoder, DepthWeights, Lfm3AudioConfig};
use crate::data_type::DataType;
use crate::encodable_block::batch_topology::BatchTopology;
use crate::encodable_block::decoder::Decoder;
use crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY;
use crate::engine::language_model::LanguageModel;
use crate::safetensors::ParameterLoader;
use crate::vision::common::encode_vlm_sampling;
use crate::vision::tower::SafetensorLoader;
use crate::vision::VlmState;
use crate::vision::VlmSampler;

const DEFAULT_DEPTH_SEED: u64 = 0x05EE_D670;

/// Sampling mask: bit set = id allowed. Covers ids beyond the real tokenizer
/// vocabulary and the caller's banned tokens (speech placeholders etc.).
pub fn sampling_mask(vocab_size: usize, vocab_limit: usize, banned: &[u64]) -> Vec<u32> {
    let words = vocab_size.div_ceil(32);
    let mut mask = vec![u32::MAX; words];
    let limit = vocab_limit.min(vocab_size);
    if limit % 32 != 0 {
        mask[limit / 32] &= (1u32 << (limit % 32)) - 1;
    }
    for word in mask.iter_mut().skip(limit.div_ceil(32)) {
        *word = 0;
    }
    for &id in banned {
        if (id as usize) < vocab_size {
            mask[id as usize / 32] &= !(1u32 << (id as usize % 32));
        }
    }
    mask
}

/// Generation mode, mirroring the reference's `Lfm3AudioGenerationMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationMode {
    Text,
    Audio,
    Interleaved,
}

#[derive(Clone, Copy, Debug)]
pub struct AudioChatTurn<'a> {
    pub role: &'a str,
    pub text: &'a str,
    pub waveform: Option<&'a [f32]>,
}

#[derive(Clone, Debug, Default)]
pub struct PreparedAudioPrompt {
    pub prompt_tokens: Vec<u64>,
    /// Projector rows (bf16 bits), one per injected audio token.
    pub features_bf16: Vec<u8>,
    /// `(start_token_index, count)` runs of injected audio rows.
    pub audio_slices: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct Lfm3AudioGenerateOptions {
    pub audio_sampling: Option<AudioSamplingConfig>,
    pub suppress_token_ids: Vec<u64>,
    pub stop_token_ids: Vec<u64>,
}

#[derive(Debug, Default)]
pub struct GenerationOutput {
    /// Every generated slot: text tokens as ids, audio frames as
    /// `Lfm3AudioConfig::audio_out_token_id`.
    pub tokens: Vec<u64>,
    /// One code vector per speech frame, EOS frames excluded.
    pub frames: Vec<Vec<i64>>,
    pub prefill_ms: f64,
    pub depth_ms: f64,
    pub backbone_ms: f64,
}

enum StepInput {
    Token(u64),
    /// Uses the pre-allocated `Lfm3AudioPipeline::step_embed_buf` (`bf16`, `model_dim`).
    PreloadedEmbedding,
}

pub struct StepOutput {
    /// Sampled text token, present when a sampler was supplied.
    sampled: Option<u64>,
}

pub struct Lfm3AudioPipeline {
    pub config: Lfm3AudioConfig,
    pub mel: MelFilterBank,
    features: std::sync::Mutex<MetalAudioFeatures>,
    trunk: Decoder<Metal>,
    pub depth: DepthDecoder,
    /// Shared `[num_codebooks * codebook_size, model_dim]` table (bf16 bits).
    codebook: Vec<u16>,
    model_dim: usize,
    /// Pre-allocated per-step GPU buffers (zero allocation during decode).
    step_hidden_buf: <Metal as Backend>::GlobalGpuBuffer,
    step_embed_buf: <Metal as Backend>::GlobalGpuBuffer,
    step_token_buf: <Metal as Backend>::GlobalGpuBuffer,
    plain_tokenizer_cache: std::sync::OnceLock<Tokenizer>,
}

impl Lfm3AudioPipeline {
    /// Loads if `audio_config.json` is present in the bundle.
    pub fn is_lfm3_audio(model_dir: &Path) -> bool {
        model_dir.join("audio_config.json").is_file()
    }

    pub async fn load_from_model_dir(
        model_dir: &std::path::Path,
        context: &<Metal as Backend>::Context,
    ) -> Result<Self, anyhow::Error> {
        let config = Lfm3AudioConfig::load(model_dir)?;
        let audio_weights_path = model_dir.join("audio.safetensors");
        let mut loader = SafetensorLoader::open(&audio_weights_path).map_err(|e| anyhow::anyhow!("{e}"))?;

        let mel = MelFilterBank::slaney(config.sample_rate, config.n_fft, config.num_mel_bins);
        let features = std::sync::Mutex::new(MetalAudioFeatures::load(
            &mut loader,
            context,
            config.num_mel_bins,
            config.conv_channels,
            config.encoder_hidden,
        )?);

        let depth_weights = DepthWeights::load(&mut loader, config.depth.layers, config.num_codebooks)?;
        let depth = DepthDecoder::with_metal(
            depth_weights,
            DepthConfig::clone(&config.depth),
            DEFAULT_DEPTH_SEED,
            context,
        )?;

        let (codebook_raw, codebook_shape) =
            loader.read_tensor_f32("audio.codebook_embedding").map_err(|e| anyhow::anyhow!("{e}"))?;
        anyhow::ensure!(
            codebook_shape[0] == config.num_codebooks * config.codebook_size,
            "unexpected codebook table shape {codebook_shape:?}"
        );
        let model_dim = codebook_shape[1];
        let codebook = codebook_raw.iter().map(|&v| bf16::from_f32(v).to_bits()).collect::<Vec<_>>();

        let audio_loader = ParameterLoader::<Metal>::new(&audio_weights_path, context)
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let trunk = Decoder::new(
            DataType::BF16,
            &config.encoder_decoder_config,
            &audio_loader.tree().subtree("audio.encoder"),
            context,
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let hidden_bytes = model_dim * std::mem::size_of::<bf16>();
        let step_hidden_buf =
            context.create_buffer_cpu_gpu(hidden_bytes).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let step_embed_buf =
            context.create_buffer_cpu_gpu(hidden_bytes).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let step_token_buf =
            context.create_buffer_cpu_gpu(std::mem::size_of::<u32>()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        Ok(Self {
            config,
            mel,
            features,
            trunk,
            depth,
            codebook,
            model_dim,
            step_hidden_buf,
            step_embed_buf,
            step_token_buf,
            plain_tokenizer_cache: std::sync::OnceLock::new(),
        })
    }

    fn plain_tokenizer(&self, tokenizer: &Tokenizer) -> Result<&Tokenizer, anyhow::Error> {
        if let Some(cached) = self.plain_tokenizer_cache.get() {
            return Ok(cached);
        }
        let plain =
            crate::utils::tokenizer::build_plain_tokenizer(tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
        let _ = self.plain_tokenizer_cache.set(plain);
        Ok(self.plain_tokenizer_cache.get().expect("just initialized"))
    }

    fn chat_special_ids(tokenizer: &Tokenizer) -> Result<(u64, u64, u64), anyhow::Error> {
        let req = |tok: &str| {
            tokenizer
                .token_to_id(tok)
                .map(|id| id as u64)
                .ok_or_else(|| anyhow::anyhow!("tokenizer missing special token {tok}"))
        };
        Ok((req("<|startoftext|>")?, req("<|im_start|>")?, req("<|im_end|>")?))
    }

    /// Waveform (16 kHz mono) -> injected feature rows (bf16 bits).
    pub fn encode_audio_features(&self, model: &LanguageModel<Metal>, waveform: &[f32]) -> Result<Vec<u8>, anyhow::Error> {
        let num_frames = self.config.num_frames(waveform.len());
        let mel = self.mel.log_mel(
            waveform,
            self.config.n_fft,
            self.config.hop_length,
            num_frames,
            self.config.log_floor,
            self.config.norm_offset,
            self.config.norm_divisor,
        );

        let context = model.context();
        let mut command_buffer = context
            .create_command_buffer(Some("lfm3_audio_encoder"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut feat_guard = self.features.lock().map_err(|_| anyhow::anyhow!("features lock poisoned"))?;
        let (downsampled_gpu, tokens, ds_scratch) = feat_guard.encode_downsampler(
            context,
            &mut command_buffer,
            &mel,
            self.config.num_mel_bins,
            num_frames,
        )?;
        anyhow::ensure!(tokens > 0, "clip too short for any encoder token");

        let mut trunk_state =
            self.trunk.create_empty_state(None, context).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        trunk_state.prepare(0, tokens as u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let token_ids: Vec<u32> = vec![1; tokens];
        let token_buffer = command_buffer
            .allocate_constant_from_slice(&token_ids)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let batch_dim = BatchTopology::flat(tokens as u32, true);
        let output = self
            .trunk
            .encode_multimodal(
                &token_buffer,
                Some(&downsampled_gpu),
                Some(&[(0usize, tokens)]),
                None,
                0,
                &batch_dim,
                Some(Range { start: 0, end: tokens as u32 }),
                Some(&[]),
                &mut trunk_state,
                &mut command_buffer,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let trunk_hidden = output
            .final_hidden
            .ok_or_else(|| anyhow::anyhow!("encoder trunk produced no hidden state"))?;
        let (projected_gpu, proj_scratch) =
            feat_guard.encode_projector(context, &mut command_buffer, &trunk_hidden, tokens)?;
        drop(feat_guard);
        drop(token_buffer);
        block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        drop(ds_scratch);
        drop(downsampled_gpu);
        drop(proj_scratch);
        block_on(context.copyout(&projected_gpu)).map_err(|e| anyhow::anyhow!("{e:?}"))
    }

    /// Audio-in prompt: user text with `<|audio|>` expanded to one placeholder
    /// per encoder token, features attached.
    pub fn prepare_transcribe(
        &self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        user_text: &str,
        waveform: &[f32],
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        self.prepare_transcribe_with_system(tokenizer, model, None, user_text, waveform)
    }

    /// Audio-in prompt with an optional system instruction (e.g. `"Perform ASR in English."`).
    pub fn prepare_transcribe_with_system(
        &self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        system_prompt: Option<&str>,
        user_text: &str,
        waveform: &[f32],
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        let mut turns = Vec::with_capacity(2);
        if let Some(sys) = system_prompt.filter(|s| !s.is_empty()) {
            turns.push(AudioChatTurn { role: "system", text: sys, waveform: None });
        }
        turns.push(AudioChatTurn { role: "user", text: user_text, waveform: Some(waveform) });
        self.prepare_chat(tokenizer, model, &turns)
    }

    /// Multi-turn chat prompt with optional per-turn 16 kHz waveforms.
    pub fn prepare_chat(
        &self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        turns: &[AudioChatTurn<'_>],
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        self.prepare_chat_with_options(tokenizer, model, turns, true)
    }

    /// Multi-turn or delta-turn chat prompt (`include_bos == false` continues after a previous `<|im_end|>`).
    pub fn prepare_chat_with_options(
        &self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        turns: &[AudioChatTurn<'_>],
        include_bos: bool,
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        self.prepare_chat_with_prefill(tokenizer, model, turns, include_bos, None)
    }

    /// Multi-turn or delta-turn chat prompt with optional assistant prefill (e.g. `"<think>"` or `"<think></think>"`).
    pub fn prepare_chat_with_prefill(
        &self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        turns: &[AudioChatTurn<'_>],
        include_bos: bool,
        assistant_prefill: Option<&str>,
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        let plain_tokenizer = self.plain_tokenizer(tokenizer)?;
        let (bos_id, im_start_id, im_end_id) = Self::chat_special_ids(tokenizer)?;
        let audio_token_id = self.config.audio_token_id;

        let mut prompt_tokens: Vec<u64> = Vec::new();
        if include_bos {
            prompt_tokens.push(bos_id);
        } else {
            let nl = plain_tokenizer.encode("\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
            prompt_tokens.extend(nl.get_ids().iter().map(|&id| id as u64));
        }

        let mut features_bf16 = Vec::new();
        let mut audio_slices = Vec::new();
        for turn in turns {
            prompt_tokens.push(im_start_id);
            let role_hdr = plain_tokenizer
                .encode(format!("{}\n", turn.role).as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            prompt_tokens.extend(role_hdr.get_ids().iter().map(|&id| id as u64));
            if let Some(wav) = turn.waveform {
                let features = self.encode_audio_features(model, wav)?;
                let rows = features.len() / 2 / self.model_dim;
                let start = prompt_tokens.len();
                prompt_tokens.resize(start + rows, audio_token_id);
                audio_slices.push((start, rows));
                features_bf16.extend_from_slice(&features);
            }
            if !turn.text.is_empty() {
                let body = plain_tokenizer
                    .encode(turn.text, false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                prompt_tokens.extend(body.get_ids().iter().map(|&id| id as u64));
            }
            prompt_tokens.push(im_end_id);
            let nl = plain_tokenizer.encode("\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
            prompt_tokens.extend(nl.get_ids().iter().map(|&id| id as u64));
        }

        prompt_tokens.push(im_start_id);
        let assist = plain_tokenizer
            .encode("assistant\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        prompt_tokens.extend(assist.get_ids().iter().map(|&id| id as u64));
        if let Some(prefill) = assistant_prefill.filter(|s| !s.is_empty()) {
            let pre = tokenizer.encode(prefill, false).map_err(|e| anyhow::anyhow!("{e}"))?;
            prompt_tokens.extend(pre.get_ids().iter().map(|&id| id as u64));
        }

        Ok(PreparedAudioPrompt {
            prompt_tokens,
            audio_slices,
            features_bf16,
        })
    }

    /// Text-only prompt (speech generation); the loop infers the mode from
    /// the prompt tail and the generated markers.
    pub fn prepare_prompt(&self, tokenizer: &Tokenizer, user_text: &str) -> Result<PreparedAudioPrompt, anyhow::Error> {
        self.prepare_prompt_with_system(tokenizer, None, user_text)
    }

    /// Text-only prompt with an optional system instruction (e.g. `"Perform TTS in American English using a female voice."`).
    pub fn prepare_prompt_with_system(
        &self,
        tokenizer: &Tokenizer,
        system_prompt: Option<&str>,
        user_text: &str,
    ) -> Result<PreparedAudioPrompt, anyhow::Error> {
        let plain_tokenizer = self.plain_tokenizer(tokenizer)?;
        let (bos_id, im_start_id, im_end_id) = Self::chat_special_ids(tokenizer)?;
        let nl = plain_tokenizer.encode("\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut prompt_tokens: Vec<u64> = vec![bos_id];
        if let Some(sys) = system_prompt.filter(|s| !s.is_empty()) {
            prompt_tokens.push(im_start_id);
            let enc = plain_tokenizer
                .encode(format!("system\n{sys}").as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            prompt_tokens.extend(enc.get_ids().iter().map(|&id| id as u64));
            prompt_tokens.push(im_end_id);
            prompt_tokens.extend(nl.get_ids().iter().map(|&id| id as u64));
        }
        prompt_tokens.push(im_start_id);
        let user_enc = plain_tokenizer
            .encode(format!("user\n{user_text}").as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        prompt_tokens.extend(user_enc.get_ids().iter().map(|&id| id as u64));
        prompt_tokens.push(im_end_id);
        prompt_tokens.extend(nl.get_ids().iter().map(|&id| id as u64));
        prompt_tokens.push(im_start_id);
        let assist = plain_tokenizer
            .encode("assistant\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        prompt_tokens.extend(assist.get_ids().iter().map(|&id| id as u64));

        Ok(PreparedAudioPrompt {
            prompt_tokens,
            audio_slices: Vec::new(),
            features_bf16: Vec::new(),
        })
    }

    /// Empty backbone KV state sized to the model recommendation.
    pub fn acquire_state(&self, model: &LanguageModel<Metal>) -> Result<VlmState, anyhow::Error> {
        let state = model
            .decoder()
            .create_empty_state(model.recommended_context_length(), model.context())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(VlmState { state, context_ring: None })
    }

    /// Chunked prefill with audio rows injected at their placeholder runs.
    /// Returns the last position's hidden state and, when a sampler is given,
    /// the sampled first slot token.
    pub fn prefill(
        &mut self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &PreparedAudioPrompt,
        mut sampler: Option<&mut VlmSampler>,
        mask: &[u32],
    ) -> Result<StepOutput, anyhow::Error> {
        let context = model.context();
        let max_batch = ATTENTION_SUFFIX_CAPACITY as usize;
        let total = prepared.prompt_tokens.len();
        anyhow::ensure!(total > 0, "empty prompt");
        let batch_count = total.div_ceil(max_batch);
        let prefix_len = state.state.context_length;
        state
            .state
            .prepare(
                state.state.context_length + ((batch_count - 1) * max_batch) as u32,
                usize::min(max_batch, total) as u32,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let elems_per_token = model.decoder().model_dim() as usize;
        let mut command_buffer = context
            .create_command_buffer(Some("lfm3_audio_prefill"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut sampled_buf: Option<<Metal as Backend>::GlobalGpuBuffer> = None;
        let mut hidden_buf: Option<<Metal as Backend>::ScratchBuffer> = None;
        let mut features_before = 0usize;
        let mut pos = 0usize;
        for batch in 0..batch_count {
            let is_last = batch + 1 == batch_count;
            let len = usize::min(max_batch, total - pos);
            let input_u32: Vec<u32> = prepared.prompt_tokens[pos..pos + len].iter().map(|&x| x as u32).collect();
            let token_ids = command_buffer
                .allocate_constant_from_slice(&input_u32)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;

            let chunk_end = pos + len;
            let mut chunk_slices: Vec<(usize, usize)> = Vec::new();
            let mut chunk_audio = 0usize;
            for &(start, count) in &prepared.audio_slices {
                let overlap_start = start.max(pos);
                let overlap_end = (start + count).min(chunk_end);
                if overlap_end > overlap_start {
                    chunk_slices.push((overlap_start - pos, overlap_end - overlap_start));
                    chunk_audio += overlap_end - overlap_start;
                }
            }
            let chunk_features: Option<&[u8]> = (chunk_audio > 0).then(|| {
                &prepared.features_bf16
                    [features_before * elems_per_token * 2..(features_before + chunk_audio) * elems_per_token * 2]
            });
            features_before += chunk_audio;
            let features_gpu = chunk_features
                .map(|features| command_buffer.allocate_constant_from_slice(features))
                .transpose()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let chunk_slices = (!chunk_slices.is_empty()).then_some(chunk_slices);

            let batch_dim = BatchTopology::flat(len as u32, true);
            let output = model
                .decoder()
                .encode_multimodal_with_logits(
                    &token_ids,
                    features_gpu.as_ref(),
                    chunk_slices.as_ref().map(|s| &s[..]),
                    None,
                    0,
                    &batch_dim,
                    is_last.then(|| Range { start: len as u32 - 1, end: len as u32 }),
                    Some(&[]),
                    is_last && sampler.is_some(),
                    &mut state.state,
                    &mut command_buffer,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;

            if is_last {
                if let Some(logits) = output.logits.as_ref()
                    && let Some(sampler) = sampler.as_deref_mut() {
                        sampled_buf = Some(
                            encode_vlm_sampling(
                                model,
                                &mut command_buffer,
                                logits,
                                &token_ids,
                                None,
                                &batch_dim,
                                Range { start: len as u32 - 1, end: len as u32 },
                                sampler,
                                mask,
                                prefix_len + total as u32 - 1,
                            )
                            .map_err(|e| anyhow::anyhow!("{e}"))?,
                        );
                    }
                hidden_buf = output.final_hidden;
            }
            state
                .state
                .encode_accept(&(0..len as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            drop(token_ids);
            drop(features_gpu);
            pos += len;
        }
        let scratch = hidden_buf.ok_or_else(|| anyhow::anyhow!("prefill produced no hidden state"))?;
        command_buffer.encode_copy(&scratch, &mut self.step_hidden_buf);
        block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let sampled = read_sampled_token(context, sampled_buf.as_ref(), sampler)?;
        Ok(StepOutput { sampled })
    }

    /// One decode step: `input` is the previously produced slot content.
    /// When `sampler` is given, the step also samples the next text token.
    fn forward_step(
        &mut self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        input: &StepInput,
        mut sampler: Option<&mut VlmSampler>,
        mask: &[u32],
    ) -> Result<StepOutput, anyhow::Error> {
        let context = model.context();
        let prefix_len = state.state.context_length;
        state.state.prepare(prefix_len, 1).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let token = match input {
            StepInput::Token(token) => *token as u32,
            StepInput::PreloadedEmbedding => 0,
        };
        self.step_token_buf.as_slice_mut::<u32>()[0] = token;
        let features_gpu = matches!(input, StepInput::PreloadedEmbedding).then_some(&self.step_embed_buf);
        let slices = features_gpu.map(|_| [(0usize, 1usize)] as [(usize, usize); 1]);
        static PROFILE_DECODE: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("UZU_PROFILE_DECODE").is_ok());
        let profile = *PROFILE_DECODE;
        let command_buffer = context
            .create_command_buffer(Some("lfm3_audio_step"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut command_buffer = if profile {
            command_buffer.with_timestamps()
        } else {
            command_buffer
        };
        let batch_dim = BatchTopology::flat(1, true);
        let output = model
            .decoder()
            .encode_multimodal_with_logits(
                &self.step_token_buf,
                features_gpu,
                slices.as_ref().map(|s| &s[..]),
                None,
                0,
                &batch_dim,
                Some(Range { start: 0, end: 1 }),
                Some(&[]),
                sampler.is_some(),
                &mut state.state,
                &mut command_buffer,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let sampled_buf = match (output.logits.as_ref(), sampler.as_deref_mut()) {
            (Some(logits), Some(sampler)) => Some(
                encode_vlm_sampling(
                    model,
                    &mut command_buffer,
                    logits,
                    &self.step_token_buf,
                    None,
                    &batch_dim,
                    Range { start: 0, end: 1 },
                    sampler,
                    mask,
                    prefix_len,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            ),
            _ => None,
        };
        let scratch = output.final_hidden.ok_or_else(|| anyhow::anyhow!("decode step produced no hidden state"))?;
        command_buffer.encode_copy(&scratch, &mut self.step_hidden_buf);
        state.state.encode_accept(&[0], &mut command_buffer).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        if profile && prefix_len % 32 == 0 {
            use crate::backends::common::CommandBufferCompleted;
            let spans = completed.timestamps();
            let names: Vec<&str> = spans.iter().map(|span| span.name.as_str()).collect();
            let mut totals: std::collections::HashMap<String, std::time::Duration> = std::collections::HashMap::new();
            for span in spans {
                if names.iter().any(|name| {
                    name.len() > span.name.len()
                        && name.starts_with(&span.name)
                        && name[span.name.len()..].starts_with('/')
                }) {
                    continue;
                }
                let duration = span.end.saturating_duration_since(span.start);
                let leaf = span.name.rsplit('/').next().unwrap_or(&span.name).to_string();
                *totals.entry(leaf).or_default() += duration;
            }
            let mut sorted: Vec<_> = totals.into_iter().collect();
            sorted.sort_by_key(|(_, duration)| std::cmp::Reverse(*duration));
            let gpu_sum: std::time::Duration = sorted.iter().map(|(_, d)| *d).sum();
            eprintln!("PROFILE_DECODE pos={prefix_len} gpu_sum={:.3}ms:", gpu_sum.as_secs_f64() * 1000.0);
            for (name, dur) in sorted.iter().take(12) {
                eprintln!("  {name}: {:.3}ms", dur.as_secs_f64() * 1000.0);
            }
        }

        let sampled = read_sampled_token(context, sampled_buf.as_ref(), sampler)?;
        Ok(StepOutput { sampled })
    }

    /// Sum of the codebook rows of one frame as `f32` (rounded at bf16 boundaries).
    pub fn frame_embedding_f32(&self, codes: &[i64]) -> Vec<f32> {
        let mut buf = vec![bf16::ZERO; self.model_dim];
        write_frame_embedding(&self.codebook, self.config.codebook_size, self.model_dim, codes, &mut buf);
        buf.into_iter().map(bf16::to_f32).collect()
    }

    /// The reference slot loop (batch 1). `generation_mode` `None` infers the
    /// mode from the prompt tail; explicit `Text` additionally bans the
    /// speech markers. `frame_orders` pins the MaskGIT reveal order per frame.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &mut PreparedAudioPrompt,
        sampler: VlmSampler,
        generation_mode: Option<GenerationMode>,
        max_tokens: usize,
        frame_orders: Option<&[Vec<usize>]>,
    ) -> Result<GenerationOutput, anyhow::Error> {
        self.generate_with_callback(
            tokenizer,
            model,
            state,
            prepared,
            sampler,
            generation_mode,
            max_tokens,
            frame_orders,
            |_| {},
        )
    }

    /// End-to-end generation loop with a per-text-token streaming callback.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_callback(
        &mut self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &mut PreparedAudioPrompt,
        sampler: VlmSampler,
        generation_mode: Option<GenerationMode>,
        max_tokens: usize,
        frame_orders: Option<&[Vec<usize>]>,
        on_text_token: impl FnMut(u64),
    ) -> Result<GenerationOutput, anyhow::Error> {
        self.generate_with_options_and_callbacks(
            tokenizer,
            model,
            state,
            prepared,
            sampler,
            generation_mode,
            max_tokens,
            frame_orders,
            &Lfm3AudioGenerateOptions::default(),
            on_text_token,
            |_| {},
        )
    }

    /// End-to-end generation loop with audio sampling overrides and per-token / per-frame callbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_options_and_callbacks(
        &mut self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &mut PreparedAudioPrompt,
        mut sampler: VlmSampler,
        generation_mode: Option<GenerationMode>,
        max_tokens: usize,
        frame_orders: Option<&[Vec<usize>]>,
        options: &Lfm3AudioGenerateOptions,
        mut on_text_token: impl FnMut(u64),
        mut on_audio_frame: impl FnMut(&[i64]),
    ) -> Result<GenerationOutput, anyhow::Error> {
        let audio_token_id = self.config.audio_token_id;
        let audio_out_token_id = self.config.audio_out_token_id;
        let audio_start_token_id = self.config.audio_start_token_id;
        let text_end_token_id = self.config.text_end_token_id;
        let mixed_start_token_id = self.config.mixed_start_token_id;
        let mut eos: HashSet<u64> = self.config.eos_token_ids.iter().copied().collect();
        eos.extend(options.stop_token_ids.iter().copied());
        let last = *prepared.prompt_tokens.last().ok_or_else(|| anyhow::anyhow!("empty prompt"))?;
        let mode = match generation_mode {
            Some(GenerationMode::Audio) => {
                if last != audio_start_token_id && last != audio_out_token_id {
                    prepared.prompt_tokens.push(audio_start_token_id);
                }
                GenerationMode::Audio
            },
            Some(GenerationMode::Interleaved) => {
                if last != mixed_start_token_id {
                    prepared.prompt_tokens.push(mixed_start_token_id);
                }
                GenerationMode::Interleaved
            },
            Some(GenerationMode::Text) => GenerationMode::Text,
            None => {
                if last == audio_start_token_id || last == audio_out_token_id {
                    GenerationMode::Audio
                } else if last == mixed_start_token_id {
                    GenerationMode::Interleaved
                } else {
                    GenerationMode::Text
                }
            },
        };

        let mut banned = vec![audio_token_id, audio_out_token_id];
        if mode == GenerationMode::Text {
            banned.push(audio_start_token_id);
            banned.push(mixed_start_token_id);
        }
        banned.extend(options.suppress_token_ids.iter().copied());
        let mask = sampling_mask(
            model.vocab_size(),
            tokenizer.get_vocab_size(true),
            &banned,
        );

        let t_prefill = std::time::Instant::now();
        let first = self.prefill(model, state, prepared, Some(&mut sampler), &mask)?;
        let mut sampled = first.sampled;

        let mut out = GenerationOutput {
            prefill_ms: t_prefill.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        };
        let audio_sampling = options.audio_sampling.unwrap_or(self.config.audio_sampling);
        let mut mode = mode;
        let mut slots_left = self.config.interleaved_n_text;
        let mut speaking = false;
        let mut text_done = false;
        let mut finished = false;
        let mut audio_slot_count = 0usize;
        let mut audio_history: Vec<Vec<i64>> = Vec::new();
        for step in 0..max_tokens {
            let interleaving = mode == GenerationMode::Interleaved;
            let audio_slot = mode == GenerationMode::Audio || (interleaving && speaking);
            let next_input;
            if audio_slot {
                let order = frame_orders.and_then(|orders| orders.get(audio_slot_count).map(Vec::as_slice));
                // Pinning `frame_orders` implies deterministic greedy eval against the MLX reference.
                let (audio_temp, audio_top_k, audio_rep_pen) = if frame_orders.is_some() {
                    (None, None, None)
                } else {
                    (
                        Some(audio_sampling.temperature),
                        Some(audio_sampling.top_k),
                        Some(audio_sampling.repetition_penalty),
                    )
                };
                let history_window = if audio_sampling.repetition_window == 0 {
                    &audio_history[..]
                } else {
                    let start = audio_history.len().saturating_sub(audio_sampling.repetition_window);
                    &audio_history[start..]
                };
                let t_depth = std::time::Instant::now();
                let mut codes = self.depth.sample_frame_gpu(
                    model.context(),
                    &self.step_hidden_buf,
                    &self.config.maskgit_schedule,
                    audio_temp,
                    audio_top_k,
                    audio_rep_pen,
                    history_window,
                    order,
                )?;
                out.depth_ms += t_depth.elapsed().as_secs_f64() * 1000.0;
                audio_slot_count += 1;
                self.depth.normalize_eos(&mut codes, self.config.audio_eos_code);
                out.tokens.push(audio_out_token_id);
                if self.config.is_eos_frame(&codes) {
                    mode = GenerationMode::Text;
                } else {
                    on_audio_frame(&codes);
                    audio_history.push(codes.clone());
                    out.frames.push(codes.clone());
                }
                write_frame_embedding(
                    &self.codebook,
                    self.config.codebook_size,
                    self.model_dim,
                    &codes,
                    self.step_embed_buf.as_slice_mut::<bf16>(),
                );
                next_input = StepInput::PreloadedEmbedding;
            } else {
                let token = sampled.ok_or_else(|| anyhow::anyhow!("text slot without a sampled token"))?;
                out.tokens.push(token);
                next_input = StepInput::Token(token);
                if token == text_end_token_id {
                    if interleaving {
                        text_done = true;
                        speaking = true;
                    } else {
                        finished = true;
                    }
                } else if eos.contains(&token) {
                    finished = true;
                } else if interleaving {
                    // mode markers are plain text inside a block
                    on_text_token(token);
                } else if token == audio_start_token_id {
                    mode = GenerationMode::Audio;
                } else if token == mixed_start_token_id {
                    mode = GenerationMode::Interleaved;
                    slots_left = self.config.interleaved_n_text;
                    speaking = false;
                    text_done = false;
                } else {
                    on_text_token(token);
                }
            }
            if interleaving && !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = !speaking;
                    slots_left = if speaking {
                        self.config.interleaved_n_audio
                    } else {
                        self.config.interleaved_n_text
                    };
                }
            }
            if finished || step == max_tokens - 1 {
                break;
            }
            let next_is_audio = mode == GenerationMode::Audio || (mode == GenerationMode::Interleaved && speaking);
            let t_step = std::time::Instant::now();
            let step_output = self.forward_step(
                model,
                state,
                &next_input,
                (!next_is_audio).then_some(&mut sampler),
                &mask,
            )?;
            out.backbone_ms += t_step.elapsed().as_secs_f64() * 1000.0;
            sampled = step_output.sampled;
        }
        Ok(out)
    }

    /// Prefill the LiveASR system + assistant header into a fresh `VlmState`.
    /// Official LiveASR uses `"Perform live ASR."` (English only).
    pub fn live_asr_prefill_prompt(
        &mut self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
    ) -> Result<(), anyhow::Error> {
        let prompt = "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n";
        let prepared = tokenize_prompt(tokenizer, prompt)?;
        let mask = sampling_mask(model.vocab_size(), tokenizer.get_vocab_size(true), &[]);
        self.prefill(model, state, &prepared, None, &mask)?;
        Ok(())
    }

    /// Feed one ~0.96s (16 kHz) audio chunk into an active LiveASR `VlmState` and
    /// decode incremental transcript tokens for this chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn live_asr_step(
        &mut self,
        tokenizer: &Tokenizer,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        chunk_waveform: &[f32],
        is_final: bool,
        max_new_tokens: usize,
        mut sampler: VlmSampler,
    ) -> Result<Vec<u64>, anyhow::Error> {
        if chunk_waveform.is_empty() && !is_final {
            return Ok(Vec::new());
        }
        let mut padded;
        let wav = if chunk_waveform.len() < self.config.n_fft {
            padded = chunk_waveform.to_vec();
            padded.resize(self.config.n_fft, 0.0);
            &padded[..]
        } else {
            chunk_waveform
        };
        let features = self.encode_audio_features(model, wav)?;
        let rows = features.len() / 2 / self.model_dim;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let audio_token_id = self.config.audio_token_id;
        let mut prepared = PreparedAudioPrompt {
            prompt_tokens: vec![audio_token_id; rows],
            features_bf16: features,
            audio_slices: vec![(0, rows)],
        };

        let mixed_end = tokenizer.token_to_id("<|mixed_end|>").map(u64::from);
        let endoftext = tokenizer.token_to_id("<|endoftext|>").map(u64::from);
        let im_start = tokenizer.token_to_id("<|im_start|>").map(u64::from);

        let mut banned = vec![
            audio_token_id,
            self.config.audio_out_token_id,
            self.config.audio_start_token_id,
            self.config.mixed_start_token_id,
        ];
        if let Some(id) = mixed_end {
            banned.push(id);
        }
        if !is_final
            && let Some(id) = endoftext {
                banned.push(id);
            }
        let mask = sampling_mask(model.vocab_size(), tokenizer.get_vocab_size(true), &banned);

        let mut stop_ids: HashSet<u64> = self.config.eos_token_ids.iter().copied().collect();
        stop_ids.insert(self.config.text_end_token_id);
        if let Some(id) = endoftext {
            stop_ids.insert(id);
        }
        if let Some(id) = im_start {
            stop_ids.insert(id);
        }

        let first = self.prefill(model, state, &prepared, Some(&mut sampler), &mask)?;
        let mut sampled = first.sampled;
        let mut tokens = Vec::new();
        for step in 0..max_new_tokens {
            let Some(token) = sampled else { break };
            if stop_ids.contains(&token) {
                break;
            }
            tokens.push(token);
            if step + 1 == max_new_tokens {
                break;
            }
            let step_out = self.forward_step(
                model,
                state,
                &StepInput::Token(token),
                Some(&mut sampler),
                &mask,
            )?;
            sampled = step_out.sampled;
        }
        // Clear `prepared` unused mut warning if any
        let _ = &mut prepared;
        Ok(tokens)
    }
}

fn write_frame_embedding(
    codebook: &[u16],
    codebook_size: usize,
    model_dim: usize,
    codes: &[i64],
    dst: &mut [bf16],
) {
    dst.fill(bf16::ZERO);
    for (slot, &code) in codes.iter().enumerate() {
        let row_base = (slot * codebook_size + code as usize) * model_dim;
        for d in 0..model_dim {
            let a = dst[d].to_f32();
            let b = bf16::from_bits(codebook[row_base + d]).to_f32();
            dst[d] = bf16::from_f32(a + b);
        }
    }
}

fn read_sampled_token(
    context: &<Metal as Backend>::Context,
    sampled_buf: Option<&<Metal as Backend>::GlobalGpuBuffer>,
    sampler: Option<&mut VlmSampler>,
) -> Result<Option<u64>, anyhow::Error> {
    let (Some(buf), Some(sampler)) = (sampled_buf, sampler) else {
        return Ok(None);
    };
    let ids: Vec<u32> = block_on(context.copyout(buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let token = ids.first().copied().unwrap_or(0) as u64;
    sampler.accept_token(token);
    Ok(Some(token))
}

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Result<PreparedAudioPrompt, anyhow::Error> {
    let encoding = tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(PreparedAudioPrompt {
        prompt_tokens: encoding.get_ids().iter().map(|&t| t as u64).collect(),
        features_bf16: Vec::new(),
        audio_slices: Vec::new(),
    })
}
