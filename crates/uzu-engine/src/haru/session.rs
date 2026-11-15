use std::{
    collections::VecDeque,
    path::Path,
    sync::Arc,
    time::Instant,
};

use anyhow::{Result, anyhow};
use futures::executor::block_on;
use half::bf16;
use tokenizers::Tokenizer;

use super::{
    config::{HaruConfig, HaruDryDefaults},
    dry::DryState,
    linear::{BiasKind, HaruLinear, MetalGpuBuffer, alloc_zero_bf16, upload_f32, upload_u32},
    meanflow_head::MeanFlowHead,
    vibevoice_codec::{CodecDecoderState, CodecEncoderState, VibeVoiceCodec},
};
use crate::{
    backends::{
        common::{
            Backend, Buffer as _, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, kernel::matmul::MatmulKernel,
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            kernel::{
                HaruLatentAffineMetalKernel, HaruPatchLogitsAndDryMetalKernel, HaruSampleUserLatentMetalKernel,
                HaruStreamEmbedSumMetalKernel, VibeVoiceRmsNormMetalKernel, matmul::MatmulMetalKernel,
            },
        },
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{PRng, SamplingMethod},
        transformer::TransformerState,
    },
    engine::{Engine, language_model::LanguageModel},
    safetensors::{ParameterLoader, ParameterTree},
    vision::common::vocab_padding_bitmask,
};

/// Simple SplitMix64 PRNG with Box-Muller Gaussian sampler (zero external crates).
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self {
            state: s,
        }
    }

    pub fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEAD_BEEF_CAFE_BABE);
        Self::new(nanos)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform_pos(&mut self) -> f32 {
        // Map top 24 bits to (0, 1].
        let v = ((self.next_u64() >> 40) as f32 + 0.5) / 16_777_216.0;
        v.clamp(1e-7, 1.0)
    }

    pub fn randn(&mut self) -> f32 {
        let u1 = self.next_uniform_pos();
        let u2 = (self.next_u64() >> 40) as f32 / 16_777_216.0;
        (-2.0f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn randn_vec(&mut self, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while i < len {
            let u1 = self.next_uniform_pos();
            let u2 = (self.next_u64() >> 40) as f32 / 16_777_216.0;
            let r = (-2.0f32 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            out.push(r * theta.cos());
            if i + 1 < len {
                out.push(r * theta.sin());
            }
            i += 2;
        }
        out
    }
}

pub struct AudioProjector {
    fc1: HaruLinear,
    norm_w: MetalGpuBuffer,
    fc2: HaruLinear,
}

impl AudioProjector {
    async fn load(
        tree: &ParameterTree<'_, Metal>,
        matmul: &MatmulMetalKernel,
        context: &MetalContext,
        latent_dim: u32,
        hidden_size: u32,
    ) -> Result<Self> {
        let fc1 =
            HaruLinear::load_from_tree(&tree.subtree("fc1"), matmul, context, hidden_size, latent_dim, BiasKind::Bf16)
                .await?;
        let norm_w = tree
            .subtree("norm")
            .read_buffer("weight", &[hidden_size], DataType::F32)
            .await
            .map_err(|e| anyhow!("{e:?}"))?;
        let fc2 = HaruLinear::load_from_tree(
            &tree.subtree("fc2"),
            matmul,
            context,
            hidden_size,
            hidden_size,
            BiasKind::Bf16,
        )
        .await?;
        Ok(Self {
            fc1,
            norm_w,
            fc2,
        })
    }

    pub fn encode_bf16(
        &self,
        scaled_bf16: impl BufferRef<Backend = Metal>,
        seq_len: u32,
        hidden_size: u32,
        rms_norm: &VibeVoiceRmsNormMetalKernel,
        matmul: &mut MatmulMetalKernel,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let mut h = cb
            .allocate_scratch_for_shape(&[seq_len, hidden_size], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        self.fc1.encode_into(matmul, None, scaled_bf16, seq_len, &mut h, cb)?;

        let mut normed = cb
            .allocate_scratch_for_shape(&[seq_len, hidden_size], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        rms_norm.encode(&h, &self.norm_w, &mut normed, seq_len, hidden_size, 1e-6f32, cb);

        let mut out = cb
            .allocate_scratch_for_shape(&[seq_len, hidden_size], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        self.fc2.encode_into(matmul, None, &normed, seq_len, &mut out, cb)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        latent_f32: impl BufferRef<Backend = Metal>,
        seq_len: u32,
        latent_dim: u32,
        hidden_size: u32,
        scale_buf: &MetalGpuBuffer,
        bias_buf: &MetalGpuBuffer,
        latent_affine: &HaruLatentAffineMetalKernel,
        rms_norm: &VibeVoiceRmsNormMetalKernel,
        matmul: &mut MatmulMetalKernel,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let mut scaled = cb
            .allocate_scratch_for_shape(&[seq_len, latent_dim], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        latent_affine.encode(latent_f32, scale_buf, bias_buf, &mut scaled, seq_len * latent_dim, cb);
        self.encode_bf16(&scaled, seq_len, hidden_size, rms_norm, matmul, cb)
    }
}

pub struct HaruPrompt {
    pub tokens: Vec<u32>,
    pub model_latents: Vec<f32>,
    pub user_latents: Vec<f32>,
    pub model_mask: Vec<u32>,
    pub user_mask: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct HaruSessionConfig {
    pub guidance_scale: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub eop_bias: f32,
    pub speak_first: bool,
    pub max_frames: usize,
    pub sink_window: Option<u32>,
    pub dry: Option<HaruDryDefaults>,
    pub seed: Option<u64>,
}

impl HaruSessionConfig {
    pub fn from_config(cfg: &HaruConfig) -> Self {
        Self {
            guidance_scale: cfg.defaults.guidance_scale,
            temperature: cfg.defaults.temperature,
            top_k: cfg.defaults.top_k,
            top_p: cfg.defaults.top_p,
            eop_bias: cfg.defaults.eop_bias,
            speak_first: true,
            max_frames: 750,
            sink_window: None,
            dry: Some(cfg.defaults.dry.clone()),
            seed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HaruFrame {
    pub audio: Vec<f32>,
    pub token: u32,
    pub text: String,
    pub index: usize,
    pub seconds: f64,
    pub done: bool,
}

struct OwnedTokenDecoder {
    tokenizer: Arc<Tokenizer>,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    exempt: [u32; 3],
}

impl OwnedTokenDecoder {
    fn new(tokenizer: Arc<Tokenizer>, pad_id: u32, eop_id: u32, eos_id: u32) -> Self {
        Self {
            tokenizer,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            exempt: [pad_id, eop_id, eos_id],
        }
    }

    fn push(&mut self, token_id: u32) -> Result<String> {
        if self.exempt.contains(&token_id) {
            return Ok(String::new());
        }
        self.decode_ids.push(token_id);
        let full = self.tokenizer.decode(&self.decode_ids, true).map_err(|e| anyhow!("{e}"))?;
        if full.len() > self.decode_prefix.len() && full.starts_with(&self.decode_prefix) {
            let delta = &full[self.decode_prefix.len()..];
            let clean = delta.trim_end_matches('\u{FFFD}');
            if !clean.is_empty() {
                let out = clean.to_string();
                let new_prefix_len = self.decode_prefix.len() + clean.len();
                self.decode_prefix = full[..new_prefix_len].to_string();
                return Ok(out);
            }
        }
        Ok(String::new())
    }
}

pub struct HaruPipeline {
    pub config: HaruConfig,
    pub model: LanguageModel<Metal>,
    pub pad_row_fp32: MetalGpuBuffer,
    pub eop_row_fp32: MetalGpuBuffer,
    pub latent_scale_val: f32,
    pub latent_bias_val: f32,
    pub proj_scale_buf: MetalGpuBuffer,
    pub proj_bias_buf: MetalGpuBuffer,
    pub unscale_scale_buf: MetalGpuBuffer,
    pub unscale_bias_buf: MetalGpuBuffer,
    pub model_audio_proj: AudioProjector,
    pub user_audio_proj: AudioProjector,
    pub head: MeanFlowHead,
    pub codec: VibeVoiceCodec,
    pub matmul: MatmulMetalKernel,
    pub latent_affine: HaruLatentAffineMetalKernel,
    pub sample_user_latent: HaruSampleUserLatentMetalKernel,
    pub stream_embed_sum: HaruStreamEmbedSumMetalKernel,
    pub patch_logits_dry: HaruPatchLogitsAndDryMetalKernel,
    pub rms_norm: VibeVoiceRmsNormMetalKernel,
    pub vocab_bitmask: Option<MetalGpuBuffer>,
    ones_mask_1: MetalGpuBuffer,
    dummy_u32: MetalGpuBuffer,
    dummy_f32: MetalGpuBuffer,
}

impl HaruPipeline {
    pub async fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = HaruConfig::from_dir(model_dir)?;
        let engine = Engine::<Metal>::new().await.map_err(|e| anyhow!("{e:?}"))?;
        let model = engine.load_language_model(model_dir).await.map_err(|e| anyhow!("{e:?}"))?;
        let context = model.context();

        let sidecar_path = model_dir.join("haru_sidecar.safetensors");
        let loader = ParameterLoader::new(&sidecar_path, context).await.map_err(|e| anyhow!("{e:?}"))?;
        let root = loader.tree();

        let pad_row_fp32 =
            root.read_buffer("pad_row_fp32", &[config.hidden_size], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let eop_row_fp32 =
            root.read_buffer("eop_row_fp32", &[config.hidden_size], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
        let latent_scale_slice = root.read_slice::<f32>("latent_scale", &[1]).await.map_err(|e| anyhow!("{e:?}"))?;
        let latent_bias_slice = root.read_slice::<f32>("latent_bias", &[1]).await.map_err(|e| anyhow!("{e:?}"))?;
        let latent_scale_val = latent_scale_slice[0];
        let latent_bias_val = latent_bias_slice[0];

        // scale_latents(raw) = (raw + b) * s = raw * s + (b * s)
        let proj_scale_buf = upload_f32(context, &[latent_scale_val])?;
        let proj_bias_buf = upload_f32(context, &[latent_bias_val * latent_scale_val])?;
        // unscale_latents(norm) = norm / s - b = norm * (1/s) + (-b)
        let unscale_scale_buf = upload_f32(context, &[1.0f32 / latent_scale_val])?;
        let unscale_bias_buf = upload_f32(context, &[-latent_bias_val])?;

        let matmul = MatmulMetalKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        let model_audio_proj = AudioProjector::load(
            &root.subtree("model_audio_proj"),
            &matmul,
            context,
            config.latent_dim,
            config.hidden_size,
        )
        .await?;
        let user_audio_proj = AudioProjector::load(
            &root.subtree("user_audio_proj"),
            &matmul,
            context,
            config.latent_dim,
            config.hidden_size,
        )
        .await?;

        let head = MeanFlowHead::load(&root.subtree("head"), &config.head, context).await?;
        let codec = VibeVoiceCodec::load(&root.subtree("codec"), &config.codec, config.frame_samples, context).await?;

        let tok_vocab = model.tokenizer().get_vocab_size(true);
        let model_vocab = model.vocab_size();
        let vocab_bitmask = if model_vocab > tok_vocab {
            let mask_words = vocab_padding_bitmask(model_vocab, tok_vocab);
            Some(upload_u32(context, &mask_words)?)
        } else {
            None
        };

        let latent_affine = HaruLatentAffineMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?;
        let sample_user_latent = HaruSampleUserLatentMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?;
        let stream_embed_sum = HaruStreamEmbedSumMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?;
        let patch_logits_dry =
            HaruPatchLogitsAndDryMetalKernel::new(context, DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        let rms_norm = VibeVoiceRmsNormMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?;
        let ones_mask_1 = upload_u32(context, &[1u32])?;
        let dummy_u32 = upload_u32(context, &[0u32])?;
        let dummy_f32 = upload_f32(context, &[0.0f32])?;

        Ok(Self {
            config,
            model,
            pad_row_fp32,
            eop_row_fp32,
            latent_scale_val,
            latent_bias_val,
            proj_scale_buf,
            proj_bias_buf,
            unscale_scale_buf,
            unscale_bias_buf,
            model_audio_proj,
            user_audio_proj,
            head,
            codec,
            matmul,
            latent_affine,
            sample_user_latent,
            stream_embed_sum,
            patch_logits_dry,
            rms_norm,
            vocab_bitmask,
            ones_mask_1,
            dummy_u32,
            dummy_f32,
        })
    }

    pub fn context(&self) -> &MetalContext {
        self.model.context()
    }

    fn system_prompt_ids(&self, system_prompt: &str) -> Result<Vec<u32>> {
        if system_prompt.is_empty() {
            return Ok(Vec::new());
        }
        let tok = self.model.tokenizer();
        let im_start = tok.token_to_id("<|im_start|>").ok_or_else(|| anyhow!("missing <|im_start|>"))?;
        let im_end = tok.token_to_id("<|im_end|>").ok_or_else(|| anyhow!("missing <|im_end|>"))?;
        let body = tok.encode(format!("system\n{system_prompt}"), false).map_err(|e| anyhow!("{e}"))?;
        let nl = tok.encode("\n", false).map_err(|e| anyhow!("{e}"))?;
        let mut out = Vec::with_capacity(2 + body.get_ids().len() + nl.get_ids().len());
        out.push(im_start);
        out.extend_from_slice(body.get_ids());
        out.push(im_end);
        out.extend_from_slice(nl.get_ids());
        Ok(out)
    }

    pub fn build_prompt(
        &mut self,
        system_prompt: &str,
        reference: Option<&[f32]>,
        speak_first: bool,
        rng: &mut SimpleRng,
    ) -> Result<HaruPrompt> {
        let context = self.model.context();
        let frame_samples = self.config.frame_samples as usize;
        let dim = self.config.latent_dim as usize;
        let pad = self.config.pad_token_id;
        let eop = self.config.eop_token_id;

        let card = self.system_prompt_ids(system_prompt)?;
        let n_card = card.len();

        let (ref_latents, n_ref) = if let Some(audio) = reference.filter(|a| !a.is_empty()) {
            let n_frames = audio.len().div_ceil(frame_samples);
            let mut padded = vec![0.0f32; n_frames * frame_samples];
            padded[..audio.len()].copy_from_slice(audio);
            let mut enc_st = self.codec.create_encoder_state(context)?;
            let mut means = Vec::with_capacity(n_frames * dim);
            for chunk in padded.chunks(16 * frame_samples) {
                let chunk_means = self.codec.encode_f32(chunk, &mut enc_st, context)?;
                means.extend_from_slice(&chunk_means);
            }
            let scale = rng.randn();
            let noise = rng.randn_vec(means.len());
            let sampled = self.codec.sample_latents(&means, scale, &noise);
            (Some(sampled), n_frames)
        } else {
            (None, 0)
        };

        let mut grid: Vec<u32> = if speak_first { vec![pad, eop] } else { Vec::new() };
        if card.is_empty() && n_ref == 0 && grid.is_empty() {
            grid.push(pad);
        }
        let n_grid = grid.len();

        let mut tokens = Vec::with_capacity(n_card + n_ref + n_grid);
        tokens.extend_from_slice(&card);
        if n_ref > 0 {
            tokens.extend(std::iter::repeat_n(pad, n_ref - 1));
            tokens.push(eop);
        }
        tokens.extend_from_slice(&grid);
        let n = tokens.len();

        let room_tone: &[f32] = if let (Some(audio), Some(rl)) = (reference, ref_latents.as_ref()) {
            let whole = audio.len() / frame_samples;
            if whole == 0 {
                self.codec.silence_latent()
            } else {
                let mut best_i = 0usize;
                let mut best_e = f64::INFINITY;
                for f in 0..whole {
                    let s = f * frame_samples;
                    let e: f64 = audio[s..s + frame_samples].iter().map(|&v| (v as f64) * (v as f64)).sum();
                    if e < best_e {
                        best_e = e;
                        best_i = f;
                    }
                }
                &rl[best_i * dim..(best_i + 1) * dim]
            }
        } else {
            self.codec.silence_latent()
        };
        let silence = self.codec.silence_latent();

        let mut model_latents = vec![0.0f32; n * dim];
        let mut user_latents = vec![0.0f32; n * dim];
        let mut model_mask = vec![0u32; n];
        let mut user_mask = vec![0u32; n];

        if let Some(rl) = ref_latents.as_ref() {
            model_latents[n_card * dim..(n_card + n_ref) * dim].copy_from_slice(rl);
        }
        for pos in 0..n_grid {
            let idx = (n - n_grid + pos) * dim;
            model_latents[idx..idx + dim].copy_from_slice(room_tone);
            user_latents[idx..idx + dim].copy_from_slice(silence);
        }
        for pos in n_card..n {
            model_mask[pos] = 1;
        }
        for pos in (n - n_grid)..n {
            user_mask[pos] = 1;
        }

        Ok(HaruPrompt {
            tokens,
            model_latents,
            user_latents,
            model_mask,
            user_mask,
        })
    }

    /// Embedded token + scaled model/user audio projection sum into `[seq_len, hidden_size]` BF16 ScratchBuffer.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_sequence_into_cb<M: BufferRef<Backend = Metal>>(
        &mut self,
        tokens: impl BufferRef<Backend = Metal>,
        model_latents_f32: impl BufferRef<Backend = Metal>,
        user_latents_f32: impl BufferRef<Backend = Metal>,
        masks_u32: Option<(M, M)>,
        seq_len: u32,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer> {
        let dim = self.config.latent_dim;
        let hidden = self.config.hidden_size;
        let tok_emb = self
            .model
            .decoder()
            .embedding()
            .encode_lookup(tokens, seq_len, cb)
            .map_err(|e| anyhow!("{e:?}"))?;
        let model_emb = self.model_audio_proj.encode(
            model_latents_f32,
            seq_len,
            dim,
            hidden,
            &self.proj_scale_buf,
            &self.proj_bias_buf,
            &self.latent_affine,
            &self.rms_norm,
            &mut self.matmul,
            cb,
        )?;
        let user_emb = self.user_audio_proj.encode(
            user_latents_f32,
            seq_len,
            dim,
            hidden,
            &self.proj_scale_buf,
            &self.proj_bias_buf,
            &self.latent_affine,
            &self.rms_norm,
            &mut self.matmul,
            cb,
        )?;
        let mut out = cb
            .allocate_scratch_for_shape(&[seq_len, hidden], DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;
        match masks_u32 {
            Some((model_mask, user_mask)) => {
                self.stream_embed_sum
                    .encode(&tok_emb, &model_emb, &user_emb, model_mask, user_mask, &mut out, hidden, seq_len, cb);
            },
            None => {
                self.stream_embed_sum.encode(
                    &tok_emb,
                    &model_emb,
                    &user_emb,
                    &self.ones_mask_1,
                    &self.ones_mask_1,
                    &mut out,
                    hidden,
                    seq_len,
                    cb,
                );
            },
        }
        Ok(out)
    }

    pub fn run_audio_proj_f32(&mut self, use_user: bool, raw_latent_f32: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.latent_dim;
        let hidden = self.config.hidden_size;
        if raw_latent_f32.is_empty() || !raw_latent_f32.len().is_multiple_of(dim as usize) {
            return Err(anyhow!("invalid latent length {}", raw_latent_f32.len()));
        }
        let seq_len = (raw_latent_f32.len() / dim as usize) as u32;
        let context = self.model.context();
        let in_gpu = upload_f32(context, raw_latent_f32)?;
        let mut out_gpu = alloc_zero_bf16(context, (seq_len * hidden) as usize)?;
        let mut cb = context.create_command_buffer(Some("haru_audio_proj"), None).map_err(|e| anyhow!("{e:?}"))?;
        let proj = if use_user { &self.user_audio_proj } else { &self.model_audio_proj };
        let out_scratch = proj.encode(
            &in_gpu,
            seq_len,
            dim,
            hidden,
            &self.proj_scale_buf,
            &self.proj_bias_buf,
            &self.latent_affine,
            &self.rms_norm,
            &mut self.matmul,
            &mut cb,
        )?;
        cb.encode_copy(&out_scratch, &mut out_gpu);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
        let out_bf16: Vec<bf16> = block_on(context.copyout(&out_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        Ok(out_bf16.into_iter().map(bf16::to_f32).collect())
    }

    pub fn run_head_f32(
        &mut self,
        cond_f32: &[f32],
        noise_f32: &[f32],
        guidance_scale: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let context = self.model.context();
        self.head.set_guidance_scale(guidance_scale, context)?;
        let cond_bf16: Vec<bf16> = cond_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let cond_gpu = super::linear::upload_slice(context, &cond_bf16)?;
        let noise_gpu = upload_f32(context, noise_f32)?;
        let mut out_norm_gpu = upload_f32(context, &vec![0.0f32; self.config.latent_dim as usize])?;
        let mut cb = context.create_command_buffer(Some("haru_head"), None).map_err(|e| anyhow!("{e:?}"))?;
        self.head.encode(&cond_gpu, &noise_gpu, &mut out_norm_gpu, &mut cb)?;
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
        let norm_f32: Vec<f32> = block_on(context.copyout(&out_norm_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        let raw_f32: Vec<f32> =
            norm_f32.iter().map(|&v| v / self.latent_scale_val - self.latent_bias_val).collect();
        Ok((norm_f32, raw_f32))
    }

    pub fn codec_encode_f32(&mut self, pcm_f32: &[f32], state: &mut CodecEncoderState) -> Result<Vec<f32>> {
        self.codec.encode_f32(pcm_f32, state, self.model.context())
    }

    pub fn codec_decode_f32(&mut self, latents_f32: &[f32], state: &mut CodecDecoderState) -> Result<Vec<f32>> {
        self.codec.decode_f32(latents_f32, state, self.model.context())
    }

    pub fn bench_breakdown(&mut self, iterations: usize) -> Result<()> {
        let lat_dim = self.config.latent_dim as usize;
        let dim = self.config.hidden_size as usize;
        let zeros_lat = vec![0.0f32; lat_dim];

        // 1. Codec decode
        let mut dec_state = self.codec.create_decoder_state(self.model.context())?;
        for _ in 0..2 {
            let _ = self.codec.decode_f32(&zeros_lat, &mut dec_state, self.model.context())?;
        }
        let mut dec_ms = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            let _ = self.codec.decode_f32(&zeros_lat, &mut dec_state, self.model.context())?;
            dec_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // 2. MeanFlowHead
        let cond_buf = alloc_zero_bf16(self.model.context(), dim)?;
        let noise_buf = upload_f32(self.model.context(), &zeros_lat)?;
        let mut out_lat_buf = upload_f32(self.model.context(), &zeros_lat)?;
        let mut head_ms = Vec::with_capacity(iterations);
        for i in 0..(iterations + 2) {
            let t0 = Instant::now();
            let mut cb = self.model.context().create_command_buffer(Some("bench_head"), None).map_err(|e| anyhow!("{e:?}"))?;
            self.head.encode(&cond_buf, &noise_buf, &mut out_lat_buf, &mut cb)?;
            block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
            if i >= 2 {
                head_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }

        // 3. LM backbone only (compute_logits=false) vs LM + readout (compute_logits=true)
        let mut t_state = self
            .model
            .decoder()
            .create_empty_state(self.model.recommended_context_length(), self.model.context())
            .map_err(|e| anyhow!("{e:?}"))?;
        let mut lm_only_ms = Vec::with_capacity(iterations);
        let mut lm_logits_ms = Vec::with_capacity(iterations);
        for i in 0..(iterations + 2) {
            let pos = t_state.context_length;
            t_state.prepare(pos, 1).map_err(|e| anyhow!("{e:?}"))?;
            let t0 = Instant::now();
            let mut cb = self.model.context().create_command_buffer(Some("bench_lm_only"), None).map_err(|e| anyhow!("{e:?}"))?;
            let emb = {
                let tok_b = cb.allocate_constant_from_slice(&[self.config.pad_token_id]).map_err(|e| anyhow!("{e:?}"))?;
                let u_lat_b = cb.allocate_constant_from_slice(&zeros_lat).map_err(|e| anyhow!("{e:?}"))?;
                self.embed_sequence_into_cb(
                    &tok_b,
                    &noise_buf,
                    &u_lat_b,
                    None::<(&MetalGpuBuffer, &MetalGpuBuffer)>,
                    1,
                    &mut cb,
                )?
            };
            let batch_dim = BatchTopology::flat(1, true);
            let _ = self
                .model
                .decoder()
                .encode_precomputed_embeddings_with_logits(
                    emb,
                    None,
                    0,
                    &batch_dim,
                    Some((0u32..1u32).into()),
                    Some(&[]),
                    false,
                    &mut t_state,
                    &mut cb,
                )
                .map_err(|e| anyhow!("{e:?}"))?;
            t_state.encode_accept(&[0], &mut cb).map_err(|e| anyhow!("{e:?}"))?;
            block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
            if i >= 2 {
                lm_only_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for i in 0..(iterations + 2) {
            let pos = t_state.context_length;
            t_state.prepare(pos, 1).map_err(|e| anyhow!("{e:?}"))?;
            let t0 = Instant::now();
            let mut cb = self.model.context().create_command_buffer(Some("bench_lm_logits"), None).map_err(|e| anyhow!("{e:?}"))?;
            let emb = {
                let tok_b = cb.allocate_constant_from_slice(&[self.config.pad_token_id]).map_err(|e| anyhow!("{e:?}"))?;
                let u_lat_b = cb.allocate_constant_from_slice(&zeros_lat).map_err(|e| anyhow!("{e:?}"))?;
                self.embed_sequence_into_cb(
                    &tok_b,
                    &noise_buf,
                    &u_lat_b,
                    None::<(&MetalGpuBuffer, &MetalGpuBuffer)>,
                    1,
                    &mut cb,
                )?
            };
            let batch_dim = BatchTopology::flat(1, true);
            let _ = self
                .model
                .decoder()
                .encode_precomputed_embeddings_with_logits(
                    emb,
                    None,
                    0,
                    &batch_dim,
                    Some((0u32..1u32).into()),
                    Some(&[]),
                    true,
                    &mut t_state,
                    &mut cb,
                )
                .map_err(|e| anyhow!("{e:?}"))?;
            t_state.encode_accept(&[0], &mut cb).map_err(|e| anyhow!("{e:?}"))?;
            block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
            if i >= 2 {
                lm_logits_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }

        let med = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let d_p50 = med(&mut dec_ms);
        let h_p50 = med(&mut head_ms);
        let l_only_p50 = med(&mut lm_only_ms);
        let l_full_p50 = med(&mut lm_logits_ms);
        println!(
            "[bench] breakdown (p50): lm_32layers={:.2} ms | lm_head(readout)={:.2} ms | head={:.2} ms | codec.dec={:.2} ms | sum={:.2} ms",
            l_only_p50,
            (l_full_p50 - l_only_p50).max(0.0),
            h_p50,
            d_p50,
            l_full_p50 + h_p50 + d_p50
        );
        Ok(())
    }

    pub fn prefill_prompt_last_step(
        &mut self,
        state: &mut TransformerState<Metal>,
        prompt: &HaruPrompt,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, <Metal as Backend>::ScratchBuffer)> {
        let n = prompt.tokens.len();
        let dim = self.config.latent_dim as usize;
        if n == 0
            || prompt.model_latents.len() != n * dim
            || prompt.user_latents.len() != n * dim
            || prompt.model_mask.len() != n
            || prompt.user_mask.len() != n
        {
            return Err(anyhow!(
                "Invalid HaruPrompt lengths: tokens={n}, m_lat={}, u_lat={}, m_mask={}, u_mask={}",
                prompt.model_latents.len(),
                prompt.user_latents.len(),
                prompt.model_mask.len(),
                prompt.user_mask.len()
            ));
        }

        let chunk_size = 256usize;
        let mut final_hidden_opt = None;
        let mut logits_opt = None;
        let mut pos = 0usize;

        while pos < n {
            let end = (pos + chunk_size).min(n);
            let len = end - pos;
            let is_last = end == n;

            state.prepare(pos as u32, len as u32).map_err(|e| anyhow!("{e:?}"))?;
            let mut cb = self
                .model
                .context()
                .create_command_buffer(Some("haru_prefill"), None)
                .map_err(|e| anyhow!("{e:?}"))?;
            let emb = {
                let tok_b = cb.allocate_constant_from_slice(&prompt.tokens[pos..end]).map_err(|e| anyhow!("{e:?}"))?;
                let m_lat_b = cb
                    .allocate_constant_from_slice(&prompt.model_latents[pos * dim..end * dim])
                    .map_err(|e| anyhow!("{e:?}"))?;
                let u_lat_b = cb
                    .allocate_constant_from_slice(&prompt.user_latents[pos * dim..end * dim])
                    .map_err(|e| anyhow!("{e:?}"))?;
                let m_mask_b =
                    cb.allocate_constant_from_slice(&prompt.model_mask[pos..end]).map_err(|e| anyhow!("{e:?}"))?;
                let u_mask_b =
                    cb.allocate_constant_from_slice(&prompt.user_mask[pos..end]).map_err(|e| anyhow!("{e:?}"))?;

                self.embed_sequence_into_cb(
                    &tok_b,
                    &m_lat_b,
                    &u_lat_b,
                    Some((&m_mask_b, &u_mask_b)),
                    len as u32,
                    &mut cb,
                )?
            };
            let batch_dim = BatchTopology::flat(len as u32, true);
            let output_range = is_last.then(|| (len as u32 - 1..len as u32).into());
            let hidden_indices = is_last.then_some(&[][..]);
            let dec_out = self
                .model
                .decoder()
                .encode_precomputed_embeddings_with_logits(
                    emb,
                    None,
                    0,
                    &batch_dim,
                    output_range,
                    hidden_indices,
                    is_last,
                    state,
                    &mut cb,
                )
                .map_err(|e| anyhow!("{e:?}"))?;
            let accepted: Box<[u32]> = (0..len as u32).collect();
            state.encode_accept(&accepted, &mut cb).map_err(|e| anyhow!("{e:?}"))?;
            block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;

            if is_last {
                final_hidden_opt = dec_out.final_hidden;
                logits_opt = dec_out.logits;
            }
            pos = end;
        }

        let hidden = final_hidden_opt.ok_or_else(|| anyhow!("prefill missing final_hidden"))?;
        let logits = logits_opt.ok_or_else(|| anyhow!("prefill missing logits"))?;
        Ok((hidden, logits))
    }

    pub fn forward_prompt_goldens(
        &mut self,
        prompt: &HaruPrompt,
    ) -> Result<(Vec<f32>, f32, f32, u32, Vec<(u32, f32)>)> {
        let mut state = self
            .model
            .decoder()
            .create_empty_state(self.model.recommended_context_length(), self.model.context())
            .map_err(|e| anyhow!("{e:?}"))?;
        let (hidden, mut logits) = self.prefill_prompt_last_step(&mut state, prompt)?;
        let context = self.model.context();
        let model_vocab = self.model.vocab_size();
        let mut hidden_gpu = alloc_zero_bf16(context, hidden.size() / std::mem::size_of::<bf16>())?;
        let mut logits_gpu = alloc_zero_bf16(context, logits.size() / std::mem::size_of::<bf16>())?;
        let mut cb = context.create_command_buffer(Some("haru_golden_patch"), None).map_err(|e| anyhow!("{e:?}"))?;
        self.patch_logits_dry.encode(
            &hidden,
            &self.pad_row_fp32,
            &self.eop_row_fp32,
            &mut logits,
            &self.dummy_u32,
            &self.dummy_f32,
            self.config.hidden_size,
            self.config.pad_token_id,
            self.config.eop_token_id,
            self.config.eos_token_id,
            0.0f32,
            0,
            0,
            &mut cb,
        );
        cb.encode_copy(&hidden, &mut hidden_gpu);
        cb.encode_copy(&logits, &mut logits_gpu);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;

        let hidden_bf16: Vec<bf16> = block_on(context.copyout(&hidden_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        let h_dim = self.config.hidden_size as usize;
        let hidden_f32: Vec<f32> = hidden_bf16.into_iter().take(h_dim).map(|b| b.to_f32()).collect();

        let logits_bf16: Vec<bf16> = block_on(context.copyout(&logits_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        let v_dim = model_vocab.min(logits_bf16.len());
        let logits_f32: Vec<f32> = logits_bf16.into_iter().take(v_dim).map(|b| b.to_f32()).collect();

        let pad_logit = logits_f32.get(self.config.pad_token_id as usize).copied().unwrap_or(f32::NAN);
        let eop_logit = logits_f32.get(self.config.eop_token_id as usize).copied().unwrap_or(f32::NAN);

        let mut indexed: Vec<(u32, f32)> = logits_f32.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let argmax = indexed.first().map(|x| x.0).unwrap_or(0);
        let top5 = indexed.into_iter().take(5).collect();

        Ok((hidden_f32, pad_logit, eop_logit, argmax, top5))
    }

    pub fn start_session(
        &mut self,
        session_config: HaruSessionConfig,
        system_prompt: &str,
        reference: Option<&[f32]>,
    ) -> Result<(HaruDuplexSession, HaruFrame)> {
        let mut session = HaruDuplexSession::new(self, session_config, system_prompt, reference)?;
        let frame0 = session.start(self)?;
        Ok((session, frame0))
    }
}

pub struct HaruDuplexSession {
    pub config: HaruSessionConfig,
    pub system_prompt: String,
    pub reference: Option<Vec<f32>>,
    transformer_state: TransformerState<Metal>,
    enc_state: CodecEncoderState,
    dec_state: CodecDecoderState,
    dry: Option<DryState>,
    detokenizer: OwnedTokenDecoder,
    rng: SimpleRng,
    prng: PRng,
    force_text: u32,
    pending_pcm: VecDeque<f32>,
    ready_means: VecDeque<Vec<f32>>,
    last_token: u32,
    norm_latent_gpu: MetalGpuBuffer,
    pcm_out_gpu: MetalGpuBuffer,
    pub index: usize,
    pub prompt_length: usize,
}

impl HaruDuplexSession {
    pub fn new(
        pipeline: &HaruPipeline,
        config: HaruSessionConfig,
        system_prompt: impl Into<String>,
        reference: Option<&[f32]>,
    ) -> Result<Self> {
        let context = pipeline.model.context();
        let recommended = pipeline.model.recommended_context_length();
        let transformer_state =
            pipeline.model.decoder().create_empty_state(recommended, context).map_err(|e| anyhow!("{e:?}"))?;
        let enc_state = pipeline.codec.create_encoder_state(context)?;
        let dec_state = pipeline.codec.create_decoder_state(context)?;
        let dry = config.dry.as_ref().map(|d| {
            DryState::new(
                pipeline.config.pad_token_id,
                pipeline.config.eop_token_id,
                pipeline.config.eos_token_id,
                d,
            )
        });
        let seed = config.seed.unwrap_or_else(|| {
            let mut r = SimpleRng::from_entropy();
            r.next_u64()
        });
        let rng = SimpleRng::new(seed);
        let prng = PRng::new(seed ^ 0xA5A5_5A5A_1234_5678);
        let detokenizer = OwnedTokenDecoder::new(
            pipeline.model.tokenizer().clone(),
            pipeline.config.pad_token_id,
            pipeline.config.eop_token_id,
            pipeline.config.eos_token_id,
        );
        let lat_dim = pipeline.config.latent_dim as usize;
        let fs = pipeline.config.frame_samples as usize;
        Ok(Self {
            config,
            system_prompt: system_prompt.into(),
            reference: reference.map(|r| r.to_vec()),
            transformer_state,
            enc_state,
            dec_state,
            dry,
            detokenizer,
            rng,
            prng,
            force_text: 0,
            pending_pcm: VecDeque::with_capacity(fs * 4),
            ready_means: VecDeque::with_capacity(16),
            last_token: pipeline.config.pad_token_id,
            norm_latent_gpu: upload_f32(context, &vec![0.0f32; lat_dim])?,
            pcm_out_gpu: alloc_zero_bf16(context, fs)?,
            index: 0,
            prompt_length: 0,
        })
    }

    pub fn frames_ready(&self) -> usize {
        self.ready_means.len()
    }

    pub fn push_pcm(&mut self, user_audio: &[f32]) {
        self.pending_pcm.extend(user_audio.iter().copied());
    }

    pub fn pending_frames_ready(&self, frame_samples: usize) -> usize {
        self.ready_means.len() + (self.pending_pcm.len() / frame_samples)
    }

    pub fn start(&mut self, pipeline: &mut HaruPipeline) -> Result<HaruFrame> {
        let started = Instant::now();
        {
            let context = pipeline.model.context();
            pipeline.head.set_guidance_scale(self.config.guidance_scale, context)?;
            self.enc_state = pipeline.codec.create_encoder_state(context)?;
            self.dec_state = pipeline.codec.create_decoder_state(context)?;
            self.transformer_state = pipeline
                .model
                .decoder()
                .create_empty_state(pipeline.model.recommended_context_length(), context)
                .map_err(|e| anyhow!("{e:?}"))?;
        }
        self.pending_pcm.clear();
        self.ready_means.clear();
        if let Some(d) = self.dry.as_mut() {
            d.clear();
        }
        self.detokenizer = OwnedTokenDecoder::new(
            pipeline.model.tokenizer().clone(),
            pipeline.config.pad_token_id,
            pipeline.config.eop_token_id,
            pipeline.config.eos_token_id,
        );

        let prompt = pipeline.build_prompt(
            &self.system_prompt,
            self.reference.as_deref(),
            self.config.speak_first,
            &mut self.rng,
        )?;
        self.force_text = if self.config.speak_first { 1 } else { 0 };
        let n = prompt.tokens.len();
        let (hidden, mut logits) = pipeline.prefill_prompt_last_step(&mut self.transformer_state, &prompt)?;

        self.prompt_length = n;
        self.index = 0;

        let lat_dim = pipeline.config.latent_dim as usize;
        let head_noise = self.rng.randn_vec(lat_dim);
        let cb = pipeline
            .model
            .context()
            .create_command_buffer(Some("haru_emit0"), None)
            .map_err(|e| anyhow!("{e:?}"))?;
        let frame = self.emit_from_hidden(pipeline, hidden, &mut logits, &head_noise, cb, started)?;
        Ok(frame)
    }

    pub fn feed(&mut self, pipeline: &mut HaruPipeline, user_audio: &[f32]) -> Result<()> {
        self.pending_pcm.extend(user_audio.iter().copied());
        let fs = pipeline.config.frame_samples as usize;
        while self.pending_pcm.len() >= fs {
            let chunk: Vec<f32> = self.pending_pcm.drain(..fs).collect();
            let context = pipeline.model.context();
            let means = pipeline.codec.encode_f32(&chunk, &mut self.enc_state, context)?;
            self.ready_means.push_back(means);
        }
        Ok(())
    }

    pub fn step_pending(&mut self, pipeline: &mut HaruPipeline) -> Result<HaruFrame> {
        let fs = pipeline.config.frame_samples as usize;
        if self.ready_means.is_empty() && self.pending_pcm.len() >= fs {
            let chunk: Vec<f32> = self.pending_pcm.drain(..fs).collect();
            return self.step(pipeline, Some(&chunk));
        }
        self.step(pipeline, None)
    }

    pub fn step(&mut self, pipeline: &mut HaruPipeline, user_audio: Option<&[f32]>) -> Result<HaruFrame> {
        let started = Instant::now();
        let fs = pipeline.config.frame_samples as usize;
        let lat_dim = pipeline.config.latent_dim as usize;
        let dim = pipeline.config.latent_dim;
        let hidden_size = pipeline.config.hidden_size;

        let mut direct_audio: Option<Vec<f32>> = None;
        if let Some(audio) = user_audio {
            if audio.len() != fs {
                return Err(anyhow!("a user frame is {} samples, got {}", fs, audio.len()));
            }
            if self.ready_means.is_empty() && self.pending_pcm.is_empty() {
                direct_audio = Some(audio.to_vec());
            } else {
                self.feed(pipeline, audio)?;
            }
        }

        let scale = self.rng.randn();
        let unit_noise = self.rng.randn_vec(lat_dim);
        let head_noise = self.rng.randn_vec(lat_dim);

        if let Some(window_len) = self.config.sink_window
            && window_len > 0
        {
            let sink_len = self.prompt_length as u32;
            let stride = 64u32.min(window_len.max(1));
            if self.transformer_state.context_length >= sink_len + window_len + stride {
                let evict = self.transformer_state.context_length - (sink_len + window_len);
                let mut compact_cb = pipeline
                    .model
                    .context()
                    .create_command_buffer(Some("haru_sink_compact"), None)
                    .map_err(|e| anyhow!("{e:?}"))?;
                self.transformer_state.compact_sink_window(sink_len, evict, &mut compact_cb);
                block_on(compact_cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;
            }
        }

        let pos = self.transformer_state.context_length;
        self.transformer_state.prepare(pos, 1).map_err(|e| anyhow!("{e:?}"))?;
        let mut cb = pipeline
            .model
            .context()
            .create_command_buffer(Some("haru_step"), None)
            .map_err(|e| anyhow!("{e:?}"))?;

        let emb = if let Some(audio) = direct_audio {
            let mean_bf16 = pipeline.codec.encode_f32_into_cb(&audio, &mut self.enc_state, &mut cb)?;
            let unit_noise_c = cb.allocate_constant_from_slice(&unit_noise).map_err(|e| anyhow!("{e:?}"))?;
            let mut user_scaled_bf16 = cb
                .allocate_scratch_for_shape(&[1, dim], DataType::BF16)
                .map_err(|e| anyhow!("{e:?}"))?;
            let noise_scale = scale * (pipeline.codec.config.latent_std / 0.8);
            let proj_scale = pipeline.latent_scale_val;
            let proj_bias = pipeline.latent_bias_val * pipeline.latent_scale_val;
            pipeline.sample_user_latent.encode(
                &mean_bf16,
                &unit_noise_c,
                &mut user_scaled_bf16,
                noise_scale,
                proj_scale,
                proj_bias,
                dim,
                &mut cb,
            );
            drop(unit_noise_c);

            let tok_b = cb.allocate_constant_from_slice(&[self.last_token]).map_err(|e| anyhow!("{e:?}"))?;
            let tok_emb = pipeline
                .model
                .decoder()
                .embedding()
                .encode_lookup(&tok_b, 1, &mut cb)
                .map_err(|e| anyhow!("{e:?}"))?;
            drop(tok_b);

            let model_emb = pipeline.model_audio_proj.encode(
                &self.norm_latent_gpu,
                1,
                dim,
                hidden_size,
                &pipeline.head.scale_one,
                &pipeline.head.bias_zero,
                &pipeline.latent_affine,
                &pipeline.rms_norm,
                &mut pipeline.matmul,
                &mut cb,
            )?;
            let user_emb = pipeline.user_audio_proj.encode_bf16(
                &user_scaled_bf16,
                1,
                hidden_size,
                &pipeline.rms_norm,
                &mut pipeline.matmul,
                &mut cb,
            )?;
            let mut out = cb
                .allocate_scratch_for_shape(&[1, hidden_size], DataType::BF16)
                .map_err(|e| anyhow!("{e:?}"))?;
            pipeline.stream_embed_sum.encode(
                &tok_emb,
                &model_emb,
                &user_emb,
                &pipeline.ones_mask_1,
                &pipeline.ones_mask_1,
                &mut out,
                hidden_size,
                1,
                &mut cb,
            );
            out
        } else {
            let user_mean = self
                .ready_means
                .pop_front()
                .ok_or_else(|| anyhow!("no ready user frame (call feed() or pass user_audio)"))?;
            let user_latent = pipeline.codec.sample_latents(&user_mean, scale, &unit_noise);

            let tok_b = cb.allocate_constant_from_slice(&[self.last_token]).map_err(|e| anyhow!("{e:?}"))?;
            let u_lat_b = cb.allocate_constant_from_slice(&user_latent).map_err(|e| anyhow!("{e:?}"))?;
            let tok_emb = pipeline
                .model
                .decoder()
                .embedding()
                .encode_lookup(&tok_b, 1, &mut cb)
                .map_err(|e| anyhow!("{e:?}"))?;
            let model_emb = pipeline.model_audio_proj.encode(
                &self.norm_latent_gpu,
                1,
                dim,
                hidden_size,
                &pipeline.head.scale_one,
                &pipeline.head.bias_zero,
                &pipeline.latent_affine,
                &pipeline.rms_norm,
                &mut pipeline.matmul,
                &mut cb,
            )?;
            let user_emb = pipeline.user_audio_proj.encode(
                &u_lat_b,
                1,
                dim,
                hidden_size,
                &pipeline.proj_scale_buf,
                &pipeline.proj_bias_buf,
                &pipeline.latent_affine,
                &pipeline.rms_norm,
                &mut pipeline.matmul,
                &mut cb,
            )?;
            let mut out = cb
                .allocate_scratch_for_shape(&[1, hidden_size], DataType::BF16)
                .map_err(|e| anyhow!("{e:?}"))?;
            pipeline.stream_embed_sum.encode(
                &tok_emb,
                &model_emb,
                &user_emb,
                &pipeline.ones_mask_1,
                &pipeline.ones_mask_1,
                &mut out,
                hidden_size,
                1,
                &mut cb,
            );
            out
        };

        let batch_dim = BatchTopology::flat(1, true);
        let dec_out = pipeline
            .model
            .decoder()
            .encode_precomputed_embeddings_with_logits(
                emb,
                None,
                0,
                &batch_dim,
                Some((0u32..1u32).into()),
                Some(&[]),
                true,
                &mut self.transformer_state,
                &mut cb,
            )
            .map_err(|e| anyhow!("{e:?}"))?;
        self.transformer_state.encode_accept(&[0], &mut cb).map_err(|e| anyhow!("{e:?}"))?;

        let hidden = dec_out.final_hidden.ok_or_else(|| anyhow!("decode step missing final_hidden"))?;
        let mut logits = dec_out.logits.ok_or_else(|| anyhow!("decode step missing logits"))?;

        self.emit_from_hidden(pipeline, hidden, &mut logits, &head_noise, cb, started)
    }

    fn emit_from_hidden(
        &mut self,
        pipeline: &mut HaruPipeline,
        hidden: <Metal as Backend>::ScratchBuffer,
        logits: &mut <Metal as Backend>::ScratchBuffer,
        head_noise: &[f32],
        mut cb_owned: MetalCommandBufferEncoding,
        started: Instant,
    ) -> Result<HaruFrame> {
        let cb = &mut cb_owned;
        let dim = pipeline.config.hidden_size;
        let fs = pipeline.config.frame_samples as usize;

        let (dry_ids, dry_pens) = self.dry.as_ref().map(|d| d.compute_penalties()).unwrap_or_default();
        let dry_count = dry_ids.len() as u32;
        let force_text = if self.force_text > 0 {
            self.force_text -= 1;
            1u32
        } else {
            0u32
        };

        if dry_count > 0 {
            let ids_c = cb.allocate_constant_from_slice(&dry_ids).map_err(|e| anyhow!("{e:?}"))?;
            let pens_c = cb.allocate_constant_from_slice(&dry_pens).map_err(|e| anyhow!("{e:?}"))?;
            pipeline.patch_logits_dry.encode(
                &hidden,
                &pipeline.pad_row_fp32,
                &pipeline.eop_row_fp32,
                &mut *logits,
                &ids_c,
                &pens_c,
                dim,
                pipeline.config.pad_token_id,
                pipeline.config.eop_token_id,
                pipeline.config.eos_token_id,
                self.config.eop_bias,
                dry_count,
                force_text,
                cb,
            );
        } else {
            pipeline.patch_logits_dry.encode(
                &hidden,
                &pipeline.pad_row_fp32,
                &pipeline.eop_row_fp32,
                &mut *logits,
                &pipeline.dummy_u32,
                &pipeline.dummy_f32,
                dim,
                pipeline.config.pad_token_id,
                pipeline.config.eop_token_id,
                pipeline.config.eos_token_id,
                self.config.eop_bias,
                0,
                force_text,
                cb,
            );
        }

        let sampling_method = if self.config.temperature <= 0.0 {
            SamplingMethod::Greedy
        } else {
            SamplingMethod::Stochastic {
                temperature: Some(self.config.temperature.max(1e-5)),
                top_k: (self.config.top_k > 0).then_some(self.config.top_k as u32),
                top_p: (self.config.top_p < 1.0).then_some(self.config.top_p),
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            }
        };

        let seeds_buf = if matches!(sampling_method, SamplingMethod::Stochastic { .. }) {
            let s = self.prng.derive(self.index as u64);
            Some(cb.allocate_constant_from_slice(&[s]).map_err(|e| anyhow!("{e:?}"))?)
        } else {
            None
        };

        let batch_dim = BatchTopology::flat(1, true);
        let sampled_buf = pipeline
            .model
            .sampling()
            .encode(
                &*logits,
                seeds_buf.as_ref(),
                pipeline.vocab_bitmask.as_ref(),
                None::<&MetalGpuBuffer>,
                None::<&MetalGpuBuffer>,
                &batch_dim,
                (0u32..1u32).into(),
                &sampling_method,
                cb,
            )
            .map_err(|e| anyhow!("{e:?}"))?;

        let noise_c = cb.allocate_constant_from_slice(head_noise).map_err(|e| anyhow!("{e:?}"))?;
        pipeline.head.encode(&hidden, &noise_c, &mut self.norm_latent_gpu, cb)?;
        drop(noise_c);
        let pcm_scratch = pipeline.codec.decode_scaled_f32_into_cb(
            &self.norm_latent_gpu,
            1,
            &pipeline.unscale_scale_buf,
            &pipeline.unscale_bias_buf,
            &mut self.dec_state,
            cb,
        )?;
        cb.encode_copy(&pcm_scratch, &mut self.pcm_out_gpu);

        drop(seeds_buf);
        block_on(cb_owned.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow!("{e:?}"))?;

        let context = pipeline.model.context();
        let toks: Vec<u32> = block_on(context.copyout(&sampled_buf)).map_err(|e| anyhow!("{e:?}"))?;
        let token = toks[0];
        let pcm_bf16: Vec<bf16> = block_on(context.copyout(&self.pcm_out_gpu)).map_err(|e| anyhow!("{e:?}"))?;
        let audio: Vec<f32> = pcm_bf16.into_iter().take(fs).map(|b| b.to_f32().clamp(-1.0, 1.0)).collect();

        self.last_token = token;
        if let Some(d) = self.dry.as_mut() {
            d.push(token);
        }
        let text = self.detokenizer.push(token)?;
        let idx = self.index;
        self.index += 1;
        let done = self.index >= self.config.max_frames;

        Ok(HaruFrame {
            audio,
            token,
            text,
            index: idx,
            seconds: started.elapsed().as_secs_f64(),
            done,
        })
    }
}
