use std::{
    collections::{HashMap, VecDeque, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::Path,
    range::Range,
    sync::Arc,
};

use futures::executor::block_on;
use half::bf16;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{
    config::{Calibration, D1Config},
    prompt::{
        D1Instruction, D1Question, SlotSpec, SpecialTokens, StatePositions, box_confidence, box_medians,
        build_plain_tokenizer, build_prefix_ids, estimate_grid, pack_set_ids, pointer_span_from_logits,
        probabilities, question_header, question_slots, suffix_layout_parts,
    },
};
use crate::{
    audio::lfm3_audio::{Lfm3AudioConfig, features::MelFilterBank, features_metal::MetalAudioFeatures},
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
        metal::Metal,
    },
    config::{model::AnyModelConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder},
    safetensors::ParameterLoader,
    vision::{lfm3_pipeline::Lfm3VlPipeline, tower::SafetensorLoader},
};

#[derive(Debug, Clone)]
pub enum D1MediaInput {
    Image(image::DynamicImage),
    Video(Vec<image::DynamicImage>),
    Audio {
        waveform: Vec<f32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoreReadout {
    #[default]
    Median,
    Mean,
    Argmax,
}

#[derive(Debug, Clone, Default)]
pub struct D1Options {
    pub score_readout: ScoreReadout,
    pub set_size: Option<usize>,
    pub abstain_below: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D1Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D1Response {
    pub answers: Vec<(String, serde_json::Value)>,
    pub usage: D1Usage,
}

pub struct D1OmniPipeline {
    pub d1_config: D1Config,
    pub calibration: Calibration,
    pub tokenizer: Tokenizer,
    pub plain_tokenizer: Tokenizer,
    pub special_tokens: SpecialTokens,
    pub context: Arc<<Metal as Backend>::Context>,
    pub decoder: Decoder<Metal>,
    decoder_state: std::sync::Mutex<crate::encodable_block::transformer::TransformerState<Metal>>,
    pub vision: Lfm3VlPipeline,
    pub audio_config: Lfm3AudioConfig,
    pub mel: MelFilterBank,
    pub audio_features: std::sync::Mutex<MetalAudioFeatures>,
    pub audio_trunk: Decoder<Metal>,
    audio_trunk_state: std::sync::Mutex<crate::encodable_block::transformer::TransformerState<Metal>>,
    pub model_dim: usize,
    pub lead: String,
    pointer_matmul: parking_lot::Mutex<<<Metal as Backend>::Kernels as Kernels>::MatmulKernel>,
    media_cache: std::sync::Mutex<(
        VecDeque<u64>,
        HashMap<u64, (usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>)>,
    )>,
}

impl D1OmniPipeline {
    pub fn is_d1_omni(model_dir: &Path) -> bool {
        model_dir.join("d1_config.json").is_file()
            && model_dir.join("vision_config.json").is_file()
            && model_dir.join("audio_config.json").is_file()
    }

    pub async fn load_metal(model_dir: &Path) -> Result<Self, anyhow::Error> {
        let context = <<Metal as Backend>::Context as Context>::new().await?;
        Self::load_from_model_dir(model_dir, context).await
    }

    pub async fn load_from_model_dir(
        model_dir: &Path,
        context: Arc<<Metal as Backend>::Context>,
    ) -> Result<Self, anyhow::Error> {
        let d1_config = D1Config::load(&model_dir.join("d1_config.json"))?;
        let calibration = Calibration::load(&model_dir.join("calibration.json"))?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        let plain_tokenizer = build_plain_tokenizer(&tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
        let special_tokens = SpecialTokens::from_tokenizer(&tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;

        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let added = tokenizer.get_added_vocabulary();
        let added_vocab = added.get_vocab();
        let lead = if added_vocab.contains_key("<think>") && added_vocab.contains_key("</think>") {
            "<think></think>".to_string()
        } else {
            String::new()
        };
        let model_config: AnyModelConfig = serde_json::from_slice(&cfg_bytes)?;
        let mut decoder_config = match model_config {
            AnyModelConfig::LanguageModelConfig(lm) => lm.decoder_config,
            other => anyhow::bail!("expected LanguageModelConfig in config.json, got {other:?}"),
        };

        // For d1 Set Attention, full_attention layers (no sliding window) use non-causal
        // attention so state and per-question groups attend bidirectionally via BatchTopology::set_attention.
        if d1_config.d1_attention == "set" {
            for layer in decoder_config.transformer_config.layer_configs.iter_mut() {
                if let AnyTokenMixerConfig::AttentionConfig(attn) = &mut layer.mixer.body_config
                    && attn.sliding_window_size.is_none() {
                        attn.is_causal = false;
                    }
            }
        }

        let model_dim = decoder_config.transformer_config.model_dim as usize;
        let model_path = model_dir.join("model.safetensors");
        let loader = ParameterLoader::new(&model_path, context.as_ref()).await?;
        let root_tree = loader.tree();
        let decoder = Decoder::new(
            DataType::BF16,
            &decoder_config,
            &root_tree.subtree("decoder"),
            context.as_ref(),
        )
        .await?;

        let vision = Lfm3VlPipeline::load_from_model_dir(model_dir, &tokenizer)?;

        let audio_config = Lfm3AudioConfig::load(model_dir)?;
        let audio_path = model_dir.join("audio.safetensors");
        let mut audio_loader =
            SafetensorLoader::open(&audio_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mel = MelFilterBank::slaney(audio_config.sample_rate, audio_config.n_fft, audio_config.num_mel_bins);
        let audio_features = std::sync::Mutex::new(MetalAudioFeatures::load(
            &mut audio_loader,
            context.as_ref(),
            audio_config.num_mel_bins,
            audio_config.conv_channels,
            audio_config.encoder_hidden,
        )?);
        let audio_param_loader = ParameterLoader::new(&audio_path, context.as_ref()).await?;
        let audio_trunk = Decoder::new(
            DataType::BF16,
            &audio_config.encoder_decoder_config,
            &audio_param_loader.tree().subtree("audio").subtree("encoder"),
            context.as_ref(),
        )
        .await?;

        let decoder_state = std::sync::Mutex::new(decoder.create_empty_state(None, context.as_ref())?);
        let audio_trunk_state = std::sync::Mutex::new(audio_trunk.create_empty_state(None, context.as_ref())?);
        let pointer_matmul = parking_lot::Mutex::new(
            <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                context.as_ref(),
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?,
        );

        Ok(Self {
            d1_config,
            calibration,
            tokenizer,
            plain_tokenizer,
            special_tokens,
            context,
            decoder,
            decoder_state,
            vision,
            audio_config,
            mel,
            audio_features,
            audio_trunk,
            audio_trunk_state,
            model_dim,
            lead,
            pointer_matmul,
            media_cache: std::sync::Mutex::new((VecDeque::new(), HashMap::new())),
        })
    }

    fn encode_audio_clip_gpu(
        &self,
        waveform: &[f32],
    ) -> Result<(usize, Arc<<Metal as Backend>::GlobalGpuBuffer>), anyhow::Error> {
        let num_frames = self.audio_config.num_frames(waveform.len());
        let mel = self.mel.log_mel(
            waveform,
            self.audio_config.n_fft,
            self.audio_config.hop_length,
            num_frames,
            self.audio_config.log_floor,
            self.audio_config.norm_offset,
            self.audio_config.norm_divisor,
        );
        let mut cb = self
            .context
            .create_command_buffer(Some("d1_audio_encoder"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut feat_guard = self.audio_features.lock().map_err(|_| anyhow::anyhow!("audio_features lock poisoned"))?;
        let (downsampled_gpu, tokens, ds_scratch) = feat_guard.encode_downsampler(
            self.context.as_ref(),
            &mut cb,
            &mel,
            self.audio_config.num_mel_bins,
            num_frames,
        )?;
        anyhow::ensure!(tokens > 0, "audio clip too short for any encoder token");

        let mut trunk_state = self
            .audio_trunk_state
            .lock()
            .map_err(|e| anyhow::anyhow!("audio_trunk_state lock: {e}"))?;
        trunk_state.reset().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        trunk_state.prepare(0, tokens as u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let token_ids = vec![1u32; tokens];
        let token_buf = cb.allocate_constant_from_slice(&token_ids).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let batch_dim = BatchTopology::flat(tokens as u32, true);
        let out = self
            .audio_trunk
            .encode_multimodal_with_logits(
                &token_buf,
                Some(&downsampled_gpu),
                Some(&[(0usize, tokens)]),
                None,
                0,
                &batch_dim,
                Some(Range {
                    start: 0,
                    end: tokens as u32,
                }),
                Some(&[]),
                false,
                &mut trunk_state,
                &mut cb,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let trunk_hidden = out.final_hidden.ok_or_else(|| anyhow::anyhow!("audio trunk produced no hidden state"))?;
        let (projected_gpu, proj_scratch) =
            feat_guard.encode_projector(self.context.as_ref(), &mut cb, &trunk_hidden, tokens)?;
        drop(feat_guard);
        drop(token_buf);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        drop(ds_scratch);
        drop(downsampled_gpu);
        drop(proj_scratch);
        Ok((tokens, Arc::new(projected_gpu)))
    }

    fn upload_bf16_gpu(
        &self,
        data: &[bf16],
    ) -> Result<Arc<<Metal as Backend>::GlobalGpuBuffer>, anyhow::Error> {
        let byte_len = data.len() * 2;
        let mut gpu = self
            .context
            .create_buffer_gpu(byte_len)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        if byte_len > 0 {
            let mut cb = self
                .context
                .create_command_buffer(Some("d1_upload_media"), None)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let src = cb.allocate_constant_from_slice(data).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            cb.encode_copy(src.subrange(0..byte_len), &mut gpu);
            drop(src);
            block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }
        Ok(Arc::new(gpu))
    }

    fn push_media_tokens(
        &self,
        dst: &mut Vec<u32>,
        tok_id: u32,
        rows: usize,
    ) {
        dst.push(self.special_tokens.vision_start);
        dst.resize(dst.len() + rows, tok_id);
        dst.push(self.special_tokens.vision_end);
    }

    fn encode_media_item(
        &self,
        item: &D1MediaInput,
    ) -> Result<(usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>), anyhow::Error> {
        match item {
            D1MediaInput::Image(img) => {
                let encoded = self.vision.encode_image(img)?;
                let gpu = self.upload_bf16_gpu(&encoded.features)?;
                Ok((encoded.rows, self.d1_config.image_token_id, gpu))
            },
            D1MediaInput::Video(frames) => {
                let encoded = self.vision.encode_video(frames)?;
                let gpu = self.upload_bf16_gpu(&encoded.features)?;
                Ok((encoded.rows, self.d1_config.video_token_id, gpu))
            },
            D1MediaInput::Audio {
                waveform,
            } => {
                let (rows, gpu) = self.encode_audio_clip_gpu(waveform)?;
                Ok((rows, self.d1_config.audio_token_id, gpu))
            },
        }
    }

    fn media_cache_key(item: &D1MediaInput) -> u64 {
        let mut h = DefaultHasher::new();
        match item {
            D1MediaInput::Image(img) => {
                0u8.hash(&mut h);
                img.width().hash(&mut h);
                img.height().hash(&mut h);
                img.as_bytes().hash(&mut h);
            },
            D1MediaInput::Video(frames) => {
                1u8.hash(&mut h);
                frames.len().hash(&mut h);
                for f in frames {
                    f.width().hash(&mut h);
                    f.height().hash(&mut h);
                    f.as_bytes().hash(&mut h);
                }
            },
            D1MediaInput::Audio {
                waveform,
            } => {
                2u8.hash(&mut h);
                waveform.len().hash(&mut h);
                for &s in waveform {
                    s.to_bits().hash(&mut h);
                }
            },
        }
        h.finish()
    }

    fn audio_cache_key(waveform: &[f32]) -> u64 {
        let mut h = DefaultHasher::new();
        2u8.hash(&mut h);
        waveform.len().hash(&mut h);
        for &s in waveform {
            s.to_bits().hash(&mut h);
        }
        h.finish()
    }

    fn cache_get(
        &self,
        key: u64,
    ) -> Option<(usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>)> {
        self.media_cache.lock().ok().and_then(|g| g.1.get(&key).cloned())
    }

    fn cache_put(
        &self,
        key: u64,
        val: (usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>),
    ) {
        const MAX_MEDIA_CACHE_ENTRIES: usize = 64;
        if let Ok(mut guard) = self.media_cache.lock() {
            if !guard.1.contains_key(&key) {
                if guard.0.len() >= MAX_MEDIA_CACHE_ENTRIES
                    && let Some(old) = guard.0.pop_front() {
                        guard.1.remove(&old);
                    }
                guard.0.push_back(key);
            }
            guard.1.insert(key, val);
        }
    }

    fn encode_media_item_cached(
        &self,
        item: &D1MediaInput,
    ) -> Result<(usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>), anyhow::Error> {
        let key = Self::media_cache_key(item);
        if let Some(hit) = self.cache_get(key) {
            return Ok(hit);
        }
        let encoded = self.encode_media_item(item)?;
        self.cache_put(key, encoded.clone());
        Ok(encoded)
    }

    fn encode_audio_cached(
        &self,
        waveform: &[f32],
    ) -> Result<(usize, u32, Arc<<Metal as Backend>::GlobalGpuBuffer>), anyhow::Error> {
        let key = Self::audio_cache_key(waveform);
        if let Some(hit) = self.cache_get(key) {
            return Ok(hit);
        }
        let (rows, gpu) = self.encode_audio_clip_gpu(waveform)?;
        let encoded = (rows, self.d1_config.audio_token_id, gpu);
        self.cache_put(key, encoded.clone());
        Ok(encoded)
    }

    /// Runs a single packed forward pass with Set Attention and returns normalized hidden states `[T, D]` as `f32`.
    pub fn d1_hidden(
        &self,
        ids: &[u32],
        state_len: usize,
        question_group: &[i32],
        media_features: &[(u32, &[bf16])],
    ) -> Result<Vec<f32>, anyhow::Error> {
        let seq_len = ids.len();
        anyhow::ensure!(seq_len > 0 && seq_len == question_group.len(), "ids/question_group mismatch");

        // Scatter multimodal features in placeholder order (matching d1_hidden in d1.py)
        let mut scattered_slices: Vec<(usize, usize)> = Vec::new();
        let mut flat_features: Vec<bf16> = Vec::new();
        let mut by_token: Vec<(u32, Vec<&[bf16]>)> = Vec::new();
        for &(tok_id, feats) in media_features {
            if let Some((_, list)) = by_token.iter_mut().find(|(t, _)| *t == tok_id) {
                list.push(feats);
            } else {
                by_token.push((tok_id, vec![feats]));
            }
        }
        for (tok_id, chunks) in by_token {
            let mut positions = Vec::new();
            for (idx, &id) in ids.iter().enumerate() {
                if id == tok_id {
                    positions.push(idx);
                }
            }
            let total_rows: usize = chunks.iter().map(|c| c.len() / self.model_dim).sum();
            anyhow::ensure!(
                positions.len() == total_rows,
                "placeholders {} ({}) != features ({})",
                tok_id,
                positions.len(),
                total_rows
            );
            let mut src_iter = chunks.into_iter().flat_map(|c| c.chunks_exact(self.model_dim));
            let mut i = 0;
            while i < positions.len() {
                let start = positions[i];
                let mut count = 1;
                while i + count < positions.len() && positions[i + count] == start + count {
                    count += 1;
                }
                for _ in 0..count {
                    flat_features.extend_from_slice(src_iter.next().unwrap());
                }
                scattered_slices.push((start, count));
                i += count;
            }
        }

        let mut cb = self
            .context
            .create_command_buffer(Some("d1_omni_forward"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let token_buf = cb.allocate_constant_from_slice(ids).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let feat_buf = if !flat_features.is_empty() {
            let bytes: Vec<u8> = flat_features.iter().flat_map(|v| v.to_le_bytes()).collect();
            Some(cb.allocate_constant_from_slice(&bytes).map_err(|e| anyhow::anyhow!("{e:?}"))?)
        } else {
            None
        };

        let mut state = self
            .decoder_state
            .lock()
            .map_err(|e| anyhow::anyhow!("decoder_state lock: {e}"))?;
        state.reset()?;
        state.prepare(0, seq_len as u32)?;
        let batch_dim = BatchTopology::set_attention(state_len as u32, question_group);
        let slices_ref = if scattered_slices.is_empty() {
            None
        } else {
            Some(scattered_slices.as_slice())
        };

        let out = self.decoder.encode_multimodal_with_logits(
            &token_buf,
            feat_buf.as_ref(),
            slices_ref,
            None,
            0,
            &batch_dim,
            Some(Range {
                start: 0,
                end: seq_len as u32,
            }),
            None,
            false,
            &mut state,
            &mut cb,
        )?;
        let hidden_buf = out.final_hidden.ok_or_else(|| anyhow::anyhow!("decoder produced no final_hidden"))?;
        let byte_len = seq_len * self.model_dim * 2;
        let mut out_global = self
            .context
            .create_buffer_gpu(byte_len)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        cb.encode_copy(hidden_buf.subrange(0..byte_len), &mut out_global);
        drop(token_buf);
        drop(feat_buf);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let raw_bytes: Vec<u8> = block_on(self.context.copyout(&out_global)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let hidden_f32: Vec<f32> = raw_bytes
            .as_chunks::<2>().0.iter()
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        Ok(hidden_f32)
    }

    /// Forward pass returning only the sorted `rows` (each of length `model_dim`),
    /// coalescing nearby rows into minimal GPU->CPU copies to avoid copying/converting
    /// Runs one decoder forward pass and fuses GPU-side slot slicing + readout matmul
    /// for `code_rows` × `readout_token_ids` and `read_slot_rows` × `state_hidden`
    /// (for `D1Question::Read` pointer spans) in the same CommandBuffer.
    /// Returns `(readout_scores_f32, pointer_logits_f32)` with zero hidden-vector copyout.
    fn d1_forward_and_readout(
        &self,
        ids: &[u32],
        state_len: usize,
        question_group: &[i32],
        media_features: &[(u32, usize, Arc<<Metal as Backend>::GlobalGpuBuffer>)],
        code_rows: &[usize],
        readout_token_ids: &[u32],
        read_slot_rows: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), anyhow::Error> {
        let seq_len = ids.len();
        anyhow::ensure!(seq_len > 0 && seq_len == question_group.len(), "ids/question_group mismatch");
        if code_rows.is_empty() && read_slot_rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let row_bytes = self.model_dim * 2;
        let mut scattered_slices: Vec<(usize, usize)> = Vec::new();
        let mut copy_ops: Vec<(Arc<<Metal as Backend>::GlobalGpuBuffer>, usize, usize, usize)> = Vec::new();
        let mut total_media_rows = 0usize;
        let mut by_token: Vec<(u32, Vec<(usize, Arc<<Metal as Backend>::GlobalGpuBuffer>)>)> = Vec::new();
        for (tok_id, rows, gpu) in media_features {
            if let Some((_, list)) = by_token.iter_mut().find(|(t, _)| *t == *tok_id) {
                list.push((*rows, Arc::clone(gpu)));
            } else {
                by_token.push((*tok_id, vec![(*rows, Arc::clone(gpu))]));
            }
        }
        for (tok_id, chunks) in by_token {
            let mut positions = Vec::new();
            for (idx, &id) in ids.iter().enumerate() {
                if id == tok_id {
                    positions.push(idx);
                }
            }
            let total_rows: usize = chunks.iter().map(|(r, _)| *r).sum();
            anyhow::ensure!(
                positions.len() == total_rows,
                "placeholders {} ({}) != features ({})",
                tok_id,
                positions.len(),
                total_rows
            );
            let mut pos_idx = 0usize;
            for (chunk_rows, gpu) in chunks {
                let mut remaining = chunk_rows;
                let mut src_row = 0usize;
                while remaining > 0 {
                    let start = positions[pos_idx];
                    let mut count = 1usize;
                    while count < remaining
                        && pos_idx + count < positions.len()
                        && positions[pos_idx + count] == start + count
                    {
                        count += 1;
                    }
                    copy_ops.push((
                        Arc::clone(&gpu),
                        src_row * row_bytes,
                        total_media_rows * row_bytes,
                        count * row_bytes,
                    ));
                    scattered_slices.push((start, count));
                    total_media_rows += count;
                    src_row += count;
                    pos_idx += count;
                    remaining -= count;
                }
            }
        }

        let mut cb = self
            .context
            .create_command_buffer(Some("d1_omni_forward"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let token_buf = cb.allocate_constant_from_slice(ids).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let feat_buf = if total_media_rows > 0 {
            let mut buf = cb
                .allocate_scratch_for_shape(&[total_media_rows as u32, self.model_dim as u32], DataType::BF16)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (gpu, src_off, dst_off, len) in &copy_ops {
                cb.encode_copy(gpu.subrange(*src_off..*src_off + *len), buf.subrange_mut(*dst_off..*dst_off + *len));
            }
            Some(buf)
        } else {
            None
        };
        let mut state = self
            .decoder_state
            .lock()
            .map_err(|e| anyhow::anyhow!("decoder_state lock: {e}"))?;
        state.reset().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        state.prepare(0, seq_len as u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let batch_dim = BatchTopology::set_attention(state_len as u32, question_group);
        let slices_ref = if scattered_slices.is_empty() {
            None
        } else {
            Some(scattered_slices.as_slice())
        };
        let out = self.decoder.encode_multimodal_with_logits(
            &token_buf,
            feat_buf.as_ref(),
            slices_ref,
            None,
            0,
            &batch_dim,
            Some(Range {
                start: 0,
                end: seq_len as u32,
            }),
            None,
            false,
            &mut state,
            &mut cb,
        )?;
        let hidden_buf = out.final_hidden.ok_or_else(|| anyhow::anyhow!("decoder produced no final_hidden"))?;

        // GPU-side slot gather + readout matmul (only copies back [code_rows.len(), readout_token_ids.len()] scores).
        let scores_global = if !code_rows.is_empty() && !readout_token_ids.is_empty() {
            let mut compact_hidden = cb
                .allocate_scratch_for_shape(&[code_rows.len() as u32, self.model_dim as u32], DataType::BF16)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (i, &r) in code_rows.iter().enumerate() {
                cb.encode_copy(
                    hidden_buf.subrange(r * row_bytes..(r + 1) * row_bytes),
                    compact_hidden.subrange_mut(i * row_bytes..(i + 1) * row_bytes),
                );
            }
            let gather_buf = cb
                .allocate_constant_from_slice(readout_token_ids)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let scores_buf = self.decoder.embedding().encode_readout(
                &compact_hidden,
                code_rows.len() as u32,
                readout_token_ids.len() as u32,
                Some(&gather_buf),
                false,
                &mut cb,
            )?;
            let scores_bytes = code_rows.len() * readout_token_ids.len() * 2;
            let mut out_global = self
                .context
                .create_buffer_gpu(scores_bytes)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            cb.encode_copy(scores_buf.subrange(0..scores_bytes), &mut out_global);
            drop(gather_buf);
            Some(out_global)
        } else {
            None
        };

        // GPU-side pointer attention logits for D1Question::Read ([read_slot_rows.len(), state_len]).
        let ptr_logits_global = if !read_slot_rows.is_empty() && state_len > 0 {
            let mut compact_queries = cb
                .allocate_scratch_for_shape(&[read_slot_rows.len() as u32, self.model_dim as u32], DataType::BF16)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (i, &r) in read_slot_rows.iter().enumerate() {
                cb.encode_copy(
                    hidden_buf.subrange(r * row_bytes..(r + 1) * row_bytes),
                    compact_queries.subrange_mut(i * row_bytes..(i + 1) * row_bytes),
                );
            }
            let state_hidden = hidden_buf.subrange(0..state_len * row_bytes);
            let mut ptr_logits_buf = cb
                .allocate_scratch_for_shape(&[read_slot_rows.len() as u32, state_len as u32], DataType::BF16)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            self.pointer_matmul
                .lock()
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &compact_queries,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: state_hidden,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        output: MatmulOutput::new(&mut ptr_logits_buf, MatmulDOps::none()),
                        gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                        m: read_slot_rows.len() as u32,
                        n: state_len as u32,
                        k: self.model_dim as u32,
                    },
                    &mut cb,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let ptr_bytes = read_slot_rows.len() * state_len * 2;
            let mut out_global = self
                .context
                .create_buffer_gpu(ptr_bytes)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            cb.encode_copy(ptr_logits_buf.subrange(0..ptr_bytes), &mut out_global);
            Some(out_global)
        } else {
            None
        };

        drop(token_buf);
        drop(feat_buf);
        block_on(cb.end_encoding().submit().wait_until_completed()).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let copyout_bf16 = |buf_opt: Option<<Metal as Backend>::GlobalGpuBuffer>| -> Result<Vec<f32>, anyhow::Error> {
            if let Some(buf) = buf_opt {
                let raw = block_on(self.context.copyout(&buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Ok(raw
                    .as_chunks::<2>().0.iter()
                    .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect())
            } else {
                Ok(Vec::new())
            }
        };

        Ok((copyout_bf16(scores_global)?, copyout_bf16(ptr_logits_global)?))
    }

    pub fn system_one(
        &self,
        state: Option<&serde_json::Value>,
        questions: &[(String, D1Question)],
        media: &[D1MediaInput],
        options: &D1Options,
    ) -> Result<D1Response, anyhow::Error> {
        anyhow::ensure!(!questions.is_empty(), "questions cannot be empty");
        let box_count = questions
            .iter()
            .filter(|(_, q)| matches!(q, D1Question::Box { .. } | D1Question::Boxes { .. }))
            .count();
        if box_count > 0 {
            let image_count = media.iter().filter(|m| matches!(m, D1MediaInput::Image(_))).count();
            anyhow::ensure!(
                image_count == 1,
                "a box or boxes question reads exactly one picture in `media`; this request has {image_count}"
            );
        }

        // Encode state media once (with cross-request GPU cache) as pure token IDs
        let mut state_media_tokens: Vec<u32> = Vec::new();
        let mut state_features: Vec<(u32, usize, Arc<<Metal as Backend>::GlobalGpuBuffer>)> = Vec::new();
        for item in media {
            let (rows, tok_id, gpu) = self.encode_media_item_cached(item)?;
            self.push_media_tokens(&mut state_media_tokens, tok_id, rows);
            state_features.push((tok_id, rows, gpu));
        }

        let (prefix_ids, prefix_positions) = build_prefix_ids(
            &self.plain_tokenizer,
            &self.special_tokens,
            state,
            &state_media_tokens,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let header = question_header(state);
        let set_size = options.set_size.unwrap_or(self.d1_config.d1_set_size).max(1);

        let names: Vec<String> = questions.iter().map(|(n, _)| n.clone()).collect();
        let q_list: Vec<&D1Question> = questions.iter().map(|(_, q)| q).collect();
        let mut question_tails: Vec<String> = Vec::with_capacity(q_list.len());
        let mut assistant_tails: Vec<Vec<u32>> = Vec::with_capacity(q_list.len());
        let mut slot_offsets_all: Vec<Vec<isize>> = Vec::with_capacity(q_list.len());
        let mut spoken_tokens_all: Vec<Vec<u32>> = Vec::with_capacity(q_list.len());
        let mut spoken_features: Vec<Vec<(u32, usize, Arc<<Metal as Backend>::GlobalGpuBuffer>)>> =
            Vec::with_capacity(q_list.len());

        for q in &q_list {
            let (tail, assistant_ids, s_offsets) = suffix_layout_parts(
                &self.tokenizer,
                &self.plain_tokenizer,
                q,
                &self.lead,
                self.d1_config.d1_codes,
                self.d1_config.d1_numbers,
                self.d1_config.d1_prompt,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut own_tokens = Vec::new();
            let mut own_feats = Vec::new();
            if let D1Instruction::Spoken {
                waveform,
                path,
                ..
            } = q.instructions()
            {
                let samples = waveform.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "spoken instruction audio must be decoded to waveform before calling system_one (path={path:?})"
                    )
                })?;
                let (rows, tok_id, gpu) = self.encode_audio_cached(samples)?;
                self.push_media_tokens(&mut own_tokens, tok_id, rows);
                own_feats.push((tok_id, rows, gpu));
            }
            question_tails.push(tail);
            assistant_tails.push(assistant_ids);
            slot_offsets_all.push(s_offsets);
            spoken_tokens_all.push(own_tokens);
            spoken_features.push(own_feats);
        }

        let mut answers = Vec::with_capacity(questions.len());
        let mut total_input_tokens = 0usize;

        for chunk_start in (0..q_list.len()).step_by(set_size) {
            let chunk_end = (chunk_start + set_size).min(q_list.len());
            let group_qs = &q_list[chunk_start..chunk_end];
            let group_names = &names[chunk_start..chunk_end];
            let group_tails = &question_tails[chunk_start..chunk_end];
            let group_assistant = &assistant_tails[chunk_start..chunk_end];
            let group_offsets = &slot_offsets_all[chunk_start..chunk_end];
            let group_spoken = &spoken_tokens_all[chunk_start..chunk_end];

            let (packed_ids, state_len, q_group, ends) = pack_set_ids(
                &self.plain_tokenizer,
                &self.special_tokens,
                &prefix_ids,
                group_qs,
                group_tails,
                group_assistant,
                group_offsets,
                group_spoken,
                header,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            total_input_tokens += packed_ids.len();

            let mut row_media: Vec<(u32, usize, Arc<<Metal as Backend>::GlobalGpuBuffer>)> =
                state_features.iter().map(|(t, r, g)| (*t, *r, Arc::clone(g))).collect();
            for item_feats in &spoken_features[chunk_start..chunk_end] {
                for (t, r, g) in item_feats {
                    row_media.push((*t, *r, Arc::clone(g)));
                }
            }

            let has_read = group_qs.iter().any(|q| matches!(q, D1Question::Read { .. }));
            let positions = if has_read { prefix_positions.as_ref() } else { None };

            // Collect all non-pointer slots, unique readout tokens, and Read pointer slots
            // BEFORE the GPU forward so slot gather + readout/pointer matmuls run in one CommandBuffer.
            struct CodeSlot<'a> {
                q_idx: usize,
                spec: SlotSpec,
                pos: usize,
                tempered: bool,
                groups: Vec<Vec<u32>>,
                _phantom: std::marker::PhantomData<&'a ()>,
            }
            let mut code_slots: Vec<CodeSlot> = Vec::new();
            let mut unique_tokens: Vec<u32> = Vec::new();
            let mut tok_to_idx: HashMap<u32, usize> = HashMap::new();
            let mut code_rows: Vec<usize> = Vec::new();
            let mut read_slot_rows: Vec<usize> = Vec::new();

            for (j, (q, slots)) in group_qs.iter().zip(ends.iter()).enumerate() {
                if matches!(q, D1Question::Read { .. }) {
                    read_slot_rows.extend_from_slice(slots);
                } else {
                    let specs = question_slots(q, &self.tokenizer, self.d1_config.d1_numbers);
                    for (spec, &pos) in specs.into_iter().zip(slots.iter()) {
                        let groups = spec
                            .readout_ids(&self.tokenizer, self.d1_config.d1_codes)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        for g in &groups {
                            for &tid in g {
                                if let std::collections::hash_map::Entry::Vacant(e) = tok_to_idx.entry(tid) {
                                    e.insert(unique_tokens.len());
                                    unique_tokens.push(tid);
                                }
                            }
                        }
                        code_rows.push(pos);
                        code_slots.push(CodeSlot {
                            q_idx: j,
                            spec,
                            pos,
                            tempered: !q.is_typed(),
                            groups,
                            _phantom: std::marker::PhantomData,
                        });
                    }
                }
            }

            code_rows.sort_unstable();
            code_rows.dedup();
            read_slot_rows.sort_unstable();
            read_slot_rows.dedup();

            let (readout_scores, ptr_logits) = self.d1_forward_and_readout(
                &packed_ids,
                state_len,
                &q_group,
                &row_media,
                &code_rows,
                &unique_tokens,
                &read_slot_rows,
            )?;

            let mut slot_probs_by_q: Vec<Vec<Vec<f64>>> = vec![Vec::new(); group_qs.len()];
            let stride = unique_tokens.len();
            for cs in &code_slots {
                let row_idx = code_rows.binary_search(&cs.pos).expect("code slot row");
                let row_scores = &readout_scores[row_idx * stride..(row_idx + 1) * stride];
                let mut scores = Vec::with_capacity(cs.groups.len());
                for g in &cs.groups {
                    let mut best = f64::NEG_INFINITY;
                    for &tid in g {
                        let idx = tok_to_idx[&tid];
                        let dot = row_scores[idx] as f64;
                        if dot > best {
                            best = dot;
                        }
                    }
                    scores.push(best);
                }
                let cal = if cs.tempered {
                    Some(&self.calibration)
                } else {
                    None
                };
                let probs = probabilities(&cs.spec, &scores, cal);
                slot_probs_by_q[cs.q_idx].push(probs);
            }

            for (j, (name, q)) in group_names.iter().zip(group_qs.iter()).enumerate() {
                let ans = self.decode_answer(
                    q,
                    &ends[j],
                    &slot_probs_by_q[j],
                    &ptr_logits,
                    &read_slot_rows,
                    state_len,
                    positions,
                    options,
                )?;
                answers.push((name.clone(), ans));
            }
        }

        Ok(D1Response {
            answers,
            usage: D1Usage {
                input_tokens: total_input_tokens,
                output_tokens: 0,
            },
        })
    }

    fn decode_answer(
        &self,
        q: &D1Question,
        slots: &[usize],
        slot_probs: &[Vec<f64>],
        ptr_logits: &[f32],
        read_slot_rows: &[usize],
        state_len: usize,
        positions: Option<&StatePositions>,
        options: &D1Options,
    ) -> Result<serde_json::Value, anyhow::Error> {
        match q {
            D1Question::Noul {
                ..
            } => {
                let p_yes = slot_probs[0][0];
                Ok(serde_json::json!({
                    "type": "noul",
                    "noul": p_yes,
                }))
            },
            D1Question::Choice {
                criteria,
                multiple,
                ..
            } => {
                if *multiple {
                    let mut choices = serde_json::Map::new();
                    for ((name, _), p) in criteria.iter().zip(slot_probs.iter()) {
                        choices.insert(name.clone(), serde_json::json!(self.calibration.through("multiple", p[0])));
                    }
                    Ok(serde_json::json!({
                        "type": "choice",
                        "choices": choices,
                    }))
                } else {
                    let probs = &slot_probs[0];
                    let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
                    let best_idx = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let mut prob_map = serde_json::Map::new();
                    for (lab, &p) in labels.iter().zip(probs.iter()) {
                        prob_map.insert(lab.clone(), serde_json::json!(p));
                    }
                    Ok(serde_json::json!({
                        "type": "choice",
                        "choice": labels[best_idx],
                        "confidence": probs[best_idx],
                        "probabilities": prob_map,
                    }))
                }
            },
            D1Question::Score {
                criteria,
                ..
            } => {
                let probs = &slot_probs[0];
                let total: f64 = probs.iter().sum::<f64>().max(1e-12);
                let level = match options.score_readout {
                    ScoreReadout::Mean => {
                        let exp: f64 = probs.iter().enumerate().map(|(i, &p)| i as f64 * p / total).sum();
                        (exp.round() as isize).clamp(0, criteria.len() as isize - 1) as usize
                    },
                    ScoreReadout::Median => {
                        let mut mass = 0.0;
                        let mut found = criteria.len() - 1;
                        for (i, &p) in probs.iter().enumerate() {
                            mass += p / total;
                            if mass >= 0.5 {
                                found = i;
                                break;
                            }
                        }
                        found
                    },
                    ScoreReadout::Argmax => probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0),
                };
                let mut prob_map = serde_json::Map::new();
                let mut legend = serde_json::Map::new();
                for (i, (label, &p)) in criteria.iter().zip(probs.iter()).enumerate() {
                    prob_map.insert(i.to_string(), serde_json::json!(p));
                    legend.insert(i.to_string(), serde_json::json!(label));
                }
                Ok(serde_json::json!({
                    "type": "score",
                    "score": level as f64,
                    "confidence": probs[level],
                    "probabilities": prob_map,
                    "legend": legend,
                }))
            },
            D1Question::Read {
                kind,
                ..
            } => {
                let pos = positions.ok_or_else(|| anyhow::anyhow!("missing state positions for Read"))?;
                let cap = match kind.as_str() {
                    "number" | "date" => self.d1_config.d1_read_short_max_tokens,
                    _ => self.d1_config.d1_read_max_tokens,
                };
                let s_idx = read_slot_rows.binary_search(&slots[0]).expect("read start slot row");
                let e_idx = read_slot_rows.binary_search(&slots[1]).expect("read end slot row");
                let s_logits = &ptr_logits[s_idx * state_len..(s_idx + 1) * state_len];
                let e_logits = &ptr_logits[e_idx * state_len..(e_idx + 1) * state_len];
                let (first, last, raw_found) = pointer_span_from_logits(s_logits, e_logits, self.model_dim, pos, cap);
                let found = self.calibration.through(&format!("found:{kind}"), raw_found);
                if let Some((text, span)) = pos.read_span(first, last) {
                    Ok(serde_json::json!({
                        "type": kind,
                        kind.as_str(): text,
                        "span": span,
                        "found": found,
                    }))
                } else {
                    Ok(serde_json::json!({
                        "type": kind,
                        kind.as_str(): serde_json::Value::Null,
                        "span": serde_json::Value::Null,
                        "found": found,
                    }))
                }
            },
            D1Question::Estimate {
                low,
                high,
                step,
                unit,
                ..
            } => {
                let probs = &slot_probs[0];
                let g = estimate_grid(*low, *high, *step);
                let val = g.median(probs);
                let lo_q = self.calibration.level("estimate", 0.05);
                let hi_q = self.calibration.level("estimate", 0.95);
                let lo = g.quantile(probs, lo_q);
                let hi = g.quantile(probs, hi_q);
                Ok(serde_json::json!({
                    "type": "number",
                    "number": val,
                    "interval": [lo, hi],
                    "unit": unit,
                }))
            },
            D1Question::Box {
                ..
            } => {
                let found = self.calibration.through("found:box", slot_probs[0][0]);
                if found < 0.5 {
                    return Ok(serde_json::json!({
                        "type": "box",
                        "box": serde_json::Value::Null,
                        "found": found,
                        "confidence": 0.0,
                    }));
                }
                let rect = box_medians(&slot_probs[1..5], &self.d1_config.d1_box_edges);
                let conf = self
                    .calibration
                    .through("box", box_confidence(&slot_probs[1..5], &rect));
                Ok(serde_json::json!({
                    "type": "box",
                    "box": {
                        "top": rect[0],
                        "bottom": rect[1],
                        "left": rect[2],
                        "right": rect[3],
                    },
                    "found": found,
                    "confidence": conf,
                }))
            },
            D1Question::Boxes {
                ..
            } => {
                let mut presence = Vec::new();
                let mut below = 1.0f64;
                for k in (0..slot_probs.len()).step_by(5) {
                    below = below.min(self.calibration.through("found:boxes", slot_probs[k][0]));
                    presence.push(below);
                }
                let mut boxes = Vec::new();
                for (k, &f) in presence.iter().enumerate() {
                    if f < 0.5 {
                        break;
                    }
                    let block = &slot_probs[5 * k + 1..5 * k + 5];
                    let rect = box_medians(block, &self.d1_config.d1_box_edges);
                    let conf = self.calibration.through("boxes", box_confidence(block, &rect));
                    boxes.push(serde_json::json!({
                        "box": {
                            "top": rect[0],
                            "bottom": rect[1],
                            "left": rect[2],
                            "right": rect[3],
                        },
                        "found": f,
                        "confidence": conf,
                    }));
                }
                boxes.sort_by(|a, b| {
                    let al = a["box"]["left"].as_f64().unwrap_or(0.0);
                    let bl = b["box"]["left"].as_f64().unwrap_or(0.0);
                    let at = a["box"]["top"].as_f64().unwrap_or(0.0);
                    let bt = b["box"]["top"].as_f64().unwrap_or(0.0);
                    al.partial_cmp(&bl)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal))
                });
                let at_least = !boxes.is_empty() && boxes.len() == presence.len();
                Ok(serde_json::json!({
                    "type": "boxes",
                    "boxes": boxes,
                    "count": boxes.len(),
                    "at_least": at_least,
                    "presence": presence,
                }))
            },
        }
    }
}
