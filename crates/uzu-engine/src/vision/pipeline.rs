use std::path::Path;

use futures::executor::block_on;
use tokenizers::Tokenizer;

use super::common::*;
use crate::{
    backends::{
        common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        common::kernel::ContextRingUpdateKernel,
        metal::Metal,
    },
    encodable_block::{batch_topology::BatchTopology, transformer::TransformerState},
    engine::language_model::LanguageModel,
    engine_verbose, verbose,
    vision::{
        CanvasConfig, MetalVisionTower, VisionConfig, build_placeholder_tokens_with_config,
        compute_canvas_mrope_with_config, preprocess_image,
    },
};

/// A message flattened onto the template's three renderable roles (system, user, tool
/// groups become user turns). Text is fully formatted; only images still need tokenization.
struct RenderedTurn {
    role: String,
    text: Option<String>,
    images: Vec<image::DynamicImage>,
}

fn normalize_turns(
    messages: &[MultimodalMessage],
    tools_preamble: Option<&str>,
) -> Vec<RenderedTurn> {
    let mut turns = Vec::new();
    let mut index = 0;
    if let Some(preamble) = tools_preamble {
        // The template renders exactly one system message: the preamble, plus the
        // client's system text (only when it is the first message) after a blank line.
        let system_text = match messages.first() {
            Some(msg) if msg.role == "system" => {
                index = 1;
                match msg.content.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
                    Some(content) => format!("{preamble}\n\n{content}"),
                    None => preamble.to_string(),
                }
            },
            _ => preamble.to_string(),
        };
        turns.push(RenderedTurn {
            role: "system".to_string(),
            text: Some(system_text),
            images: Vec::new(),
        });
    }
    while index < messages.len() {
        let msg = &messages[index];
        match msg.role.as_str() {
            "system" => {
                turns.push(RenderedTurn {
                    role: "system".to_string(),
                    text: msg.content.clone(),
                    images: Vec::new(),
                });
            },
            "user" => {
                turns.push(RenderedTurn {
                    role: "user".to_string(),
                    text: msg.content.clone(),
                    images: msg.images.clone(),
                });
            },
            "assistant" => {
                turns.push(RenderedTurn {
                    role: "assistant".to_string(),
                    text: Some(render_assistant_text(msg)),
                    images: Vec::new(),
                });
            },
            "tool" => {
                let mut parts = Vec::new();
                while index < messages.len() && messages[index].role == "tool" {
                    let content = messages[index].content.as_deref().unwrap_or_default();
                    parts.push(format!("<tool_response>\n{}\n</tool_response>", content.trim()));
                    index += 1;
                }
                turns.push(RenderedTurn {
                    role: "user".to_string(),
                    text: Some(parts.join("\n")),
                    images: Vec::new(),
                });
                continue;
            },
            _ => {},
        }
        index += 1;
    }
    turns
}

pub struct PreparedMultimodalInput {
    pub prompt_tokens: Vec<u64>,
    pub features_bf16: Vec<half::bf16>,
    pub visual_slices: Vec<(usize, usize)>,
    pub mrope_3d: Vec<[u32; 3]>,
    pub delta: i64,
}

/// Cache stores only what a hit consumes: the placeholder grids and the
/// projector features (shared through `Arc`, so a hit never copies features
/// or the multi-megabyte patch tensor).
#[derive(Clone)]
pub struct CachedImageEntry {
    pub hash: u64,
    pub grids: Vec<(usize, usize, usize)>,
    pub slice_grid: Option<(usize, usize)>,
    pub features_bf16: std::sync::Arc<Vec<half::bf16>>,
}

pub struct VlmPipeline {
    pub vision_tower: MetalVisionTower,
    pub vision_config: VisionConfig,
    pub canvas_config: CanvasConfig,
    cached_states: std::sync::Mutex<Vec<TransformerState<Metal>>>,
    image_cache: std::sync::Mutex<Vec<CachedImageEntry>>,
    /// Lazily built GPU sampling bitmask (1 = valid id) for padded vocab rows.
    vocab_padding_mask: std::sync::OnceLock<Vec<u32>>,
}

impl VlmPipeline {
    pub fn load_from_model_dir(model_dir: &Path) -> Result<Self, anyhow::Error> {
        let vision_tower = MetalVisionTower::load_from_model_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to load MetalVisionTower: {e}"))?;

        let (vision_config, canvas_config) =
            if let Ok(data) = std::fs::read_to_string(model_dir.join("vision_config.json")) {
                let val: serde_json::Value = serde_json::from_str(&data).unwrap_or_default();
                let v_cfg: VisionConfig =
                    serde_json::from_value(val.get("vision_config").cloned().unwrap_or_default()).unwrap_or_default();
                let c_cfg: CanvasConfig =
                    serde_json::from_value(val.get("canvas_config").cloned().unwrap_or_default()).unwrap_or_default();
                (v_cfg, c_cfg)
            } else {
                (VisionConfig::default(), CanvasConfig::default())
            };

        Ok(Self {
            vision_tower,
            vision_config,
            canvas_config,
            cached_states: std::sync::Mutex::new(Vec::new()),
            image_cache: std::sync::Mutex::new(Vec::new()),
            vocab_padding_mask: std::sync::OnceLock::new(),
        })
    }

    pub fn prepare_inputs(
        &self,
        images: &[image::DynamicImage],
        prompt_text: &str,
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<PreparedMultimodalInput, anyhow::Error> {
        let msg = MultimodalMessage {
            role: "user".to_string(),
            content: Some(prompt_text.to_string()),
            reasoning_content: None,
            images: images.to_vec(),
            videos: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        self.prepare_conversation(&[msg], tokenizer, enable_thinking)
    }

    pub fn prepare_conversation(
        &self,
        messages: &[MultimodalMessage],
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<PreparedMultimodalInput, anyhow::Error> {
        self.prepare_conversation_with_tools(messages, tokenizer, enable_thinking, None)
    }

    /// Same as `prepare_conversation`, with tool support: `tools_preamble` (see
    /// [`tools_system_preamble`]) is rendered as the system message exactly the way
    /// the model's chat template does when tools are declared.
    pub fn prepare_conversation_with_tools(
        &self,
        messages: &[MultimodalMessage],
        tokenizer: &Tokenizer,
        enable_thinking: bool,
        tools_preamble: Option<&str>,
    ) -> Result<PreparedMultimodalInput, anyhow::Error> {
        if messages.iter().any(|message| !message.videos.is_empty()) {
            anyhow::bail!("MiniCPM-V does not support video input; use an LFM3-VL model");
        }
        let mut canvas_cfg = self.canvas_config.clone();
        canvas_cfg.resolve_from_tokenizer(tokenizer);
        let divisor = canvas_cfg.merge_factor() * canvas_cfg.merge_factor();

        let turns = normalize_turns(messages, tools_preamble);

        let mut all_features_bf16 = Vec::new();
        let mut all_grids = Vec::new();
        let mut prompt_tokens = Vec::new();
        let mut global_image_count = 0;

        for turn in &turns {
            match turn.role.as_str() {
                "system" => {
                    if let Some(c) = &turn.text {
                        let formatted = format!("<|im_start|>system\n{c}<|im_end|>\n");
                        let enc = tokenizer.encode(formatted, false).map_err(|e| anyhow::anyhow!("{e}"))?;
                        prompt_tokens.extend(enc.get_ids().iter().map(|&t| t as u64));
                    }
                },
                "user" => {
                    let enc_prefix =
                        tokenizer.encode("<|im_start|>user\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
                    prompt_tokens.extend(enc_prefix.get_ids().iter().map(|&t| t as u64));

                    if !turn.images.is_empty() {
                        for (local_idx, img) in turn.images.iter().enumerate() {
                            let img_hash = hash_image(img);
                            let cached = {
                                let guard = self.image_cache.lock().unwrap();
                                guard.iter().find(|e| e.hash == img_hash).cloned()
                            };
                            let (grids, slice_grid, features) = match cached {
                                Some(entry) => {
                                    engine_verbose!(
                                        "⚡ [Image Cache Hit] Reusing precomputed visual features for image {global_image_count} (0.000s)"
                                    );
                                    (entry.grids, entry.slice_grid, entry.features_bf16)
                                },
                                None => {
                                    let t_img_pre = std::time::Instant::now();
                                    let pre = preprocess_image(img, &self.vision_config, &canvas_cfg, 9, 448);
                                    let t_img_pre_done = t_img_pre.elapsed();

                                    let t_tower = std::time::Instant::now();
                                    let visual_tokens = self.vision_tower.forward_f16(&pre).map_err(|e| {
                                        anyhow::anyhow!("Vision forward failed for image {global_image_count}: {e}")
                                    })?;
                                    let t_tower_done = t_tower.elapsed();
                                    engine_verbose!(
                                        "🔍 [Vision Breakdown] Image {}: preprocess_image = {:.3}s, vision_tower = {:.3}s ({} patches, {} out tokens)",
                                        global_image_count,
                                        t_img_pre_done.as_secs_f64(),
                                        t_tower_done.as_secs_f64(),
                                        pre.patches.len() / (14 * 14 * 3),
                                        visual_tokens.len() / self.vision_config.hidden_size
                                    );
                                    let feat: std::sync::Arc<Vec<half::bf16>> = std::sync::Arc::new(
                                        visual_tokens.iter().map(|&x| half::bf16::from_f32(x.to_f32())).collect(),
                                    );
                                    let mut guard = self.image_cache.lock().unwrap();
                                    if guard.len() >= 8 {
                                        guard.remove(0);
                                    }
                                    guard.push(CachedImageEntry {
                                        hash: img_hash,
                                        grids: pre.grids.clone(),
                                        slice_grid: pre.slice_grid,
                                        features_bf16: feat.clone(),
                                    });
                                    (pre.grids.clone(), pre.slice_grid, feat)
                                },
                            };

                            let placeholders = build_placeholder_tokens_with_config(
                                &grids,
                                slice_grid,
                                global_image_count,
                                true,
                                divisor,
                                &canvas_cfg,
                            );
                            prompt_tokens.extend(placeholders.into_iter().map(|t| t as u64));
                            if local_idx + 1 < turn.images.len() {
                                prompt_tokens.push(canvas_cfg.newline_id as u64);
                            }
                            all_features_bf16.extend(features.iter().copied());
                            all_grids.extend(grids);
                            global_image_count += 1;
                        }
                        prompt_tokens.push(canvas_cfg.newline_id as u64);
                    }

                    if let Some(c) = &turn.text
                        && !c.trim().is_empty() {
                            let enc = tokenizer.encode(c.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
                            prompt_tokens.extend(enc.get_ids().iter().map(|&t| t as u64));
                        }
                    let enc_end = tokenizer.encode("<|im_end|>\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
                    prompt_tokens.extend(enc_end.get_ids().iter().map(|&t| t as u64));
                },
                "assistant" => {
                    let enc_prefix =
                        tokenizer.encode("<|im_start|>assistant\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
                    prompt_tokens.extend(enc_prefix.get_ids().iter().map(|&t| t as u64));
                    if let Some(body) = &turn.text
                        && !body.is_empty() {
                            let enc = tokenizer.encode(body.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
                            prompt_tokens.extend(enc.get_ids().iter().map(|&t| t as u64));
                        }
                    let enc_end = tokenizer.encode("<|im_end|>\n", false).map_err(|e| anyhow::anyhow!("{e}"))?;
                    prompt_tokens.extend(enc_end.get_ids().iter().map(|&t| t as u64));
                },
                _ => {},
            }
        }

        let suffix_str = if enable_thinking {
            "<|im_start|>assistant\n<think>\n"
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        };
        let t_suffix = tokenizer.encode(suffix_str, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        prompt_tokens.extend(t_suffix.get_ids().iter().map(|&t| t as u64));

        let pad_id = canvas_cfg.image_token_id as u64;
        let visual_token_indices: Vec<usize> =
            prompt_tokens.iter().enumerate().filter(|(_, t)| **t == pad_id).map(|(i, _)| i).collect();

        let (mrope_3d, delta) =
            compute_canvas_mrope_with_config(&prompt_tokens, &all_grids, canvas_cfg.merge_factor(), &canvas_cfg);

        let mut visual_slices: Vec<(usize, usize)> = Vec::new();
        if !visual_token_indices.is_empty() {
            let mut start = visual_token_indices[0];
            let mut count = 1;
            for &idx in &visual_token_indices[1..] {
                if idx == start + count {
                    count += 1;
                } else {
                    visual_slices.push((start, count));
                    start = idx;
                    count = 1;
                }
            }
            visual_slices.push((start, count));
        }

        Ok(PreparedMultimodalInput {
            prompt_tokens,
            features_bf16: all_features_bf16,
            visual_slices,
            mrope_3d,
            delta,
        })
    }

    pub fn prefill(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &PreparedMultimodalInput,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        let context = model.context();

        // The attention suffix is hardcoded to ATTENTION_SUFFIX_CAPACITY, so
        // large prompts (multi-slice images) must be encoded in chunks like the
        // text path in engine/language_model/stream does: all but the last
        // chunk are committed to the KV prefix via encode_accept.
        let max_batch = crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY as usize;
        let total = prepared.prompt_tokens.len();
        let batch_count = total.div_ceil(max_batch);
        state
            .state
            .prepare(
                state.state.context_length + ((batch_count - 1) * max_batch) as u32,
                usize::min(max_batch, total) as u32,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        // Per-turn GPU ring for the repetition penalty kernel; created empty,
        // then fed with this turn's prompt tokens chunk by chunk.
        let prefix_len = state.state.context_length;
        if sampler.repetition_penalty().is_some() {
            let suffix = prefix_len + total as u32;
            state.context_ring = Some((create_context_ring(context, suffix)?, suffix));
        }

        let mut command_buffer = context
            .create_command_buffer(Some("vlm_prefill"), None)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .with_timestamps();
        let elems_per_token = model.decoder().model_dim() as usize;

        let t_enc_start = std::time::Instant::now();
        let mut sampled_buf: Option<<Metal as crate::backends::common::Backend>::GlobalGpuBuffer> = None;
        let mut visual_before = 0usize;
        let mut pos = 0usize;
        for batch in 0..batch_count {
            let is_last = batch + 1 == batch_count;
            let len = usize::min(max_batch, total - pos);
            let input_u32: Vec<u32> = prepared.prompt_tokens[pos..pos + len].iter().map(|&x| x as u32).collect();
            let token_ids =
                command_buffer.allocate_constant_from_slice(&input_u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;

            // Rebase the visual-token runs overlapping this chunk. Visual
            // features are consumed sequentially by the decoder, so each chunk
            // receives a feature slice starting at its first visual token.
            let mut chunk_slices: Vec<(usize, usize)> = Vec::new();
            let mut chunk_features: Option<&[half::bf16]> = None;
            if !prepared.visual_slices.is_empty() {
                let chunk_end = pos + len;
                let mut chunk_visual = 0usize;
                for &(start, count) in &prepared.visual_slices {
                    let overlap_start = start.max(pos);
                    let overlap_end = (start + count).min(chunk_end);
                    if overlap_end > overlap_start {
                        chunk_slices.push((overlap_start - pos, overlap_end - overlap_start));
                        chunk_visual += overlap_end - overlap_start;
                    }
                }
                if chunk_visual > 0 {
                    chunk_features = Some(
                        &prepared.features_bf16
                            [visual_before * elems_per_token..(visual_before + chunk_visual) * elems_per_token],
                    );
                }
                visual_before += chunk_visual;
            }
            let features_gpu = chunk_features
                .map(|features| {
                    command_buffer.allocate_constant_from_slice(features).map_err(|e| anyhow::anyhow!("{e:?}"))
                })
                .transpose()?;
            let chunk_slices = (!chunk_slices.is_empty()).then_some(chunk_slices);

            let batch_dim = BatchTopology::flat(len as u32, true);
            let output = model
                .decoder()
                .encode_multimodal(
                    &token_ids,
                    features_gpu.as_ref(),
                    chunk_slices.as_deref(),
                    Some(&prepared.mrope_3d[pos..pos + len]),
                    prepared.delta,
                    &batch_dim,
                    is_last.then(|| (len as u32 - 1..len as u32).into()),
                    None,
                    &mut state.state,
                    &mut command_buffer,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;

            if is_last {
                // Full GPU sampling (greedy or stochastic): only 4 bytes per
                // row come back.
                let logits = output.logits.as_ref().unwrap();
                let mask = self.vocab_mask(model.vocab_size(), sampler.vocab_limit());
                let ring = state.context_ring.as_ref().map(|(ring, suffix)| (ring, *suffix));
                sampled_buf = Some(
                    encode_vlm_sampling(
                        model,
                        &mut command_buffer,
                        logits,
                        &token_ids,
                        ring,
                        &batch_dim,
                        (len as u32 - 1..len as u32).into(),
                        sampler,
                        mask,
                        prefix_len + total as u32 - 1,
                    )
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?,
                );
            }
            if let Some((ring, suffix)) = state.context_ring.as_mut() {
                model.context_ring_update().encode(&token_ids, ring, *suffix, len as u32, &mut command_buffer);
            }
            state
                .state
                .encode_accept(&(0..len as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            drop(token_ids);
            drop(features_gpu);
            pos += len;
        }
        let t_enc = t_enc_start.elapsed();

        let t_submit_start = std::time::Instant::now();
        let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let t_gpu = t_submit_start.elapsed();

        let t_copy_start = std::time::Instant::now();
        let buf = sampled_buf.expect("prefill produced no sampled tokens");
        let ids: Vec<u32> = block_on(context.copyout(&buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let t_copy = t_copy_start.elapsed();

        engine_verbose!(
            "🔍 [Prefill Breakdown] CPU Encode: {:.3}s, GPU Execute: {:.3}s, Copyout: {:.3}s",
            t_enc.as_secs_f64(),
            t_gpu.as_secs_f64(),
            t_copy.as_secs_f64()
        );

        if verbose::is_verbose() {
            let mut span_times: std::collections::HashMap<String, std::time::Duration> =
                std::collections::HashMap::new();
            use crate::backends::common::CommandBufferCompleted;
            for span in completed.timestamps() {
                let dur = span.end.saturating_duration_since(span.start);
                // Extract leaf name (last component of path)
                let leaf = span.name.rsplit('/').next().unwrap_or(&span.name).to_string();
                *span_times.entry(leaf).or_insert(std::time::Duration::ZERO) += dur;
            }
            let mut sorted_spans: Vec<(String, std::time::Duration)> = span_times.into_iter().collect();
            sorted_spans.sort_by_key(|(_, duration)| std::cmp::Reverse(*duration));
            engine_verbose!("🔍 [Top GPU Operator Spans in Prefill]");
            for (name, dur) in sorted_spans.iter().take(10) {
                engine_verbose!("   - {name}: {:.3}s", dur.as_secs_f64());
            }
        }

        let first_token = ids.first().copied().unwrap_or(0) as u64;
        sampler.accept_token(first_token);
        Ok(first_token)
    }

    /// Lazily built padding bitmask over the decoder's readout width; ids past
    /// the tokenizer vocabulary are untrained padding and never sampled.
    fn vocab_mask(&self, decoder_vocab: usize, vocab_limit: Option<usize>) -> &[u32] {
        self.vocab_padding_mask.get_or_init(|| vocab_padding_bitmask(decoder_vocab, vocab_limit.unwrap_or(decoder_vocab)))
    }

    pub fn acquire_state(
        &self,
        model: &LanguageModel<Metal>,
        max_context_tokens: Option<u32>,
    ) -> Result<VlmState, anyhow::Error> {
        let mut guard = self.cached_states.lock().unwrap();
        while let Some(mut state) = guard.pop() {
            if state.reset().is_ok() {
                return Ok(VlmState {
                    state,
                    context_ring: None,
                });
            }
        }
        drop(guard);
        // Server override first, then the model's platform recommendation
        // (sparse-capable devices: RoPE max; otherwise the platform clamp).
        let recommended = max_context_tokens.or_else(|| model.recommended_context_length());
        let state =
            model.decoder().create_empty_state(recommended, model.context()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(VlmState {
            state,
            context_ring: None,
        })
    }

    pub fn release_state(
        &self,
        state: VlmState,
    ) {
        let mut guard = self.cached_states.lock().unwrap();
        if guard.len() < 4 {
            guard.push(state.state);
        }
    }

    pub fn prefill_with_state(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prepared: &PreparedMultimodalInput,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        self.prefill(model, state, prepared, sampler)
    }

    pub fn decode_step_with_state(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        token: u64,
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        let context = model.context();
        state.state.prepare(state.state.context_length, 1).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut command_buffer =
            context.create_command_buffer(Some("vlm_decode"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let next_u32 = [token as u32];
        let token_ids = command_buffer.allocate_constant_from_slice(&next_u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let batch_dim = BatchTopology::flat(1, true);

        let output = model
            .decoder()
            .encode(&token_ids, &batch_dim, Some((0..1).into()), None, &mut state.state, &mut command_buffer)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        // Full GPU sampling (greedy or stochastic): only 4 bytes come back.
        let logits = output.logits.as_ref().unwrap();
        let mask = self.vocab_mask(model.vocab_size(), sampler.vocab_limit());
        let ring = state.context_ring.as_ref().map(|(ring, suffix)| (ring, *suffix));
        let sampled_buf = encode_vlm_sampling(
            model,
            &mut command_buffer,
            logits,
            &token_ids,
            ring,
            &batch_dim,
            (0u32..1u32).into(),
            sampler,
            mask,
            state.state.context_length,
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        if let Some((ring, suffix)) = state.context_ring.as_mut() {
            model.context_ring_update().encode(&token_ids, ring, *suffix, 1, &mut command_buffer);
        }
        state.state.encode_accept(&[0], &mut command_buffer).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        drop(token_ids);
        block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let ids: Vec<u32> = block_on(context.copyout(&sampled_buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let next_token = ids.first().copied().unwrap_or(0) as u64;
        sampler.accept_token(next_token);
        Ok(next_token)
    }

    /// Formats a follow-up user prompt into ChatML tokens for multi-turn conversation.
    /// Closes previous turn with `<|im_end|>\n` and opens new user query and assistant prefix.
    pub fn format_followup_tokens(
        &self,
        prompt_text: &str,
        tokenizer: &Tokenizer,
        enable_thinking: bool,
    ) -> Result<Vec<u64>, anyhow::Error> {
        let suffix_str = if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        let formatted =
            format!("<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n{suffix_str}");
        let enc = tokenizer.encode(formatted, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(enc.get_ids().iter().map(|&t| t as u64).collect())
    }

    /// Incremental text-only prefill on existing TransformerState for multi-turn conversation.
    /// Reuses all previously cached visual tokens and conversation history without recomputation!
    pub fn prefill_incremental_text(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        new_tokens: &[u64],
        sampler: &mut VlmSampler,
    ) -> Result<u64, anyhow::Error> {
        let context = model.context();
        let suffix_len = new_tokens.len() as u32;
        if suffix_len == 0 {
            anyhow::bail!("Cannot prefill empty tokens");
        }

        // Chunk like the text path: the attention suffix is capped at
        // ATTENTION_SUFFIX_CAPACITY; all but the last chunk are committed to
        // the KV prefix via encode_accept.
        let max_batch = crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY as usize;
        let batch_count = new_tokens.len().div_ceil(max_batch);
        state
            .state
            .prepare(
                state.state.context_length + ((batch_count - 1) * max_batch) as u32,
                usize::min(max_batch, new_tokens.len()) as u32,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        // Per-turn GPU ring for the repetition penalty kernel.
        let prefix_len = state.state.context_length;
        if sampler.repetition_penalty().is_some() {
            let suffix = prefix_len + new_tokens.len() as u32;
            state.context_ring = Some((create_context_ring(context, suffix)?, suffix));
        }

        let mut command_buffer =
            context.create_command_buffer(Some("vlm_text_prefill"), None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut sampled_buf: Option<<Metal as crate::backends::common::Backend>::GlobalGpuBuffer> = None;
        let mut pos = 0usize;
        for batch in 0..batch_count {
            let is_last = batch + 1 == batch_count;
            let len = usize::min(max_batch, new_tokens.len() - pos);
            let input_u32: Vec<u32> = new_tokens[pos..pos + len].iter().map(|&x| x as u32).collect();
            let token_ids =
                command_buffer.allocate_constant_from_slice(&input_u32).map_err(|e| anyhow::anyhow!("{e:?}"))?;

            let batch_dim = BatchTopology::flat(len as u32, true);
            let output = model
                .decoder()
                .encode(
                    &token_ids,
                    &batch_dim,
                    is_last.then(|| (len as u32 - 1..len as u32).into()),
                    None,
                    &mut state.state,
                    &mut command_buffer,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;

            if is_last {
                // Full GPU sampling (greedy or stochastic): only 4 bytes per
                // row come back.
                let logits = output.logits.as_ref().unwrap();
                let mask = self.vocab_mask(model.vocab_size(), sampler.vocab_limit());
                let ring = state.context_ring.as_ref().map(|(ring, suffix)| (ring, *suffix));
                sampled_buf = Some(
                    encode_vlm_sampling(
                        model,
                        &mut command_buffer,
                        logits,
                        &token_ids,
                        ring,
                        &batch_dim,
                        (len as u32 - 1..len as u32).into(),
                        sampler,
                        mask,
                        prefix_len + new_tokens.len() as u32 - 1,
                    )
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?,
                );
            }
            if let Some((ring, suffix)) = state.context_ring.as_mut() {
                model.context_ring_update().encode(&token_ids, ring, *suffix, len as u32, &mut command_buffer);
            }
            state
                .state
                .encode_accept(&(0..len as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            drop(token_ids);
            pos += len;
        }
        let buf = sampled_buf.expect("prefill produced no sampled tokens");
        block_on(command_buffer.end_encoding().submit().wait_until_completed())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let ids: Vec<u32> = block_on(context.copyout(&buf)).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let first_token = ids.first().copied().unwrap_or(0) as u64;
        sampler.accept_token(first_token);
        Ok(first_token)
    }

    /// High-level follow-up turn prefill: tokenizes prompt and executes incremental prefill.
    pub fn prefill_followup_turn(
        &self,
        model: &LanguageModel<Metal>,
        state: &mut VlmState,
        prompt_text: &str,
        tokenizer: &Tokenizer,
        sampler: &mut VlmSampler,
        enable_thinking: bool,
    ) -> Result<u64, anyhow::Error> {
        let tokens = self.format_followup_tokens(prompt_text, tokenizer, enable_thinking)?;
        self.prefill_incremental_text(model, state, &tokens, sampler)
    }
}

#[cfg(test)]
mod render_tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn tools_preamble_lists_declarations() {
        let preamble = tools_system_preamble(&[
            r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}"#.to_string(),
            r#"{"type":"function","function":{"name":"get_time","parameters":{"type":"object"}}}"#.to_string(),
        ]);
        assert!(preamble.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>\n"));
        assert!(preamble.contains(r#""name":"get_weather""#));
        assert!(preamble.contains("<function=example_function_name>"));
        assert!(preamble.ends_with("do not tell the user about function calls\n</IMPORTANT>"));
    }

    #[uzu_test]
    fn assistant_text_renders_empty_think_block_and_content() {
        let msg = MultimodalMessage::new("assistant", Some("Hello there".to_string()));
        assert_eq!(render_assistant_text(&msg), "<think>\n\n</think>\n\nHello there");
    }

    #[uzu_test]
    fn assistant_text_prefers_reasoning_in_think_block() {
        let mut msg = MultimodalMessage::new("assistant", Some("Answer".to_string()));
        msg.reasoning_content = Some(" chain of thought ".to_string());
        assert_eq!(render_assistant_text(&msg), "<think>\nchain of thought\n</think>\n\nAnswer");
    }

    #[uzu_test]
    fn assistant_text_renders_tool_calls_after_content() {
        let mut msg = MultimodalMessage::new("assistant", Some("Checking now".to_string()));
        msg.tool_calls.push(MultimodalToolCall {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"Tokyo","days":3}"#.to_string(),
        });
        assert_eq!(
            render_assistant_text(&msg),
            "<think>\n\n</think>\n\nChecking now\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
        );
    }

    #[uzu_test]
    fn assistant_text_first_tool_call_without_content_has_no_leading_newlines() {
        let mut msg = MultimodalMessage::new("assistant", None);
        msg.tool_calls.push(MultimodalToolCall {
            name: "ping".to_string(),
            arguments: "{}".to_string(),
        });
        assert_eq!(
            render_assistant_text(&msg),
            "<think>\n\n</think>\n\n<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        );
    }

    #[uzu_test]
    fn assistant_text_second_tool_call_is_newline_separated() {
        let mut msg = MultimodalMessage::new("assistant", None);
        msg.tool_calls.push(MultimodalToolCall {
            name: "a".to_string(),
            arguments: "{}".to_string(),
        });
        msg.tool_calls.push(MultimodalToolCall {
            name: "b".to_string(),
            arguments: "{}".to_string(),
        });
        let text = render_assistant_text(&msg);
        assert!(text.contains("</function>\n</tool_call>\n<tool_call>\n<function=b>"));
    }

    #[uzu_test]
    fn assistant_text_renders_structured_arguments_as_json() {
        let mut msg = MultimodalMessage::new("assistant", None);
        msg.tool_calls.push(MultimodalToolCall {
            name: "write".to_string(),
            arguments: r#"{"path":"a.rs","content":{"lines":["x","y"]}}"#.to_string(),
        });
        let text = render_assistant_text(&msg);
        assert!(text.contains("<parameter=content>\n{\"lines\":[\"x\",\"y\"]}\n</parameter>"));
    }

    #[uzu_test]
    fn tool_responses_join_into_one_user_turn() {
        let text = render_tool_response_text(&["sunny", "{\"temp\": 21}"]);
        assert_eq!(text, "<tool_response>\nsunny\n</tool_response>\n<tool_response>\n{\"temp\": 21}\n</tool_response>");
    }

    #[uzu_test]
    fn normalize_turns_combines_preamble_with_leading_system() {
        let messages = vec![
            MultimodalMessage::new("system", Some("Be terse.".to_string())),
            MultimodalMessage::new("user", Some("hi".to_string())),
        ];
        let turns = normalize_turns(&messages, Some("PREAMBLE"));
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, "system");
        assert_eq!(turns[0].text.as_deref(), Some("PREAMBLE\n\nBe terse."));
        assert_eq!(turns[1].role, "user");
    }

    #[uzu_test]
    fn normalize_turns_preamble_only_without_system() {
        let messages = vec![MultimodalMessage::new("user", Some("hi".to_string()))];
        let turns = normalize_turns(&messages, Some("PREAMBLE"));
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, "system");
        assert_eq!(turns[0].text.as_deref(), Some("PREAMBLE"));
    }

    #[uzu_test]
    fn normalize_turns_groups_consecutive_tool_messages() {
        let mut assistant = MultimodalMessage::new("assistant", None);
        assistant.tool_calls.push(MultimodalToolCall {
            name: "f".to_string(),
            arguments: "{}".to_string(),
        });
        let messages = vec![
            MultimodalMessage::new("user", Some("run".to_string())),
            assistant,
            MultimodalMessage::new("tool", Some("one".to_string())),
            MultimodalMessage::new("tool", Some("two".to_string())),
        ];
        let turns = normalize_turns(&messages, None);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[2].role, "user");
        assert_eq!(
            turns[2].text.as_deref(),
            Some("<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>")
        );
    }
}

#[cfg(test)]
mod sampler_tests {
    use uzu_engine_macros::uzu_test;

    use super::*;
    use crate::encodable_block::sampling::SamplingMethod;

    #[uzu_test]
    fn sampler_builds_engine_sampling_methods() {
        // Plain greedy (no temperature, no grammar).
        let sampler = VlmSampler::new(VlmSamplingConfig::default());
        assert_eq!(sampler.sampling_method(None), SamplingMethod::Greedy);

        // Stochastic with clamped temperature and full parameter passthrough.
        let sampler = VlmSampler::new(VlmSamplingConfig {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.05),
            repetition_penalty: Some(1.3),
            seed: Some(42),
            ..Default::default()
        });
        assert_eq!(
            sampler.sampling_method(Some(2048)),
            SamplingMethod::Stochastic {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: Some(0.9),
                min_p: Some(0.05),
                repetition_penalty: Some(1.3),
                suffix_repetition_length: Some(2048),
            }
        );

        // Seeds are deterministic per position and differ across positions.
        assert_eq!(sampler.sampling_seed(10), sampler.sampling_seed(10));
        assert_ne!(sampler.sampling_seed(10), sampler.sampling_seed(11));
    }

    #[uzu_test]
    fn vocab_padding_bitmask_masks_padded_rows() {
        // Partial word: 65 valid ids over 70 slots.
        assert_eq!(vocab_padding_bitmask(70, 65), [u32::MAX, u32::MAX, 0b1]);

        // Word-aligned limit.
        assert_eq!(vocab_padding_bitmask(96, 64), [u32::MAX, u32::MAX, 0]);

        // No padding to mask.
        assert!(vocab_padding_bitmask(4, 4).is_empty());
    }
}
