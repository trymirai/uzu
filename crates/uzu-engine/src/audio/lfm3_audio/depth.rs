//! MaskGIT depth decoder: one speech frame (12 codebooks) from a backbone
//! hidden state, as a bidirectional 3-layer LFM3 trunk over 12 slots.
//!
//! Mirrors `mlx_vlm/models/lfm3_audio/depth.py` and the checkpoint reference
//! (`modeling_lfm3_audio.py`): per-codebook embedding tables double as tied
//! output heads; codebooks are revealed in a random order over the MaskGIT
//! passes, each pass scoring the still-masked codebooks with a bidirectional
//! forward. The checkpoint exports dead post norms for this trunk
//! (`use_sandwich_norm: false`); the converter drops them.

use anyhow::{ensure, Context, Result};

use crate::audio::liquid_codec::{bf16_round, round_bf16};
use crate::vision::tower::{matmul_nt, LinearWeight, SafetensorLoader};

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub struct DepthLayerWeights {
    pub(crate) q_proj: LinearWeight,
    pub(crate) k_proj: LinearWeight,
    pub(crate) v_proj: LinearWeight,
    pub(crate) o_proj: LinearWeight,
    pub(crate) gate_proj: LinearWeight,
    pub(crate) w1: LinearWeight,
    pub(crate) w2: LinearWeight,
    pub(crate) w3: LinearWeight,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
    pub(crate) pre_mixer_norm: Vec<f32>,
    pub(crate) pre_mlp_norm: Vec<f32>,
}

pub struct CodebookEmbedding {
    /// `[codebook_size, hidden]`, doubles as the tied output head.
    pub(crate) embedding: Vec<f32>,
    /// Head RMSNorm weight (`embedding_norm`).
    pub(crate) norm: Vec<f32>,
}

pub struct DepthWeights {
    pub linear: LinearWeight,
    pub mask: Vec<f32>,
    pub layers: Vec<DepthLayerWeights>,
    pub embeddings: Vec<CodebookEmbedding>,
    pub embedding_norm: Vec<f32>,
}

impl DepthWeights {
    pub(crate) fn load(loader: &mut SafetensorLoader, layers: usize, codebooks: usize) -> Result<Self> {
        let read_linear = |loader: &mut SafetensorLoader, base: &str| -> Result<LinearWeight> {
            let (w, shape) = loader
                .read_tensor_f32(base)
                .with_context(|| format!("Reading {base}"))?;
            ensure!(shape.len() == 2, "expected 2D weight for {base}, got {shape:?}");
            let bias_key = format!("{base}.biases");
            let bias = loader.contains(&bias_key).then(|| loader.read_tensor_f32(&bias_key)).transpose()?.map(|(b, _)| b);
            Ok(LinearWeight { weight: w, bias, in_features: shape[1], out_features: shape[0] })
        };
        let linear = read_linear(loader, "audio.depth.linear.weights")?;
        let (mask, _) = loader.read_tensor_f32("audio.depth.mask")?;

        let mut layer_weights = Vec::with_capacity(layers);
        for layer in 0..layers {
            let base = format!("audio.depth.model.layers.{layer}");
            layer_weights.push(DepthLayerWeights {
                q_proj: read_linear(loader, &format!("{base}.q_proj.weights"))?,
                k_proj: read_linear(loader, &format!("{base}.k_proj.weights"))?,
                v_proj: read_linear(loader, &format!("{base}.v_proj.weights"))?,
                o_proj: read_linear(loader, &format!("{base}.o_proj.weights"))?,
                gate_proj: read_linear(loader, &format!("{base}.gate_proj.weights"))?,
                w1: read_linear(loader, &format!("{base}.w1.weights"))?,
                w2: read_linear(loader, &format!("{base}.w2.weights"))?,
                w3: read_linear(loader, &format!("{base}.w3.weights"))?,
                query_norm: loader.read_tensor_f32(&format!("{base}.query_norm.scales"))?.0,
                key_norm: loader.read_tensor_f32(&format!("{base}.key_norm.scales"))?.0,
                pre_mixer_norm: loader.read_tensor_f32(&format!("{base}.pre_mixer_norm.scales"))?.0,
                pre_mlp_norm: loader.read_tensor_f32(&format!("{base}.pre_mlp_norm.scales"))?.0,
            });
        }

        let mut embeddings = Vec::with_capacity(codebooks);
        for codebook in 0..codebooks {
            let (embedding, shape) = loader.read_tensor_f32(&format!("audio.depth.embeddings.{codebook}.embedding"))?;
            ensure!(shape.len() == 2, "embedding {codebook} must be 2D, got {shape:?}");
            let (norm, _) = loader.read_tensor_f32(&format!("audio.depth.embeddings.{codebook}.norm"))?;
            embeddings.push(CodebookEmbedding { embedding, norm });
        }
        let (embedding_norm, _) = loader.read_tensor_f32("audio.depth.embedding_norm")?;
        Ok(Self {
            linear,
            mask,
            layers: layer_weights,
            embeddings,
            embedding_norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

/// MLX `nn.RMSNorm`: normalize in f32, scale in f32, round once to bf16.
fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    for row in x.chunks_exact_mut(dim) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for (d, v) in row.iter_mut().enumerate() {
            *v = bf16_round(*v * inv * weight[d]);
        }
    }
}

fn linear(w: &LinearWeight, x: &[f32], tokens: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * w.out_features];
    matmul_nt(x, tokens, w.in_features, &w.weight, w.out_features, w.bias.as_deref(), &mut out);
    round_bf16(&mut out);
    out
}

/// Bidirectional trunk forward: `hidden` [2048] -> logits [target_slots.len(), V] (or [codebooks, V] when `target_slots` is `None`).
fn trunk_forward(
    weights: &DepthWeights,
    config: &super::DepthConfig,
    hidden: &[f32],
    codes: &[i64],
    masked: &[bool],
    target_slots: Option<&[usize]>,
) -> Vec<f32> {
    let slots = codes.len();
    let hidden_dim = config.hidden;
    let eps = config.eps as f32;

    // depth_linear + per-slot code embeddings or the mask embedding.
    let projected = linear(&weights.linear, hidden, 1);
    let mut x = vec![0.0f32; slots * hidden_dim];
    for slot in 0..slots {
        let source = &projected[slot * hidden_dim..(slot + 1) * hidden_dim];
        for (d, value) in source.iter().enumerate() {
            let base = if masked[slot] {
                weights.mask[d]
            } else {
                weights.embeddings[slot].embedding[codes[slot] as usize * hidden_dim + d]
            };
            x[slot * hidden_dim + d] = bf16_round(value + base);
        }
    }

    let rope = rope_tables(slots, config.head_dim, config.rope_theta);

    for layer in &weights.layers {
        let mut normed = x.clone();
        rms_norm(&mut normed, &layer.pre_mixer_norm, eps);
        let attended = attention(layer, config, &normed, slots, &rope);
        for d in 0..slots * hidden_dim {
            x[d] = bf16_round(x[d] + attended[d]);
        }

        let mut normed = x.clone();
        rms_norm(&mut normed, &layer.pre_mlp_norm, eps);
        let gate = linear(&layer.w1, &normed, slots);
        let up = linear(&layer.w3, &normed, slots);
        let mut mlp_in = vec![0.0f32; gate.len()];
        for ((g, u), out) in gate.iter().zip(&up).zip(&mut mlp_in) {
            let silu = g / (1.0 + (-g).exp());
            *out = bf16_round(silu * u);
        }
        let mlp_out = linear(&layer.w2, &mlp_in, slots);
        for d in 0..slots * hidden_dim {
            x[d] = bf16_round(x[d] + mlp_out[d]);
        }
    }

    rms_norm(&mut x, &weights.embedding_norm, eps);

    // Per-codebook tied heads: HeadRMSNorm (f32 normalize + weight, round once)
    // then the embedding table as a linear head.
    let vocab = weights.embeddings[0].embedding.len() / hidden_dim;
    let out_slots = target_slots.map_or(slots, |s| s.len());
    let mut logits = vec![0.0f32; out_slots * vocab];
    let mut normed_row = vec![0.0f32; hidden_dim];
    for (out_idx, slot) in (0..out_slots).map(|i| (i, target_slots.map_or(i, |s| s[i]))) {
        let row = &x[slot * hidden_dim..(slot + 1) * hidden_dim];
        let norm = &weights.embeddings[slot].norm;
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();
        for (d, value) in row.iter().enumerate() {
            normed_row[d] = value * inv * norm[d];
        }
        round_bf16(&mut normed_row);
        matmul_nt(
            &normed_row,
            1,
            hidden_dim,
            &weights.embeddings[slot].embedding,
            vocab,
            None,
            &mut logits[out_idx * vocab..(out_idx + 1) * vocab],
        );
    }
    round_bf16(&mut logits);
    logits
}

/// cos/sin per slot: `[slots, head_dim/2]` of (cos, sin) — MLX concatenates
/// the half rows, so both rotated halves share the same angle per frequency.
fn rope_tables(slots: usize, head_dim: usize, theta: f64) -> Vec<(f32, f32)> {
    let angles = head_dim / 2;
    let mut tables = Vec::with_capacity(slots * angles);
    for position in 0..slots {
        for k in 0..angles {
            let exponent = (2 * k) as f64 / head_dim as f64;
            let angle = position as f64 * theta.powf(-exponent);
            tables.push((angle.cos() as f32, angle.sin() as f32));
        }
    }
    tables
}

/// `[a | b] -> [a·cos − b·sin | b·cos + a·sin]` per head, bf16 rounding.
fn apply_rope(tensor: &mut [f32], slots: usize, head_dim: usize, rope: &[(f32, f32)]) {
    let half = head_dim / 2;
    let flat = tensor.len() / (slots * head_dim);
    for slot in 0..slots {
        for head in 0..flat {
            let base = slot * flat * head_dim + head * head_dim;
            for offset in 0..half {
                let (c, s) = rope[slot * half + offset];
                let a = tensor[base + offset];
                let b = tensor[base + half + offset];
                tensor[base + offset] = bf16_round(a * c - b * s);
                tensor[base + half + offset] = bf16_round(b * c + a * s);
            }
        }
    }
}

/// LFM3 attention: q/k RMSNorm, full rotary, bidirectional softmax, per-head
/// sigmoid gate. Linear layouts: q/gate [slots, heads*head_dim / heads],
/// k/v [slots, kv_heads*head_dim].
fn attention(layer: &DepthLayerWeights, config: &super::DepthConfig, x: &[f32], slots: usize, rope: &[(f32, f32)]) -> Vec<f32> {
    let heads = config.heads;
    let kv_heads = config.kv_heads;
    let head_dim = config.head_dim;
    let eps = config.eps as f32;
    let scale = (head_dim as f32).sqrt().recip();
    let group = heads / kv_heads;

    let mut q = linear(&layer.q_proj, x, slots);
    let mut k = linear(&layer.k_proj, x, slots);
    let v = linear(&layer.v_proj, x, slots);
    let gate = linear(&layer.gate_proj, x, slots);

    rms_norm(&mut q, &layer.query_norm, eps);
    rms_norm(&mut k, &layer.key_norm, eps);
    apply_rope(&mut q, slots, head_dim, rope);
    apply_rope(&mut k, slots, head_dim, rope);

    let mut scores = vec![0.0f32; heads * slots * slots];
    for head in 0..heads {
        let kv_head = head / group;
        for row in 0..slots {
            let q_row = &q[row * heads * head_dim + head * head_dim..][..head_dim];
            for col in 0..slots {
                let k_row = &k[col * kv_heads * head_dim + kv_head * head_dim..][..head_dim];
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += q_row[d] * k_row[d];
                }
                scores[head * slots * slots + row * slots + col] = sum * scale;
            }
        }
    }
    // Softmax in f32 over the (bidirectional, full) slot axis.
    for head in 0..heads {
        for row in 0..slots {
            let base = head * slots * slots + row * slots;
            let row_scores = &mut scores[base..base + slots];
            let max = row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row_scores.iter().map(|s| (s - max).exp()).sum();
            for s in row_scores.iter_mut() {
                *s = (*s - max).exp() / exp_sum;
            }
        }
    }
    // Weighted value sum with the per-head sigmoid gate folded in.
    let mut attended = vec![0.0f32; slots * heads * head_dim];
    for row in 0..slots {
        for head in 0..heads {
            let g = gate[row * heads + head];
            let gate_value = 1.0 / (1.0 + (-g).exp());
            let kv_head = head / group;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for col in 0..slots {
                    let weight = scores[head * slots * slots + row * slots + col];
                    sum += weight * v[col * kv_heads * head_dim + kv_head * head_dim + d];
                }
                attended[row * heads * head_dim + head * head_dim + d] = bf16_round(sum * gate_value);
            }
        }
    }
    linear(&layer.o_proj, &attended, slots)
}

// ---------------------------------------------------------------------------
// MaskGIT sampling
// ---------------------------------------------------------------------------

struct SampleScratch {
    scores: Vec<f32>,
    topk: Vec<f32>,
    penalized: Vec<bool>,
}

impl SampleScratch {
    fn new(vocab: usize) -> Self {
        Self {
            scores: vec![0.0; vocab],
            topk: vec![0.0; vocab],
            penalized: vec![false; vocab],
        }
    }
}

/// The depth decoder bound to its weights and config.
pub struct DepthDecoder {
    weights: DepthWeights,
    config: super::DepthConfig,
    rng_state: u64,
    scratch: SampleScratch,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    metal: Option<super::depth_metal::MetalDepth>,
}

impl DepthDecoder {
    pub fn new(weights: DepthWeights, config: super::DepthConfig, seed: u64) -> Self {
        let vocab = weights.embeddings[0].embedding.len() / config.hidden;
        Self {
            weights,
            config,
            rng_state: seed | 1,
            scratch: SampleScratch::new(vocab),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal: None,
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn with_metal(
        weights: DepthWeights,
        config: super::DepthConfig,
        seed: u64,
        context: &crate::backends::metal::context::MetalContext,
    ) -> Result<Self> {
        let vocab = weights.embeddings[0].embedding.len() / config.hidden;
        let metal = super::depth_metal::MetalDepth::new(context, &weights, &config)?;
        Ok(Self {
            weights,
            config,
            rng_state: seed | 1,
            scratch: SampleScratch::new(vocab),
            metal: Some(metal),
        })
    }

    /// Forward logits for one frame: `hidden` [2048], `codes`/`masked` [K].
    pub fn logits(&mut self, hidden: &[f32], codes: &[i64], masked: &[bool]) -> Vec<f32> {
        trunk_forward(&self.weights, &self.config, hidden, codes, masked, None)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn logits_metal(
        &mut self,
        context: &crate::backends::metal::context::MetalContext,
        hidden: &[f32],
        codes: &[i64],
        masked: &[bool],
    ) -> Result<Vec<f32>> {
        let metal = self.metal.as_mut().ok_or_else(|| anyhow::anyhow!("MetalDepth not initialized"))?;
        let all_slots: Vec<usize> = (0..codes.len()).collect();
        metal.forward_pass(context, Some(hidden), codes, masked, &all_slots)
    }

    /// Frame embedding: sum of the codebook embeddings for `codes` [K].
    pub fn frame_embedding(&self, codes: &[i64]) -> Vec<f32> {
        let hidden_dim = self.config.hidden;
        let mut out = vec![0.0f32; hidden_dim];
        for (slot, &code) in codes.iter().enumerate() {
            let table = &self.weights.embeddings[slot].embedding;
            for d in 0..hidden_dim {
                out[d] = bf16_round(out[d] + table[code as usize * hidden_dim + d]);
            }
        }
        out
    }

    fn next_random(&mut self) -> u64 {
        let mut state = self.rng_state;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.rng_state = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn sample_frame_gpu(
        &mut self,
        context: &crate::backends::metal::context::MetalContext,
        hidden_gpu: &<crate::backends::metal::Metal as crate::backends::common::Backend>::GlobalGpuBuffer,
        schedule: &[usize],
        temperature: Option<f32>,
        top_k: Option<usize>,
        repetition_penalty: Option<f32>,
        history: &[Vec<i64>],
        order: Option<&[usize]>,
    ) -> Result<Vec<i64>> {
        let codebooks = self.weights.embeddings.len();
        let vocab = self.weights.embeddings[0].embedding.len() / self.config.hidden;
        let order = self.resolve_order(codebooks, order)?;
        let metal = self.metal.as_mut().ok_or_else(|| anyhow::anyhow!("MetalDepth not initialized"))?;

        let mut codes = vec![0i64; codebooks];
        let mut masked = vec![true; codebooks];
        let mut start = 0;
        for &size in schedule {
            let picks = &order[start..(start + size).min(order.len())];
            let logits = metal.forward_pass_gpu(
                context,
                (start == 0).then_some(hidden_gpu),
                &codes,
                &masked,
                picks,
            )?;
            for (pick_idx, &slot) in picks.iter().enumerate() {
                codes[slot] = sample_logits(
                    &logits[pick_idx * vocab..(pick_idx + 1) * vocab],
                    temperature,
                    top_k,
                    repetition_penalty,
                    history,
                    slot,
                    &mut self.rng_state,
                    &mut self.scratch,
                ) as i64;
            }
            for &slot in picks {
                masked[slot] = false;
            }
            start += size;
        }
        Ok(codes)
    }

    fn resolve_order(&mut self, codebooks: usize, order: Option<&[usize]>) -> Result<Vec<usize>> {
        match order {
            Some(order) => {
                ensure!(order.len() == codebooks, "reveal order must cover every codebook");
                Ok(order.to_vec())
            },
            None => {
                let mut order: Vec<usize> = (0..codebooks).collect();
                for index in (1..codebooks).rev() {
                    let swap = (self.next_random() % (index as u64 + 1)) as usize;
                    order.swap(index, swap);
                }
                Ok(order)
            },
        }
    }

    /// MaskGIT one frame: raw codes [K] (before EOS normalization).
    ///
    /// `order`: permutation of the codebooks revealing which slots each pass
    /// uncovers; `None` draws one from the decoder RNG. `history`: previous
    /// frames (per codebook) for the repetition penalty.
    pub fn sample_frame(
        &mut self,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        context: Option<&crate::backends::metal::context::MetalContext>,
        hidden: &[f32],
        schedule: &[usize],
        temperature: Option<f32>,
        top_k: Option<usize>,
        repetition_penalty: Option<f32>,
        history: &[Vec<i64>],
        order: Option<&[usize]>,
    ) -> Result<Vec<i64>> {
        let codebooks = self.weights.embeddings.len();
        let vocab = self.weights.embeddings[0].embedding.len() / self.config.hidden;
        let order = self.resolve_order(codebooks, order)?;

        let mut codes = vec![0i64; codebooks];
        let mut masked = vec![true; codebooks];
        let mut start = 0;
        for &size in schedule {
            let picks = &order[start..(start + size).min(order.len())];
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let logits = if let (Some(metal), Some(ctx)) = (self.metal.as_mut(), context) {
                metal.forward_pass(ctx, (start == 0).then_some(hidden), &codes, &masked, picks)?
            } else {
                trunk_forward(&self.weights, &self.config, hidden, &codes, &masked, Some(picks))
            };
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let logits = trunk_forward(&self.weights, &self.config, hidden, &codes, &masked, Some(picks));
            for (pick_idx, &slot) in picks.iter().enumerate() {
                codes[slot] = sample_logits(
                    &logits[pick_idx * vocab..(pick_idx + 1) * vocab],
                    temperature,
                    top_k,
                    repetition_penalty,
                    history,
                    slot,
                    &mut self.rng_state,
                    &mut self.scratch,
                ) as i64;
            }
            for &slot in picks {
                masked[slot] = false;
            }
            start += size;
        }
        Ok(codes)
    }

    /// A frame with any EOS code becomes all EOS.
    pub fn normalize_eos(&self, codes: &mut [i64], eos_code: i64) {
        if codes.contains(&eos_code) {
            codes.fill(eos_code);
        }
    }
}

/// Reference `_sample`: repetition penalty, temperature, top-k, softmax;
/// greedy when `temperature` is 0/None without top_k.
fn sample_logits(
    logits: &[f32],
    temperature: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    history: &[Vec<i64>],
    slot: usize,
    rng_state: &mut u64,
    scratch: &mut SampleScratch,
) -> usize {
    let scores = &mut scratch.scores[..logits.len()];
    scores.copy_from_slice(logits);
    if let Some(penalty) = repetition_penalty.filter(|p| *p != 1.0) {
        for frame in history {
            let id = frame[slot];
            if id < 0 {
                continue;
            }
            let idx = id as usize;
            if idx < scores.len() && !scratch.penalized[idx] {
                scratch.penalized[idx] = true;
                let score = &mut scores[idx];
                *score = if *score < 0.0 { *score * penalty } else { *score / penalty };
            }
        }
        for frame in history {
            let id = frame[slot];
            if id >= 0 && (id as usize) < scratch.penalized.len() {
                scratch.penalized[id as usize] = false;
            }
        }
    }
    let temperature = temperature.filter(|t| *t != 0.0);
    if temperature.is_none() && top_k.is_none() {
        return argmax(scores);
    }
    if let Some(temperature) = temperature {
        for score in scores.iter_mut() {
            *score /= temperature;
        }
    }
    if let Some(top_k) = top_k.filter(|k| *k > 0 && *k < scores.len()) {
        let threshold = top_k_threshold(scores, top_k, &mut scratch.topk[..scores.len()]);
        for score in scores.iter_mut() {
            if *score < threshold {
                *score = f32::NEG_INFINITY;
            }
        }
    }
    // Softmax + multinomial in-place on `scores`.
    let mut best_idx = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (idx, &s) in scores.iter().enumerate() {
        if s > max {
            max = s;
            best_idx = idx;
        }
    }
    let mut total = 0.0f32;
    for s in scores.iter_mut() {
        let e = (*s - max).exp();
        *s = e;
        total += e;
    }
    let mut draw = pseudo_uniform(rng_state) * total;
    for (index, &mass) in scores.iter().enumerate() {
        draw -= mass;
        if draw <= 0.0 {
            return index;
        }
    }
    best_idx
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best
}

fn pseudo_uniform(rng_state: &mut u64) -> f32 {
    let mut state = *rng_state;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    *rng_state = state;
    (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f32 / (1u64 << 53) as f32
}

/// Smallest score among the top `k` (the cutoff of the top-k filter).
fn top_k_threshold(scores: &[f32], k: usize, scratch: &mut [f32]) -> f32 {
    scratch.copy_from_slice(scores);
    let (_, nth, _) = scratch.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    *nth
}
