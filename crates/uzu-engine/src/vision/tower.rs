use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

use super::{
    config::{CanvasConfig, VisionConfig},
    image_processing::{PreprocessedVision, nearest_position_ids, window_index},
};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Fallback or native matmul: C = A * B^T + bias
/// A: [M, K], B: [N, K], C: [M, N]
pub fn matmul_nt(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            101, // CblasRowMajor
            111, // CblasNoTrans
            112, // CblasTrans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                let a_row = &a[i * k..(i + 1) * k];
                let b_row = &b[j * k..(j + 1) * k];
                for idx in 0..k {
                    sum += a_row[idx] * b_row[idx];
                }
                c[i * n + j] = sum;
            }
        }
    }

    if let Some(bias) = bias {
        for i in 0..m {
            let row_offset = i * n;
            for j in 0..n {
                c[row_offset + j] += bias[j];
            }
        }
    }
}

/// Matrix multiplication: C = A * B
/// A: [M, K], B: [K, N], C: [M, N]
pub fn matmul_nn(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            101, // CblasRowMajor
            111, // CblasNoTrans
            111, // CblasNoTrans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for idx in 0..k {
                    sum += a[i * k + idx] * b[idx * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6_f32;
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

#[inline(always)]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    hidden_size: usize,
    out: &mut [f32],
) {
    let num_tokens = x.len() / hidden_size;
    assert_eq!(x.len(), num_tokens * hidden_size);
    assert_eq!(out.len(), x.len());
    assert_eq!(gamma.len(), hidden_size);

    for i in 0..num_tokens {
        let offset = i * hidden_size;
        let slice = &x[offset..offset + hidden_size];
        let out_slice = &mut out[offset..offset + hidden_size];

        let mut sum = 0.0f32;
        for &val in slice {
            sum += val;
        }
        let mean = sum / hidden_size as f32;

        let mut var_sum = 0.0f32;
        for &val in slice {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        if let Some(beta) = beta {
            for j in 0..hidden_size {
                out_slice[j] = (slice[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        } else {
            for j in 0..hidden_size {
                out_slice[j] = (slice[j] - mean) * inv_std * gamma[j];
            }
        }
    }
}

pub struct LinearWeight {
    pub weight: Vec<f32>, // [out_features, in_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearWeight {
    pub fn forward(
        &self,
        x: &[f32],
        num_tokens: usize,
        out: &mut [f32],
    ) {
        matmul_nt(x, num_tokens, self.in_features, &self.weight, self.out_features, self.bias.as_deref(), out);
    }
}

pub struct LayerNormWeights {
    pub gamma: Vec<f32>,
    pub beta: Option<Vec<f32>>,
}

pub struct SiglipAttentionWeights {
    pub qkv_proj: LinearWeight,
    pub out_proj: LinearWeight,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
}

impl SiglipAttentionWeights {
    pub fn forward(
        &self,
        x: &[f32],
        segment_lengths: &[usize],
        num_tokens: usize,
        out: &mut [f32],
    ) {
        let mut qkv = vec![0.0f32; num_tokens * self.hidden_size * 3];
        self.qkv_proj.forward(x, num_tokens, &mut qkv);

        let mut attended = vec![0.0f32; num_tokens * self.hidden_size];
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut start = 0;
        let qkv_stride = 3 * self.hidden_size;
        for &seg_len in segment_lengths {
            if seg_len == 0 {
                continue;
            }
            // Process attention for this segment
            for h in 0..self.num_heads {
                // Extract Q_h, K_h, V_h for this segment: shape [seg_len, head_dim]
                let mut q_h = vec![0.0f32; seg_len * self.head_dim];
                let mut k_h = vec![0.0f32; seg_len * self.head_dim];
                let mut v_h = vec![0.0f32; seg_len * self.head_dim];

                for t in 0..seg_len {
                    let token_idx = start + t;
                    let base = token_idx * qkv_stride;
                    let target_offset = t * self.head_dim;
                    let head_off = h * self.head_dim;
                    q_h[target_offset..target_offset + self.head_dim]
                        .copy_from_slice(&qkv[base + head_off..base + head_off + self.head_dim]);
                    k_h[target_offset..target_offset + self.head_dim].copy_from_slice(
                        &qkv[base + self.hidden_size + head_off..base + self.hidden_size + head_off + self.head_dim],
                    );
                    v_h[target_offset..target_offset + self.head_dim].copy_from_slice(
                        &qkv[base + 2 * self.hidden_size + head_off
                            ..base + 2 * self.hidden_size + head_off + self.head_dim],
                    );
                }

                // S = Q_h * K_h^T * scale: [seg_len, seg_len]
                let mut scores = vec![0.0f32; seg_len * seg_len];
                matmul_nt(&q_h, seg_len, self.head_dim, &k_h, seg_len, None, &mut scores);

                for val in &mut scores {
                    *val *= scale;
                }

                // Softmax each row
                for r in 0..seg_len {
                    let row = &mut scores[r * seg_len..(r + 1) * seg_len];
                    let mut max_val = f32::NEG_INFINITY;
                    for &val in row.iter() {
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for val in row.iter_mut() {
                        *val = (*val - max_val).exp();
                        sum_exp += *val;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for val in row.iter_mut() {
                        *val *= inv_sum;
                    }
                }

                // Context = scores * V_h: [seg_len, seg_len] * [seg_len, head_dim] -> [seg_len, head_dim]
                let mut context = vec![0.0f32; seg_len * self.head_dim];
                matmul_nn(&scores, seg_len, seg_len, &v_h, self.head_dim, &mut context);

                // Scatter back to attended
                for t in 0..seg_len {
                    let token_idx = start + t;
                    let head_offset = token_idx * self.hidden_size + h * self.head_dim;
                    let src_offset = t * self.head_dim;
                    attended[head_offset..head_offset + self.head_dim]
                        .copy_from_slice(&context[src_offset..src_offset + self.head_dim]);
                }
            }
            start += seg_len;
        }

        self.out_proj.forward(&attended, num_tokens, out);
    }
}

pub struct SiglipMLPWeights {
    pub fc1: LinearWeight,
    pub fc2: LinearWeight,
}

impl SiglipMLPWeights {
    pub fn forward(
        &self,
        x: &[f32],
        num_tokens: usize,
        out: &mut [f32],
    ) {
        let mut hidden = vec![0.0f32; num_tokens * self.fc1.out_features];
        self.fc1.forward(x, num_tokens, &mut hidden);
        for val in &mut hidden {
            *val = gelu_tanh(*val);
        }
        self.fc2.forward(&hidden, num_tokens, out);
    }
}

pub struct EncoderLayerWeights {
    pub layer_norm1: LayerNormWeights,
    pub self_attn: SiglipAttentionWeights,
    pub layer_norm2: LayerNormWeights,
    pub mlp: SiglipMLPWeights,
    pub hidden_size: usize,
    pub eps: f32,
}

impl EncoderLayerWeights {
    pub fn forward(
        &self,
        x: &mut [f32],
        segment_lengths: &[usize],
        num_tokens: usize,
    ) {
        let mut normed = vec![0.0f32; x.len()];
        layer_norm(
            x,
            &self.layer_norm1.gamma,
            self.layer_norm1.beta.as_deref(),
            self.eps,
            self.hidden_size,
            &mut normed,
        );

        let mut attn_out = vec![0.0f32; x.len()];
        self.self_attn.forward(&normed, segment_lengths, num_tokens, &mut attn_out);
        for i in 0..x.len() {
            x[i] += attn_out[i];
        }

        layer_norm(
            x,
            &self.layer_norm2.gamma,
            self.layer_norm2.beta.as_deref(),
            self.eps,
            self.hidden_size,
            &mut normed,
        );

        let mut mlp_out = vec![0.0f32; x.len()];
        self.mlp.forward(&normed, num_tokens, &mut mlp_out);
        for i in 0..x.len() {
            x[i] += mlp_out[i];
        }
    }
}

pub struct WindowAttentionMergerWeights {
    pub layer_norm1: LayerNormWeights,
    pub self_attn: SiglipAttentionWeights,
    pub pre_norm: LayerNormWeights,
    pub linear_1: LinearWeight,
    pub linear_2: LinearWeight,
    pub window: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

impl WindowAttentionMergerWeights {
    pub fn forward(
        &self,
        x: &[f32],
        grids: &[(usize, usize, usize)],
    ) -> Vec<f32> {
        let mut indices = Vec::new();
        let mut offset = 0;
        for &(_, h, w) in grids {
            indices.extend(window_index(h, w, self.window, offset));
            offset += h * w;
        }

        let num_tokens = x.len() / self.hidden_size;
        let window_size = self.window * self.window;
        assert_eq!(indices.len(), num_tokens);

        // Gather x by indices
        let mut x_gathered = vec![0.0f32; x.len()];
        for (i, &idx) in indices.iter().enumerate() {
            let src = idx * self.hidden_size;
            let dst = i * self.hidden_size;
            x_gathered[dst..dst + self.hidden_size].copy_from_slice(&x[src..src + self.hidden_size]);
        }

        // LayerNorm1 on x, then gather
        let mut normed_x = vec![0.0f32; x.len()];
        layer_norm(
            x,
            &self.layer_norm1.gamma,
            self.layer_norm1.beta.as_deref(),
            self.eps,
            self.hidden_size,
            &mut normed_x,
        );
        let mut normed_gathered = vec![0.0f32; x.len()];
        for (i, &idx) in indices.iter().enumerate() {
            let src = idx * self.hidden_size;
            let dst = i * self.hidden_size;
            normed_gathered[dst..dst + self.hidden_size].copy_from_slice(&normed_x[src..src + self.hidden_size]);
        }

        // Window self attention: segments are each of length window_size = 4
        let num_windows = num_tokens / window_size;
        let segments = vec![window_size; num_windows];
        let mut attended = vec![0.0f32; x.len()];
        self.self_attn.forward(&normed_gathered, &segments, num_tokens, &mut attended);

        // h = x_gathered + attended
        let mut h = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            h[i] = x_gathered[i] + attended[i];
        }

        // Patch residual = mean over window
        let mut patch_residual = vec![0.0f32; num_windows * self.hidden_size];
        for w in 0..num_windows {
            let w_offset = w * window_size * self.hidden_size;
            let dst_offset = w * self.hidden_size;
            for d in 0..self.hidden_size {
                let mut sum = 0.0f32;
                for p in 0..window_size {
                    sum += h[w_offset + p * self.hidden_size + d];
                }
                patch_residual[dst_offset + d] = sum / (window_size as f32);
            }
        }

        // pre_norm(flat): flat has shape [num_windows, window_size * hidden_size]
        let flat_dim = window_size * self.hidden_size;
        let mut pre_normed = vec![0.0f32; num_windows * flat_dim];
        layer_norm(&h, &self.pre_norm.gamma, self.pre_norm.beta.as_deref(), self.eps, flat_dim, &mut pre_normed);

        let mut out1 = vec![0.0f32; num_windows * self.linear_1.out_features];
        self.linear_1.forward(&pre_normed, num_windows, &mut out1);
        for val in &mut out1 {
            *val = gelu_tanh(*val);
        }

        let mut out2 = vec![0.0f32; num_windows * self.linear_2.out_features];
        self.linear_2.forward(&out1, num_windows, &mut out2);

        // Add patch residual
        assert_eq!(out2.len(), patch_residual.len());
        for i in 0..out2.len() {
            out2[i] += patch_residual[i];
        }

        out2
    }
}

pub struct DownsampleMLPWeights {
    pub pre_norm: LayerNormWeights,
    pub linear_1: LinearWeight,
    pub linear_2: LinearWeight,
    pub eps: f32,
}

impl DownsampleMLPWeights {
    pub fn forward(
        &self,
        x: &[f32],
        num_tokens: usize,
        out: &mut [f32],
    ) {
        let in_features = self.linear_1.in_features;
        let mut pre_normed = vec![0.0f32; num_tokens * in_features];
        layer_norm(x, &self.pre_norm.gamma, self.pre_norm.beta.as_deref(), self.eps, in_features, &mut pre_normed);

        let mut out1 = vec![0.0f32; num_tokens * self.linear_1.out_features];
        self.linear_1.forward(&pre_normed, num_tokens, &mut out1);
        for val in &mut out1 {
            *val = gelu_erf(*val); // standard erf for DownsampleMLP
        }

        self.linear_2.forward(&out1, num_tokens, out);
    }
}

pub struct TopMergerWeights {
    pub mlp: DownsampleMLPWeights,
}

impl TopMergerWeights {
    pub fn forward(
        &self,
        x: &[f32],
        grids: &[(usize, usize, usize)],
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut all_outputs = Vec::new();
        let mut start = 0;

        for &(_, h, w) in grids {
            let count = h * w;
            let crop_x = &x[start * hidden_size..(start + count) * hidden_size];
            start += count;

            let h_half = h / 2;
            let w_half = w / 2;
            let out_tokens = h_half * w_half;
            let flat_dim = 4 * hidden_size;

            // Group: [h//2, 2, w//2, 2, embed_dim] -> permute -> [h//2, w//2, 2, 2, embed_dim]
            let mut flat = vec![0.0f32; out_tokens * flat_dim];
            for oh in 0..h_half {
                for ow in 0..w_half {
                    let out_idx = oh * w_half + ow;
                    let mut p_idx = 0;
                    for r in 0..2 {
                        for c in 0..2 {
                            let in_y = oh * 2 + r;
                            let in_x = ow * 2 + c;
                            let in_idx = in_y * w + in_x;
                            let src_start = in_idx * hidden_size;
                            let dst_start = out_idx * flat_dim + p_idx * hidden_size;
                            flat[dst_start..dst_start + hidden_size]
                                .copy_from_slice(&crop_x[src_start..src_start + hidden_size]);
                            p_idx += 1;
                        }
                    }
                }
            }

            let mut out = vec![0.0f32; out_tokens * self.mlp.linear_2.out_features];
            self.mlp.forward(&flat, out_tokens, &mut out);
            all_outputs.extend(out);
        }

        all_outputs
    }
}

pub struct VisionTower {
    pub vision_config: VisionConfig,
    pub canvas_config: CanvasConfig,
    pub patch_proj: LinearWeight,
    pub position_embedding: Vec<f32>, // [4900, 1152]
    pub layers: Vec<EncoderLayerWeights>,
    pub vit_merger: WindowAttentionMergerWeights,
    pub post_layernorm: LayerNormWeights,
    pub merger: TopMergerWeights,
}

#[derive(Deserialize, Default)]
struct SafetensorEntry {
    #[serde(default)]
    dtype: String,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    data_offsets: (u64, u64),
}

pub(crate) struct SafetensorLoader {
    file: File,
    header_offset: u64,
    entries: HashMap<String, SafetensorEntry>,
}

impl SafetensorLoader {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact(&mut header_buf)?;
        let entries: HashMap<String, SafetensorEntry> = serde_json::from_slice(&header_buf)?;
        let header_offset = 8 + header_len;
        Ok(Self {
            file,
            header_offset,
            entries,
        })
    }

    pub(crate) fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub(crate) fn read_tensor_f32(
        &mut self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let entry = self.entries.get(key).ok_or_else(|| anyhow!("Key '{key}' not found in safetensors"))?;
        let (start, end) = entry.data_offsets;
        let len = (end - start) as usize;
        let mut raw = vec![0u8; len];
        self.file.seek(SeekFrom::Start(self.header_offset + start))?;
        self.file.read_exact(&mut raw)?;

        match entry.dtype.as_str() {
            "BF16" => {
                let u16s = bytemuck::cast_slice::<u8, u16>(&raw);
                let f32s: Vec<f32> = u16s.iter().map(|&u| f32::from_bits((u as u32) << 16)).collect();
                Ok((f32s, entry.shape.clone()))
            },
            "F32" => {
                let f32s = bytemuck::cast_slice::<u8, f32>(&raw).to_vec();
                Ok((f32s, entry.shape.clone()))
            },
            other => Err(anyhow!("Unsupported dtype for '{key}': {other}")),
        }
    }

    fn read_tensor_u8(
        &mut self,
        key: &str,
    ) -> Result<(Vec<u8>, Vec<usize>)> {
        let entry = self.entries.get(key).ok_or_else(|| anyhow!("Key '{key}' not found in safetensors"))?;
        let (start, end) = entry.data_offsets;
        let len = (end - start) as usize;
        let mut raw = vec![0u8; len];
        self.file.seek(SeekFrom::Start(self.header_offset + start))?;
        self.file.read_exact(&mut raw)?;
        Ok((raw, entry.shape.clone()))
    }

    fn load_linear_weight(
        &mut self,
        prefix: &str,
    ) -> Result<LinearWeight> {
        let weight_key = format!("{prefix}.weight");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");
        let bias_key = format!("{prefix}.bias");

        let has_scales = self.entries.contains_key(&scales_key);

        let (weight, in_features, out_features) = if has_scales {
            let (raw_w, shape_w) = self.read_tensor_u8(&weight_key)?;
            let (scales, shape_s) = self.read_tensor_f32(&scales_key)?;
            let (biases, _shape_b) = self.read_tensor_f32(&biases_key)?;

            let n = shape_w[0];
            let k = raw_w.len() / n;
            let g = shape_s[1];
            let group_size = k / g;
            ensure!(group_size == 64, "Expected group size 64");

            let mut dequant_w = vec![0.0f32; n * k];
            for row in 0..n {
                let row_w_offset = row * k;
                let row_s_offset = row * g;
                for col in 0..k {
                    let group = col / group_size;
                    let u8_val = raw_w[row_w_offset + col] as f32;
                    let s = scales[row_s_offset + group];
                    let b = biases[row_s_offset + group];
                    dequant_w[row_w_offset + col] = u8_val * s + b;
                }
            }
            (dequant_w, k, n)
        } else {
            let (w, shape) = self.read_tensor_f32(&weight_key)?;
            let n = shape[0];
            let k = shape[1];
            (w, k, n)
        };

        let bias = if self.entries.contains_key(&bias_key) {
            let (b, _) = self.read_tensor_f32(&bias_key)?;
            Some(b)
        } else {
            None
        };

        Ok(LinearWeight {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    fn load_layer_norm(
        &mut self,
        prefix: &str,
    ) -> Result<LayerNormWeights> {
        let (gamma, _) = self.read_tensor_f32(&format!("{prefix}.weight"))?;
        let bias_key = format!("{prefix}.bias");
        let beta = if self.entries.contains_key(&bias_key) {
            let (b, _) = self.read_tensor_f32(&bias_key)?;
            Some(b)
        } else {
            None
        };
        Ok(LayerNormWeights {
            gamma,
            beta,
        })
    }
}

impl VisionTower {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let shard_path = model_dir.join("model-00003-of-00003.safetensors");
        let mut loader = SafetensorLoader::open(&shard_path)
            .with_context(|| format!("Opening vision shard {}", shard_path.display()))?;

        // Load configs
        let vision_config_path = model_dir.join("vision_config.json");
        let (vision_config, canvas_config) = if vision_config_path.exists() {
            let json_bytes = std::fs::read(&vision_config_path)?;
            let parsed: super::config::VisionModelConfig = serde_json::from_slice(&json_bytes)?;
            (parsed.vision_config, parsed.canvas_config)
        } else {
            (VisionConfig::default(), CanvasConfig::default())
        };

        let hidden_size = vision_config.hidden_size;
        let num_heads = vision_config.num_attention_heads;
        let head_dim = vision_config.head_dim();
        let eps = vision_config.layer_norm_eps;

        // Patch embedding
        let (pw, shape_pw) = loader.read_tensor_f32("vision_tower.embeddings.patch_embedding.weight")?;
        let (pb, _) = loader.read_tensor_f32("vision_tower.embeddings.patch_embedding.bias")?;
        let in_patch_features = shape_pw[1] * shape_pw[2] * shape_pw[3];
        ensure!(in_patch_features == 14 * 14 * 3);
        let patch_proj = LinearWeight {
            weight: pw,
            bias: Some(pb),
            in_features: in_patch_features,
            out_features: hidden_size,
        };

        // Position embedding
        let (position_embedding, _) = loader.read_tensor_f32("vision_tower.embeddings.position_embedding.weight")?;
        let pos_grid = vision_config.image_size / vision_config.patch_size;
        ensure!(
            position_embedding.len() == pos_grid * pos_grid * hidden_size,
            "Position embedding holds {} values, expected {pos_grid}x{pos_grid}x{hidden_size}",
            position_embedding.len()
        );

        // 27 Encoder Layers
        let mut layers = Vec::with_capacity(vision_config.num_hidden_layers);
        for i in 0..vision_config.num_hidden_layers {
            let prefix = format!("vision_tower.encoder.layers.{i}");
            let layer_norm1 = loader.load_layer_norm(&format!("{prefix}.layer_norm1"))?;
            let layer_norm2 = loader.load_layer_norm(&format!("{prefix}.layer_norm2"))?;

            let qkv_proj = loader.load_linear_weight(&format!("{prefix}.self_attn.qkv_proj"))?;
            let out_proj = loader.load_linear_weight(&format!("{prefix}.self_attn.out_proj"))?;
            let self_attn = SiglipAttentionWeights {
                qkv_proj,
                out_proj,
                num_heads,
                head_dim,
                hidden_size,
            };

            let fc1 = loader.load_linear_weight(&format!("{prefix}.mlp.fc1"))?;
            let fc2 = loader.load_linear_weight(&format!("{prefix}.mlp.fc2"))?;
            let mlp = SiglipMLPWeights {
                fc1,
                fc2,
            };

            layers.push(EncoderLayerWeights {
                layer_norm1,
                self_attn,
                layer_norm2,
                mlp,
                hidden_size,
                eps,
            });
        }

        // vit_merger
        let vit_merger_prefix = "vision_tower.vit_merger";
        let vm_ln1 = loader.load_layer_norm(&format!("{vit_merger_prefix}.layer_norm1"))?;
        let vm_pre_norm = loader.load_layer_norm(&format!("{vit_merger_prefix}.pre_norm"))?;
        let vm_qkv = loader.load_linear_weight(&format!("{vit_merger_prefix}.self_attn.qkv_proj"))?;
        let vm_out = loader.load_linear_weight(&format!("{vit_merger_prefix}.self_attn.out_proj"))?;
        let vm_attn = SiglipAttentionWeights {
            qkv_proj: vm_qkv,
            out_proj: vm_out,
            num_heads,
            head_dim,
            hidden_size,
        };
        let vm_l1 = loader.load_linear_weight(&format!("{vit_merger_prefix}.linear_1"))?;
        let vm_l2 = loader.load_linear_weight(&format!("{vit_merger_prefix}.linear_2"))?;
        let vit_merger = WindowAttentionMergerWeights {
            layer_norm1: vm_ln1,
            self_attn: vm_attn,
            pre_norm: vm_pre_norm,
            linear_1: vm_l1,
            linear_2: vm_l2,
            window: 2,
            hidden_size,
            eps,
        };

        // post_layernorm
        let post_layernorm = loader.load_layer_norm("vision_tower.post_layernorm")?;

        // Top merger
        let merger_mlp_prefix = "merger.mlp.0";
        let top_pre_norm = loader.load_layer_norm(&format!("{merger_mlp_prefix}.pre_norm"))?;
        let top_l1 = loader.load_linear_weight(&format!("{merger_mlp_prefix}.linear_1"))?;
        let top_l2 = loader.load_linear_weight(&format!("{merger_mlp_prefix}.linear_2"))?;
        let merger = TopMergerWeights {
            mlp: DownsampleMLPWeights {
                pre_norm: top_pre_norm,
                linear_1: top_l1,
                linear_2: top_l2,
                eps,
            },
        };

        Ok(Self {
            vision_config,
            canvas_config,
            patch_proj,
            position_embedding,
            layers,
            vit_merger,
            post_layernorm,
            merger,
        })
    }

    pub fn forward(
        &self,
        preprocessed: &PreprocessedVision,
    ) -> Result<Vec<f32>> {
        let num_patches = preprocessed.patches.len() / (14 * 14 * 3);
        let hidden_size = self.vision_config.hidden_size;

        // 1. Patch projection: [num_patches, 14*14*3] -> [num_patches, hidden_size]
        let mut embeds = vec![0.0f32; num_patches * hidden_size];
        self.patch_proj.forward(&preprocessed.patches, num_patches, &mut embeds);

        // 2. Add position embeddings
        let mut pos_indices = Vec::with_capacity(num_patches);
        for &(_, h, w) in &preprocessed.grids {
            pos_indices.extend(nearest_position_ids(h, w, 70));
        }
        ensure!(pos_indices.len() == num_patches, "Position indices length mismatch");

        for (i, &pos_idx) in pos_indices.iter().enumerate() {
            let pos_src = pos_idx * hidden_size;
            let dst = i * hidden_size;
            let pos_slice = &self.position_embedding[pos_src..pos_src + hidden_size];
            for d in 0..hidden_size {
                embeds[dst + d] += pos_slice[d];
            }
        }

        // 3. Encoder layers with window merger spliced in
        let mut h = embeds;
        let mut current_grids = preprocessed.grids.clone();
        let mut segments: Vec<usize> = current_grids.iter().map(|gr| gr.1 * gr.2).collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let current_num_tokens = h.len() / hidden_size;
            layer.forward(&mut h, &segments, current_num_tokens);

            if i == self.canvas_config.insert_layer_id {
                h = self.vit_merger.forward(&h, &current_grids);
                let k = self.vit_merger.window;
                current_grids = current_grids.iter().map(|&(t, gr_h, gr_w)| (t, gr_h / k, gr_w / k)).collect();
                segments = current_grids.iter().map(|gr| gr.1 * gr.2).collect();
            }
        }

        // 4. Post layer norm
        let mut post_normed = vec![0.0f32; h.len()];
        layer_norm(
            &h,
            &self.post_layernorm.gamma,
            self.post_layernorm.beta.as_deref(),
            self.vision_config.layer_norm_eps,
            hidden_size,
            &mut post_normed,
        );

        // 5. Top merger
        let visual_tokens = self.merger.forward(&post_normed, &current_grids, hidden_size);
        Ok(visual_tokens)
    }
}
