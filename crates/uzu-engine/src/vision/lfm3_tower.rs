//! LFM3-VL vision tower + multimodal projector, CPU f32.
//!
//! Ports the MLX reference (`mlx_vlm/models/lfm3_vl/vision.py`, `lfm3_vl.py`):
//! patchifier -> RMSNorm -> 14 conv/attention hybrid layers (RMSNorm pre-norm,
//! no sandwich norm) -> LayerNorm -> 4-pack merge -> linear_1 -> erf GELU ->
//! linear_2. Weights come from the bundle's `vision.safetensors` (BF16 raw,
//! F32 norms — see `scripts/lfm3-convert.py`), geometry from
//! `vision_config.json`.
//!
//! Conventions proven in phase A and reused here: fused MLP `up_projection`
//! is **[up (w3) | gate (w1)]** with SiLU on the second half; conv
//! `in_projection` splits into [b | c | v] exactly as stored.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

use super::tower::{LinearWeight, SafetensorLoader, gelu_erf};

#[inline]
fn load_linear(
    loader: &mut SafetensorLoader,
    key: &str,
    bias_key: Option<&str>,
) -> Result<LinearWeight> {
    let (weight, shape) = loader.read_tensor_f32(key)?;
    ensure!(shape.len() == 2, "'{key}' must be 2-D, got {shape:?}");
    let bias = match bias_key {
        Some(k) => Some(loader.read_tensor_f32(k)?.0),
        None => None,
    };
    Ok(LinearWeight {
        out_features: shape[0],
        in_features: shape[1],
        weight,
        bias,
    })
}

#[inline]
fn load_norm(
    loader: &mut SafetensorLoader,
    key: &str,
) -> Result<Vec<f32>> {
    Ok(loader.read_tensor_f32(key)?.0)
}

#[derive(Clone, Deserialize)]
pub struct Lfm3ProjectorConfig {
    pub epsilon: f32,
    pub has_biases: bool,
}

/// `image` section of `vision_config.json` (Qwen3VL-style processor limits and
/// normalization). Defaults match the LFM3-VL bundle.
#[derive(Clone, Deserialize)]
pub struct Lfm3ImageConfig {
    #[serde(default = "Lfm3ImageConfig::default_min_pixels")]
    pub min_pixels: u64,
    #[serde(default = "Lfm3ImageConfig::default_max_pixels")]
    pub max_pixels: u64,
    #[serde(default = "Lfm3ImageConfig::default_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "Lfm3ImageConfig::default_std")]
    pub image_std: [f32; 3],
    #[serde(default = "Lfm3ImageConfig::default_rescale")]
    pub rescale_factor: f32,
}

impl Default for Lfm3ImageConfig {
    fn default() -> Self {
        Self {
            min_pixels: Self::default_min_pixels(),
            max_pixels: Self::default_max_pixels(),
            image_mean: Self::default_mean(),
            image_std: Self::default_std(),
            rescale_factor: Self::default_rescale(),
        }
    }
}

impl Lfm3ImageConfig {
    fn default_min_pixels() -> u64 {
        65536
    }

    fn default_max_pixels() -> u64 {
        16_777_216
    }

    fn default_mean() -> [f32; 3] {
        [0.5, 0.5, 0.5]
    }

    fn default_std() -> [f32; 3] {
        [0.5, 0.5, 0.5]
    }

    fn default_rescale() -> f32 {
        1.0 / 255.0
    }
}

#[derive(Clone, Deserialize)]
pub struct Lfm3VisionConfig {
    #[serde(default = "Lfm3VisionConfig::default_type")]
    #[allow(dead_code)]
    pub type_: String,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub norm_epsilon: f32,
    pub rope_theta: f64,
    pub layer_types: Vec<String>,
    pub projector: Lfm3ProjectorConfig,
    #[serde(default)]
    pub image: Lfm3ImageConfig,
}

impl Lfm3VisionConfig {
    fn default_type() -> String {
        "Lfm3VlVisionConfig".to_string()
    }

    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("vision_config.json");
        let bytes = std::fs::read(&path).with_context(|| format!("Reading {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("Parsing {}", path.display()))
    }
}

pub(crate) enum Lfm3Mixer {
    Conv {
        in_projection: LinearWeight,  // [3 * dim, dim], rows [b | c | v]
        spatial_conv: Vec<f32>,       // [C, k, k] depthwise, F32
        out_projection: LinearWeight, // [dim, dim]
    },
    Attention {
        q_proj: LinearWeight, // [dim, dim]
        k_proj: LinearWeight, // [kv_dim, dim]
        v_proj: LinearWeight, // [kv_dim, dim]
        out_proj: LinearWeight,
        query_norm: Vec<f32>, // RMSNorm scales [head_dim]
        key_norm: Vec<f32>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
}

pub(crate) struct Lfm3VisionLayer {
    pub(crate) mixer: Lfm3Mixer,
    pub(crate) pre_mixer_norm: Vec<f32>,
    pub(crate) pre_mlp_norm: Vec<f32>,
    pub(crate) mlp_up: LinearWeight,   // fused [up (w3) | gate (w1)] [2 * hidden, dim]
    pub(crate) mlp_down: LinearWeight, // [dim, hidden]
}

pub struct Lfm3VisionTower {
    pub config: Lfm3VisionConfig,
    pub(crate) patchifier: LinearWeight,
    pub(crate) embedding_norm: Vec<f32>,
    pub(crate) layers: Vec<Lfm3VisionLayer>,
    pub(crate) projector_ln: (Vec<f32>, Option<Vec<f32>>), // (gamma, beta)
    pub(crate) projector_linear_1: LinearWeight,
    pub(crate) projector_linear_2: LinearWeight,
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// MLX `nn.RMSNorm`: `x * weight / sqrt(mean(x^2) + eps)`, in place per row.
fn rms_norm(
    x: &mut [f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
) {
    for row in x.chunks_exact_mut(dim) {
        let msq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (msq + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v *= inv * w;
        }
    }
}

/// MLX `nn.LayerNorm`, in place per row; beta (bias) is optional.
fn layer_norm(
    x: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    dim: usize,
) {
    for row in x.chunks_exact_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        match beta {
            Some(beta) => {
                for d in 0..dim {
                    row[d] = (row[d] - mean) * inv * gamma[d] + beta[d];
                }
            },
            None => {
                for d in 0..dim {
                    row[d] = (row[d] - mean) * inv * gamma[d];
                }
            },
        }
    }
}

/// Half-rotation RoPE for one head, in place: index pairs `(d, d + half)` are
/// computed from each other's original value, so no snapshot copy is needed.
fn rope_in_place(
    head: &mut [f32],
    cs: &[f32],
    sn: &[f32],
) {
    let half = head.len() / 2;
    for d in 0..half {
        let (a, b) = (head[d], head[d + half]);
        head[d] = a * cs[d] - b * sn[d];
        head[d + half] = b * cs[d + half] + a * sn[d + half];
    }
}

impl Lfm3VisionTower {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Lfm3VisionConfig::load(model_dir)?;
        let mut loader = SafetensorLoader::open(model_dir.join("vision.safetensors"))
            .with_context(|| format!("Opening vision weights in {}", model_dir.display()))?;

        let dim = config.model_dim;
        ensure!(config.layer_types.len() == config.num_layers, "layer_types length mismatch");
        let head_dim = dim / config.num_heads;

        let patchifier = load_linear(&mut loader, "vision.patchifier.weights", None)?;
        ensure!(
            patchifier.in_features
                == config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size,
            "patchifier input dim mismatch"
        );
        let embedding_norm = load_norm(&mut loader, "vision.embedding_norm.scales")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, lt) in config.layer_types.iter().enumerate() {
            let ldst = format!("vision.layers.{i}");
            let mixer = match lt.as_str() {
                "full_attention" => Lfm3Mixer::Attention {
                    q_proj: load_linear(&mut loader, &format!("{ldst}.mixer.q_proj.weights"), None)?,
                    k_proj: load_linear(&mut loader, &format!("{ldst}.mixer.k_proj.weights"), None)?,
                    v_proj: load_linear(&mut loader, &format!("{ldst}.mixer.v_proj.weights"), None)?,
                    out_proj: load_linear(&mut loader, &format!("{ldst}.mixer.out_proj.weights"), None)?,
                    query_norm: load_norm(&mut loader, &format!("{ldst}.mixer.query_norm.scales"))?,
                    key_norm: load_norm(&mut loader, &format!("{ldst}.mixer.key_norm.scales"))?,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim,
                },
                "conv" => Lfm3Mixer::Conv {
                    in_projection: load_linear(&mut loader, &format!("{ldst}.mixer.in_projection.weights"), None)?,
                    spatial_conv: loader.read_tensor_f32(&format!("{ldst}.mixer.spatial_conv.weights"))?.0,
                    out_projection: load_linear(&mut loader, &format!("{ldst}.mixer.out_projection.weights"), None)?,
                },
                other => return Err(anyhow!("Unknown vision layer type '{other}'")),
            };
            layers.push(Lfm3VisionLayer {
                mixer,
                pre_mixer_norm: load_norm(&mut loader, &format!("{ldst}.pre_mixer_norm.scales"))?,
                pre_mlp_norm: load_norm(&mut loader, &format!("{ldst}.pre_mlp_norm.scales"))?,
                mlp_up: load_linear(&mut loader, &format!("{ldst}.mlp.up_projection.weights"), None)?,
                mlp_down: load_linear(&mut loader, &format!("{ldst}.mlp.down_projection.weights"), None)?,
            });
        }

        let ln_gamma = load_norm(&mut loader, "vision.projector.layer_norm.scales")?;
        let ln_beta = loader.read_tensor_f32("vision.projector.layer_norm.biases").ok().map(|(b, _)| b);
        let linear_1 = load_linear(
            &mut loader,
            "vision.projector.linear_1.weights",
            config.projector.has_biases.then_some("vision.projector.linear_1.biases"),
        )?;
        let linear_2 = load_linear(
            &mut loader,
            "vision.projector.linear_2.weights",
            config.projector.has_biases.then_some("vision.projector.linear_2.biases"),
        )?;

        Ok(Self {
            config,
            patchifier,
            embedding_norm,
            layers,
            projector_ln: (ln_gamma, ln_beta),
            projector_linear_1: linear_1,
            projector_linear_2: linear_2,
        })
    }

    /// 2D axial RoPE tables, block-major patch coordinates. Returns
    /// `(cos, sin)`, each `[n, head_dim]` — reference `grid_positions` +
    /// angle construction in `vision.py`.
    pub(crate) fn rope_tables(
        &self,
        grid_h: usize,
        grid_w: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        rope_tables_for(&self.config, grid_h, grid_w)
    }

    /// Patchifier + embedding RMSNorm for one frame. Shared by `forward` and the
    /// Metal parity tests.
    pub(crate) fn embed(
        &self,
        pixel_values: &[f32],
        n: usize,
    ) -> Vec<f32> {
        let dim = self.config.model_dim;
        let mut x = vec![0f32; n * dim];
        self.patchifier.forward(pixel_values, n, &mut x);
        rms_norm(&mut x, &self.embedding_norm, self.config.norm_epsilon, dim);
        x
    }

    pub fn forward(
        &self,
        pixel_values: &[f32],
        grid: [usize; 3],
    ) -> Result<Vec<f32>> {
        let [grid_t, grid_h, grid_w] = grid;
        let cfg = &self.config;
        let dim = cfg.model_dim;
        let m = cfg.spatial_merge_size;
        let eps = cfg.norm_epsilon;
        let n = grid_h * grid_w;
        let patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        ensure!(
            pixel_values.len() == grid_t * n * patch_dim,
            "pixel_values size mismatch: {} != {grid_t} * {n} * {patch_dim}",
            pixel_values.len()
        );

        let (cos, sin) = self.rope_tables(grid_h, grid_w);

        let mut frames: Vec<f32> = Vec::with_capacity(grid_t * n * dim);
        for t in 0..grid_t {
            let input = &pixel_values[t * n * patch_dim..(t + 1) * n * patch_dim];
            let mut x = self.embed(input, n);
            if let Ok(dir) = std::env::var("LFM3_VL_DEBUG_DIR") {
                let _ = std::fs::write(format!("{dir}/embed.f32"), f32_le(&x));
            }

            for (li, layer) in self.layers.iter().enumerate() {
                // x = x + mixer(pre_mixer_norm(x))
                let mut h = x.clone();
                rms_norm(&mut h, &layer.pre_mixer_norm, eps, dim);
                let mixer_out = match &layer.mixer {
                    Lfm3Mixer::Conv {
                        in_projection,
                        spatial_conv,
                        out_projection,
                    } => conv_forward(&h, n, grid_h, grid_w, m, dim, in_projection, spatial_conv, out_projection),
                    Lfm3Mixer::Attention {
                        ..
                    } => attention_forward(&h, n, &layer.mixer, &cos, &sin, eps),
                };
                if let Ok(dir) = std::env::var("LFM3_VL_DEBUG_DIR") {
                    let _ = std::fs::write(format!("{dir}/mixer{li:02}.f32"), f32_le(&mixer_out));
                }
                for (xi, mi) in x.iter_mut().zip(mixer_out) {
                    *xi += mi;
                }
                // x = x + mlp(pre_mlp_norm(x))
                let mut h = x.clone();
                rms_norm(&mut h, &layer.pre_mlp_norm, eps, dim);
                let mlp_out = mlp_forward(&h, n, &layer.mlp_up, &layer.mlp_down);
                if let Ok(dir) = std::env::var("LFM3_VL_DEBUG_DIR") {
                    let _ = std::fs::write(format!("{dir}/ffn{li:02}.f32"), f32_le(&mlp_out));
                }
                for (xi, mi) in x.iter_mut().zip(mlp_out) {
                    *xi += mi;
                }
            }
            frames.extend(x);
        }
        ensure!(frames.len() == grid_t * n * dim, "frame size mismatch");

        // Projector: LN(eps 1e-6, bias) -> merge 4-pack -> linear_1 -> GELU -> linear_2.
        let packed_dim = dim * m * m;
        ensure!(self.projector_linear_1.in_features == packed_dim, "projector input dim mismatch");
        let mut p = frames;
        layer_norm(&mut p, &self.projector_ln.0, self.projector_ln.1.as_deref(), self.config.projector.epsilon, dim);
        // Row-major 4-pack: consecutive block-major tokens form one merged row.
        let num_merged = p.len() / packed_dim;
        let mut packed = vec![0f32; num_merged * packed_dim];
        for (o, row) in p.chunks_exact(dim).enumerate() {
            packed[(o / (m * m)) * packed_dim + (o % (m * m)) * dim
                ..(o / (m * m)) * packed_dim + (o % (m * m)) * dim + dim]
                .copy_from_slice(row);
        }
        let mut h1 = vec![0f32; num_merged * self.projector_linear_1.out_features];
        self.projector_linear_1.forward(&packed, num_merged, &mut h1);
        for v in &mut h1 {
            *v = gelu_erf(*v);
        }
        let mut out = vec![0f32; num_merged * self.projector_linear_2.out_features];
        self.projector_linear_2.forward(&h1, num_merged, &mut out);
        Ok(out)
    }
}

/// 2D axial RoPE tables for a `grid_h x grid_w` frame, block-major patch
/// coordinates. Shared by the CPU and Metal towers.
pub(crate) fn rope_tables_for(
    config: &Lfm3VisionConfig,
    grid_h: usize,
    grid_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let m = config.spatial_merge_size;
    let head_dim = config.model_dim / config.num_heads;
    let spatial_dim = head_dim / 2;
    let freqs: Vec<f64> =
        (0..spatial_dim).step_by(2).map(|i| config.rope_theta.powf(-(i as f64) / spatial_dim as f64)).collect();

    let gh_m = grid_h / m;
    let gw_m = grid_w / m;
    let n = grid_h * grid_w;
    let mut angles = vec![0f64; n * head_dim];
    let mut i = 0usize;
    for hb in 0..gh_m {
        for wb in 0..gw_m {
            for ir in 0..m {
                for ic in 0..m {
                    let (row, col) = (hb * m + ir, wb * m + ic);
                    let base = i * head_dim;
                    let block = freqs.len(); // half of spatial_dim
                    for (j, f) in freqs.iter().enumerate() {
                        let ra = row as f64 * f;
                        let ca = col as f64 * f;
                        // angles = [row*f | col*f] (n, spatial_dim), duplicated
                        // twice via concatenate -> (n, head_dim).
                        angles[base + j] = ra;
                        angles[base + block + j] = ca;
                        angles[base + 2 * block + j] = ra;
                        angles[base + 3 * block + j] = ca;
                    }
                    i += 1;
                }
            }
        }
    }
    let cos = angles.iter().map(|a| a.cos() as f32).collect();
    let sin = angles.iter().map(|a| a.sin() as f32).collect();
    (cos, sin)
}

/// Vision conv block: `[b|c|v]` split, gate `b*v` scattered to a
/// `[1, H, W, C]` grid, depthwise k×k (padding k/2), regathered, `out_proj(c * out)`.
#[allow(clippy::too_many_arguments)]
fn conv_forward(
    x: &[f32],
    n: usize,
    grid_h: usize,
    grid_w: usize,
    m: usize,
    dim: usize,
    in_projection: &LinearWeight,
    spatial_conv: &[f32],
    out_projection: &LinearWeight,
) -> Vec<f32> {
    let mut proj = vec![0f32; n * 3 * dim];
    in_projection.forward(x, n, &mut proj);
    // `in_projection` emits `[n, 3*dim]` row-major, split per token into
    // [b | c | v].
    let row_stride = 3 * dim;

    // gate grid: token order is block-major, scatter to (H, W, C).
    let k = spatial_conv.len() / dim; // kernel*kernel per channel
    let kw = (k as f64).sqrt() as usize;
    let pad = kw / 2;
    let gh_m = grid_h / m;
    let gw_m = grid_w / m;
    let mut grid = vec![0f32; grid_h * grid_w * dim];
    let mut token = 0usize;
    for hb in 0..gh_m {
        for wb in 0..gw_m {
            for ir in 0..m {
                for ic in 0..m {
                    let (y, xx) = (hb * m + ir, wb * m + ic);
                    let dst = (y * grid_w + xx) * dim;
                    let src = token * row_stride;
                    for d in 0..dim {
                        grid[dst + d] = proj[src + d] * proj[src + 2 * dim + d];
                    }
                    token += 1;
                }
            }
        }
    }

    // Depthwise conv, padding = kw/2.
    let mut out_grid = vec![0f32; grid_h * grid_w * dim];
    for y in 0..grid_h {
        for x in 0..grid_w {
            let base = (y * grid_w + x) * dim;
            for ky in 0..kw as isize {
                let sy = y as isize + ky - pad as isize;
                if !(0..grid_h as isize).contains(&sy) {
                    continue;
                }
                for kx in 0..kw as isize {
                    let sx = x as isize + kx - pad as isize;
                    if !(0..grid_w as isize).contains(&sx) {
                        continue;
                    }
                    let src = (sy as usize * grid_w + sx as usize) * dim;
                    for d in 0..dim {
                        out_grid[base + d] += grid[src + d] * spatial_conv[d * k + ky as usize * kw + kx as usize];
                    }
                }
            }
        }
    }

    // Regather in block-major order, multiply by c, project.
    let mut gated = vec![0f32; n * dim];
    let mut token = 0usize;
    for hb in 0..gh_m {
        for wb in 0..gw_m {
            for ir in 0..m {
                for ic in 0..m {
                    let (y, xx) = (hb * m + ir, wb * m + ic);
                    let src = (y * grid_w + xx) * dim;
                    let dst = token * dim;
                    for d in 0..dim {
                        gated[dst + d] = proj[token * row_stride + dim + d] * out_grid[src + d];
                    }
                    token += 1;
                }
            }
        }
    }
    let mut out = vec![0f32; n * dim];
    out_projection.forward(&gated, n, &mut out);
    out
}

/// Vision attention: q/k RMSNorm per head, 2D axial rope (NeoX half-rotation),
/// GQA, full bidirectional attention, out projection.
fn attention_forward(
    x: &[f32],
    n: usize,
    mixer: &Lfm3Mixer,
    cos: &[f32],
    sin: &[f32],
    eps: f32,
) -> Vec<f32> {
    let Lfm3Mixer::Attention {
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        query_norm,
        key_norm,
        num_heads,
        num_kv_heads,
        head_dim,
    } = mixer
    else {
        unreachable!("attention_forward called on conv mixer")
    };
    let (num_heads, num_kv_heads, head_dim) = (*num_heads, *num_kv_heads, *head_dim);
    let dim = q_proj.out_features;
    let kv_dim = num_kv_heads * head_dim;
    let scale = (head_dim as f32).powf(-0.5);
    let groups = num_heads / num_kv_heads;

    let mut q = vec![0f32; n * dim];
    let mut k = vec![0f32; n * kv_dim];
    let mut v = vec![0f32; n * kv_dim];
    q_proj.forward(x, n, &mut q);
    k_proj.forward(x, n, &mut k);
    v_proj.forward(x, n, &mut v);

    // Per-head RMSNorm then rope (NeoX half rotation), f32.
    for (i, row) in q.chunks_exact_mut(dim).enumerate() {
        for head in row.chunks_exact_mut(head_dim) {
            let msq = head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (msq + eps).sqrt();
            for (d, val) in head.iter_mut().enumerate() {
                *val *= inv * query_norm[d];
            }
            rope_in_place(head, &cos[i * head_dim..(i + 1) * head_dim], &sin[i * head_dim..(i + 1) * head_dim]);
        }
    }
    for (i, row) in k.chunks_exact_mut(kv_dim).enumerate() {
        for head in row.chunks_exact_mut(head_dim) {
            let msq = head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (msq + eps).sqrt();
            for (d, val) in head.iter_mut().enumerate() {
                *val *= inv * key_norm[d];
            }
            rope_in_place(head, &cos[i * head_dim..(i + 1) * head_dim], &sin[i * head_dim..(i + 1) * head_dim]);
        }
    }

    // Attention: per query token, per head. kv head = q head / groups.
    let mut out = vec![0f32; n * dim];
    let mut scores = vec![0f32; n];
    for (i, qrow) in q.chunks_exact(dim).enumerate() {
        let orow = &mut out[i * dim..(i + 1) * dim];
        for h in 0..num_heads {
            let kvh = h / groups;
            let qh = &qrow[h * head_dim..(h + 1) * head_dim];
            // scores over all keys
            let mut max = f32::NEG_INFINITY;
            for (j, krow) in k.chunks_exact(kv_dim).enumerate() {
                let kh = &krow[kvh * head_dim..(kvh + 1) * head_dim];
                let s: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores[j] = s;
                max = max.max(s);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for (j, vrow) in v.chunks_exact(kv_dim).enumerate() {
                let p = scores[j] * inv_sum;
                let vhead = &vrow[kvh * head_dim..(kvh + 1) * head_dim];
                for (d, vo) in vhead.iter().enumerate() {
                    orow[h * head_dim + d] += p * vo;
                }
            }
        }
    }
    let mut projected = vec![0f32; n * dim];
    out_proj.forward(&out, n, &mut projected);
    projected
}

/// Fused SwiGLU MLP: `up_projection` rows are **[up (w3) | gate (w1)]**,
/// SiLU on the second half; `down_projection` = w2.
fn mlp_forward(
    x: &[f32],
    n: usize,
    up: &LinearWeight,
    down: &LinearWeight,
) -> Vec<f32> {
    let hidden = up.out_features / 2;
    let mut fused = vec![0f32; n * up.out_features];
    up.forward(x, n, &mut fused);
    let mut gated = vec![0f32; n * hidden];
    for (i, row) in fused.chunks_exact(up.out_features).enumerate() {
        let (u, g) = row.split_at(hidden);
        for d in 0..hidden {
            gated[i * hidden + d] = silu(g[d]) * u[d];
        }
    }
    let mut out = vec![0f32; n * down.out_features];
    down.forward(&gated, n, &mut out);
    out
}

/// Loads an f32 flat file (`len/4` little-endian f32s).
#[cfg(test)]
pub fn read_f32_file(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Loads an i64 flat file.
#[cfg(test)]
pub fn read_i64_file(path: impl AsRef<Path>) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    /// Parity against the MLX f32 reference (env-gated):
    /// LFM3_VL_BUNDLE (uzu bundle dir), LFM3_VL_PIXELS (.f32 patch input),
    /// LFM3_VL_GRID (.i64 [t,h,w]), LFM3_VL_REF (.f32 projector output).
    #[uzu_test]
    fn lfm3_vision_tower_matches_mlx_reference() {
        let parse = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let (Some(bundle), Some(pixels_path), Some(grid_path), Some(ref_path)) = (
            parse("LFM3_VL_BUNDLE"),
            parse("LFM3_VL_PIXELS"),
            parse("LFM3_VL_GRID"),
            parse("LFM3_VL_REF"),
        ) else {
            println!("{{\"skipped\": \"set LFM3_VL_BUNDLE/LFM3_VL_PIXELS/LFM3_VL_GRID/LFM3_VL_REF to run\"}}");
            return;
        };
        let pixels = read_f32_file(&pixels_path).unwrap();
        let grid = read_i64_file(&grid_path).unwrap();
        let expected = read_f32_file(&ref_path).unwrap();

        let tower = Lfm3VisionTower::load_from_model_dir(&bundle).unwrap();
        let out = tower.forward(&pixels, [grid[0] as usize, grid[1] as usize, grid[2] as usize]).unwrap();
        assert_eq!(out.len(), expected.len(), "output size mismatch");
        let max_diff = out.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let denom = expected.iter().map(|b| b.abs()).fold(1e-9f32, f32::max);
        let rel = out.iter().zip(&expected).map(|(a, b)| (a - b).abs() / denom).fold(0.0f32, f32::max);
        println!("{{\"max_abs_diff\": {max_diff}, \"max_rel\": {rel}, \"rows\": {}}}", out.len() / 2048);
        assert!(max_diff < 1e-2, "vision tower output differs from MLX reference: max_abs_diff = {max_diff}");
    }
}
