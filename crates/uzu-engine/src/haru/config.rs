use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HaruDryDefaults {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub lookback: usize,
    pub max_match_length: usize,
    pub max_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HaruGenerationDefaults {
    pub guidance_scale: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub eop_bias: f32,
    pub dry: HaruDryDefaults,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HaruCodecConfig {
    pub latent_dim: u32,
    pub base_channels: u32,
    pub ratios: Vec<u32>,
    pub encoder_depths: Vec<u32>,
    pub decoder_depths: Vec<u32>,
    pub norm_eps: f32,
    pub latent_std: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HaruHeadConfig {
    pub hidden_size: u32,
    pub latent_dim: u32,
    pub num_layers: u32,
    pub ffn_ratio: f32,
    pub norm_eps: f32,
    pub interval_embed: f32,
    pub num_steps: u32,
    pub guidance_scale: f32,
    pub guidance_range: [f32; 2],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HaruConfig {
    pub model_type: String,
    pub sample_rate: u32,
    pub frame_samples: u32,
    pub frame_rate: f32,
    pub latent_dim: u32,
    pub hidden_size: u32,
    pub pad_token_id: u32,
    pub eop_token_id: u32,
    pub eos_token_id: u32,
    pub mlp_format: String,
    pub codec: HaruCodecConfig,
    pub head: HaruHeadConfig,
    pub defaults: HaruGenerationDefaults,
}

impl HaruConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("haru_config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}
