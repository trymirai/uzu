use std::{collections::HashMap, fs, io, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct D1Config {
    #[serde(default = "default_codes")]
    pub d1_codes: u32,
    #[serde(default = "default_attention")]
    pub d1_attention: String,
    #[serde(default)]
    pub d1_numbers: bool,
    #[serde(default = "default_set_size")]
    pub d1_set_size: usize,
    #[serde(default)]
    pub d1_vision_max_pixels: Option<u64>,
    #[serde(default = "default_box_edges")]
    pub d1_box_edges: String,
    #[serde(default = "default_prompt")]
    pub d1_prompt: u32,
    #[serde(default = "default_read_max_tokens")]
    pub d1_read_max_tokens: usize,
    #[serde(default = "default_read_short_max_tokens")]
    pub d1_read_short_max_tokens: usize,
    #[serde(default = "default_vision_max_pixels")]
    pub vision_max_pixels: u64,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}

fn default_codes() -> u32 {
    1
}
fn default_attention() -> String {
    "set".to_string()
}
fn default_set_size() -> usize {
    4
}
fn default_box_edges() -> String {
    "median".to_string()
}
fn default_prompt() -> u32 {
    1
}
fn default_read_max_tokens() -> usize {
    512
}
fn default_read_short_max_tokens() -> usize {
    32
}
fn default_vision_max_pixels() -> u64 {
    2_097_152
}
fn default_bos() -> u32 {
    1
}
fn default_eos() -> u32 {
    7
}

impl D1Config {
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Calibration {
    #[serde(default)]
    pub temperatures: HashMap<String, f64>,
    #[serde(default = "default_temp")]
    pub default: f64,
    #[serde(default)]
    pub tables: HashMap<String, Vec<[f64; 2]>>,
}

fn default_temp() -> f64 {
    1.0
}

impl Calibration {
    pub fn load(path: &Path) -> io::Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn temperature(
        &self,
        bucket: &str,
    ) -> f64 {
        self.temperatures.get(bucket).copied().unwrap_or(self.default)
    }

    pub fn through(
        &self,
        name: &str,
        value: f64,
    ) -> f64 {
        match self.tables.get(name) {
            Some(knots) => through_knots(value, knots),
            None => value,
        }
    }

    pub fn level(
        &self,
        name: &str,
        share: f64,
    ) -> f64 {
        match self.tables.get(name) {
            Some(knots) => {
                let flipped: Vec<[f64; 2]> = knots.iter().map(|&[x, y]| [y, x]).collect();
                through_knots(share, &flipped)
            },
            None => share,
        }
    }
}

pub fn through_knots(
    value: f64,
    knots: &[[f64; 2]],
) -> f64 {
    if knots.is_empty() {
        return value;
    }
    let idx = knots.partition_point(|&[raw, _]| raw < value);
    if idx == 0 {
        return knots[0][1];
    }
    if idx == knots.len() {
        return knots[knots.len() - 1][1];
    }
    let [a, pa] = knots[idx - 1];
    let [b, pb] = knots[idx];
    pa + (value - a) * (pb - pa) / (b - a)
}

pub fn question_bucket(
    kind: &str,
    k: usize,
) -> String {
    match kind {
        "noul" => "noul".to_string(),
        "score" => format!("score:{k}"),
        _ => {
            if k <= 3 {
                "choice:small".to_string()
            } else if k <= 8 {
                "choice:mid".to_string()
            } else if k <= 26 {
                "choice:large".to_string()
            } else {
                "choice:inventory".to_string()
            }
        },
    }
}
