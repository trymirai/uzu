use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::encodable_block::conf_head::CONF_HEAD_DIR;

#[derive(Debug, Clone, Deserialize)]
pub struct ConfidenceHeadTreeConfig {
    pub layers: u32,
    pub heads: u32,
    pub width: u32,
    pub mlp_hidden: u32,
    pub relations: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ConfidenceHeadSampling {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "ConfidenceHeadFile")]
pub struct ConfidenceHeadConfig {
    pub format: u32,
    pub speculator_sha256: String,
    pub sampling: ConfidenceHeadSampling,
    pub hidden: u32,
    pub pool_hidden: u32,
    pub edge_hidden: u32,
    pub children: u32,
    pub pool: u32,
    pub max_depth: u32,
    pub model_width: u32,
    pub cap: f32,
    pub layer_norm_eps: f32,
    pub tree: Option<ConfidenceHeadTreeConfig>,
}

impl ConfidenceHeadConfig {
    pub fn new(model_dir: &Path) -> Result<Self, std::io::Error> {
        let conf_head_dir = PathBuf::from(model_dir).join(CONF_HEAD_DIR);
        let config_file_path = conf_head_dir.join("config.json");
        let config_file = File::open(config_file_path)?;
        let config_reader = BufReader::new(config_file);
        let config: ConfidenceHeadConfig = serde_json::from_reader(config_reader)?;
        Ok(config)
    }
}

#[derive(Deserialize)]
struct ConfidenceHeadFile {
    #[serde(rename = "type")]
    _type: monostate::MustBe!("ConfidenceHeadConfig"),
    format: u32,
    trained_with: TrainedWith,
    target_width: u32,
    local: LocalSizes,
    attention: Option<ConfidenceHeadTreeConfig>,
    layer_norm_eps: f32,
}

#[derive(Deserialize)]
struct TrainedWith {
    speculator_sha256: String,
    tree: TrainedTree,
    sampling: ConfidenceHeadSampling,
}

#[derive(Deserialize)]
struct TrainedTree {
    children: u32,
    pool: u32,
    max_depth: u32,
}

#[derive(Deserialize)]
struct LocalSizes {
    hidden: u32,
    pool_hidden: u32,
    edge_hidden: u32,
    logit_cap: f32,
}

impl From<ConfidenceHeadFile> for ConfidenceHeadConfig {
    fn from(file: ConfidenceHeadFile) -> Self {
        Self {
            format: file.format,
            speculator_sha256: file.trained_with.speculator_sha256,
            sampling: file.trained_with.sampling,
            hidden: file.local.hidden,
            pool_hidden: file.local.pool_hidden,
            edge_hidden: file.local.edge_hidden,
            children: file.trained_with.tree.children,
            pool: file.trained_with.tree.pool,
            max_depth: file.trained_with.tree.max_depth,
            model_width: file.target_width,
            cap: file.local.logit_cap,
            layer_norm_eps: file.layer_norm_eps,
            tree: file.attention,
        }
    }
}
