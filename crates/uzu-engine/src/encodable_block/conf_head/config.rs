use std::{fs::File, io::BufReader, path::Path};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ConfidenceHeadTreeConfig {
    pub layers: u32,
    pub heads: u32,
    pub width: u32,
    pub mlp_hidden: u32,
    pub relations: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "ConfidenceHeadFile")]
pub struct ConfidenceHeadConfig {
    /// Serialized configuration and weights format version. The loader currently supports version 2.
    pub format: u32,

    /// Lowercase hex SHA-256 of the speculator's `model.safetensors` used during training.
    pub speculator_sha256: String,

    /// Local feature width H: projected drafter rows, projected token embeddings and node vectors z_u.
    /// Also the residual stream width of the edge MLP; independent of the tree attention width.
    pub hidden: u32,

    /// Per-candidate feature width P inside the pool set encoder.
    /// Its weighted mean and componentwise maximum are concatenated into 2P features, then projected to `hidden` to form the pool contribution to z_u.
    pub pool_hidden: u32,

    /// Intermediate width of the edge MLP's residual block: H -> edge_hidden -> H.
    /// The edge MLP predicts the child-specific logit correction delta_{u,t}.
    pub edge_hidden: u32,

    /// Children per expanded node used during training, must match Weaver's `expand_width`.
    /// Determines the number of child feature records and edge scores stored per tree slot.
    pub children: u32,

    /// Candidates per depth in the drafter's top-K pool, must match Weaver's candidate pool size.
    /// The head normalizes its transition scores over all these candidates, including unselected children.
    pub pool: u32,

    /// Maximum tree depth in edges, with the root at depth 0, must match Weaver's configured maximum.
    /// Tree attention has `max_depth + 1` learned depth embeddings.
    pub max_depth: u32,

    /// Target embedding and drafter hidden-state width D, serialized as `target_width`.
    /// Both token and drafter projections map D features to `hidden` features.
    pub model_width: u32,

    /// Upper bound applied after softplus to the node's alpha and beta coefficients.
    /// These scale Weaver logits and sampler Gumbel noise in alpha * logits + beta * noise + delta.
    pub cap: f32,

    /// Epsilon added to the variance in the head's LayerNorm operations for numerical stability.
    pub layer_norm_eps: f32,

    /// Optional tree attention configuration, serialized as `attention`.
    /// Refines node and child features after tree growth, before final edge rescoring and pruning.
    /// `None` selects the local head without this tree-wide refinement.
    pub tree: Option<ConfidenceHeadTreeConfig>,
}

impl ConfidenceHeadConfig {
    pub fn new(conf_head_dir: &Path) -> Result<Self, std::io::Error> {
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
