use std::{
    fs::File,
    path::{Path, PathBuf},
};

use crate::{
    backends::Backend,
    encodable_block::conf_head::{
        CONF_HEAD_DIR,
        config::ConfidenceHeadConfig,
        error::ConfidenceHeadError,
        tree::ConfidenceHeadTree,
        weights::{ConfidenceHeadWeights, get_local_weights_buffer},
    },
    safetensors::ParameterLoader,
    utils,
};

pub struct ConfidenceHead<B: Backend> {
    config: ConfidenceHeadConfig,
    weights: ConfidenceHeadWeights<B>,
    local_weights: B::GlobalBuffer,
    tree: Option<ConfidenceHeadTree<B>>,
}

impl<B: Backend> ConfidenceHead<B> {
    pub fn new(
        context: &B::Context,
        model_dir: &Path,
        config: ConfidenceHeadConfig,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        validate_config(&config)?;

        let model_path = PathBuf::from(model_dir).join("model.safetensors");
        let model_sha256_bytes = utils::hash::sha256_file(&model_path).map_err(ConfidenceHeadError::Io)?;
        let model_sha256_str: String = model_sha256_bytes.into_iter().map(|byte| format!("{byte:02x}")).collect();
        if model_sha256_str != config.speculator_sha256 {
            return Err(ConfidenceHeadError::ModelMismatch);
        }

        let weights_file_path = PathBuf::from(model_dir).join(CONF_HEAD_DIR).join("model.safetensors");
        let weights_file = File::open(weights_file_path)?;
        let weights_loader = ParameterLoader::new(&weights_file, context)?;
        let weights_tree = weights_loader.tree();

        let weights = ConfidenceHeadWeights::new(&weights_tree, &config)?;
        let local_weights = get_local_weights_buffer(context, &config, &weights_tree)?;
        let tree: Option<ConfidenceHeadTree<B>> = if let Some(ref tree_config) = config.tree {
            Some(ConfidenceHeadTree::new(context, &weights_tree, &config, tree_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            weights,
            local_weights,
            tree,
        })
    }

    pub fn config(&self) -> &ConfidenceHeadConfig {
        &self.config
    }
}

fn validate_config<B: Backend>(config: &ConfidenceHeadConfig) -> Result<(), ConfidenceHeadError<B>> {
    let mut problems = Vec::new();

    let format = 2;
    if config.format != format {
        problems.push(format!("format {} (expected {format})", config.format));
    }

    let hidden_max = 64;
    if config.hidden == 0 || config.hidden > hidden_max {
        problems.push(format!("hidden {} outside 1..={hidden_max}", config.hidden));
    }

    if config.children == 0 || config.children > 4 || config.children * config.hidden > 256 {
        problems.push(format!("{} children of width {}", config.children, config.hidden));
    }

    let candidates_max = 512;
    if config.pool == 0 || config.pool > candidates_max {
        problems.push(format!("pool {} outside 1..={candidates_max}", config.pool));
    }

    let pool_hidden_max = 16;
    if config.pool_hidden == 0 || config.pool_hidden > pool_hidden_max {
        problems.push(format!("pool width {} outside 1..={pool_hidden_max}", config.pool_hidden));
    }

    let edge_hidden_max = 128;
    if config.edge_hidden == 0 || config.edge_hidden > edge_hidden_max {
        problems.push(format!("edge width {} outside 1..={edge_hidden_max}", config.edge_hidden));
    }

    // The Metal dense layers read four adjacent inputs at once:
    // every layer's input width is a multiple of four (hidden, 2 * pool width, 3 * hidden + 4, edge width).
    if !config.hidden.is_multiple_of(4)
        || !config.pool_hidden.is_multiple_of(2)
        || !config.edge_hidden.is_multiple_of(4)
    {
        problems.push(format!(
            "widths hidden {}, pool {}, edge {} not multiples of 4, 2, 4",
            config.hidden, config.pool_hidden, config.edge_hidden
        ));
    }

    if config.max_depth + 1 > 32 {
        problems.push(format!("max depth {}", config.max_depth));
    }

    if config.speculator_sha256.len() != 64
        || !config.speculator_sha256.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        problems.push(format!("speculator sha256 {:?} is not 64 lowercase hex digits", config.speculator_sha256));
    }

    if let Some(tree) = &config.tree {
        // Tree relations the attention biases: self, parent, child, ancestor two or more edges up, descendant likewise, sibling.
        let relations = 6;
        if tree.relations != relations {
            problems.push(format!("{} tree relations (expected {relations})", tree.relations));
        }

        if tree.heads == 0
            || tree.width % tree.heads != 0
            || 256 % (tree.width / tree.heads) != 0
            || tree.width / tree.heads > 64
            || (tree.width / tree.heads) % 8 != 0
        {
            problems.push(format!("{} heads over width {}", tree.heads, tree.width));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(ConfidenceHeadError::Unsupported(problems.join("; ")))
    }
}
