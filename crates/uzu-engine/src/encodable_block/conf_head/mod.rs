mod config;
mod error;
mod head;
mod tree;
mod weights;

use std::path::Path;

pub use error::ConfidenceHeadError;
pub use head::ConfidenceHead;

use crate::{backends::Backend, encodable_block::conf_head::config::ConfidenceHeadConfig};

const CONF_HEAD_DIR: &str = "confidence_head";

/// Outputs of the node's output layer: alpha, beta and an unused third.
const CONF_HEAD_NODE_OUTPUTS: usize = 3;

/// Numbers per pool candidate the set encoder reads: log pi_a, log pi_l - log pi_a and the Gumbel noise.
const CONF_HEAD_POOL_INPUTS: usize = 3;

/// Standardised numbers per child: log pi_a(c), log pi_l(c), G_c and log(1 + drafter rank).
const CONF_HEAD_SLOT_NUMBERS: usize = 4;

pub struct ConfidenceHeadTreeParameters {
    pub rounds: u32,
    pub expand_per_round: u32,
    pub expand_width: u32,
    pub max_depth: u32,
    pub pool: u32,
}

pub fn get_conf_head<B: Backend>(
    context: &B::Context,
    model_dir: &Path,
    params: &ConfidenceHeadTreeParameters,
) -> Option<Result<ConfidenceHead<B>, ConfidenceHeadError<B>>> {
    let dir = model_dir.join(CONF_HEAD_DIR);
    if !dir.is_dir() {
        return None;
    }

    Some(load_conf_head(context, &dir, params))
}

pub fn get_conf_head_mismatch(
    config: &ConfidenceHeadConfig,
    params: &ConfidenceHeadTreeParameters,
) -> Option<String> {
    let slot_count = 1 + params.rounds.saturating_sub(1) * params.expand_per_round;
    (config.children != params.expand_width
        || config.max_depth != params.max_depth
        || config.pool != params.pool
		|| !(config.tree.is_none() || slot_count * config.children < 512)
	)
    .then(|| {
        format!(
			"the confidence head was trained with {} children, depth {} and a pool of {}; the tree has {} children, \
             depth {}, a pool of {} and {} slots",
			config.children,
			config.max_depth,
			config.pool,
			params.expand_width,
			params.max_depth,
			params.pool,
			slot_count
        )
    })
}

fn load_conf_head<B: Backend>(
    context: &B::Context,
    model_dir: &Path,
    params: &ConfidenceHeadTreeParameters,
) -> Result<ConfidenceHead<B>, ConfidenceHeadError<B>> {
    let config = ConfidenceHeadConfig::new(model_dir).map_err(ConfidenceHeadError::Io)?;
    if let Some(mismatch) = get_conf_head_mismatch(&config, params) {
        return Err(ConfidenceHeadError::TreeMismatch(mismatch));
    }

    Ok(ConfidenceHead::new(context, model_dir, config)?)
}
