//! A confidence head that helps DFlash + Weaver spend a limited tree budget on branches the target model is likely to follow.
//!
//! DFlash proposes candidate tokens for several future positions.
//! Weaver connects those candidates into a tree, adjusting their scores to each branch's context.
//! Since the tree cannot grow indefinitely, some nodes must be prioritized for expansion and some branches must be discarded before target verification.
//!
//! ConfidenceHead estimates how likely each branch is to lie on the target model's sampled path.
//! A high score from DFlash + Weaver alone does not guarantee that the target will choose the same token.
//! The head therefore combines DFlash hidden features, Weaver-adjusted candidate logits, token features and
//! the target sampler's Gumbel noise to predict which branches are worth keeping.
//!
//! It acts at two stages:
//! - During tree growth, `encode_round` rescores the children Weaver has just selected and updates their frontier priorities.
//!   These priorities determine which nodes are expanded in the next round.
//! - After tree growth, `encode_prune` optionally uses tree attention to refine the scores using the whole built tree.
//!   The subsequent budget prune decides which nodes are sent to the target model for verification.
//!
//! For example, Weaver may prefer "buy -> coffee", while the head predicts that the target is more likely to follow "drink -> tea".
//! The latter branch can then be expanded earlier and survive the final budget prune.
//!
//! The head does not propose new tokens: Weaver still selects the children of each expanded node.
//! By changing which nodes are expanded next, however, the head changes the resulting tree.
//! Target verification remains responsible for accepting or rejecting the proposed tokens.
//!
//! The goal is to accept more tokens per target verification at the same tree budget.
//! Any speedup depends on whether those extra accepted tokens offset the head's own computation cost.
//! The head is enabled for stochastic sampling.
//! Greedy decoding, including temperature zero or top-k one, uses the ordinary DFlash + Weaver path.

mod config;
mod error;
mod head;
pub(crate) mod layout;
mod tree;
mod util;
mod weights;

use std::path::Path;

pub use error::ConfidenceHeadError;
pub use head::{ConfidenceHead, ConfidenceHeadStep};

use crate::{backends::Backend, encodable_block::conf_head::config::ConfidenceHeadConfig};

const CONF_HEAD_DIR: &str = "confidence_head";

/// Outputs of the node's output layer: alpha, beta and an unused third.
const CONF_HEAD_NODE_OUTPUTS: usize = 3;

/// Numbers per pool candidate the set encoder reads: log pi_a, log pi_l - log pi_a and the Gumbel noise.
const CONF_HEAD_POOL_INPUTS: usize = 3;

/// Standardised numbers per child: log pi_a(c), log pi_l(c), G_c and log(1 + drafter rank).
pub(crate) const CONF_HEAD_SLOT_NUMBERS: usize = 4;

pub struct ConfidenceHeadTreeParameters {
    pub rounds: u32,
    pub expand_per_round: u32,
    pub expand_width: u32,
    pub max_depth: u32,
    pub pool: u32,
}

pub fn get_conf_head<B: Backend>(
    context: &B::Context,
    speculator_dir: &Path,
    params: &ConfidenceHeadTreeParameters,
) -> Option<Result<ConfidenceHead<B>, ConfidenceHeadError<B>>> {
    let conf_head_dir = speculator_dir.join(CONF_HEAD_DIR);
    if !conf_head_dir.is_dir() {
        return None;
    }

    let config = match ConfidenceHeadConfig::new(&conf_head_dir) {
        Ok(config) => config,
        Err(err) => return Some(Err(ConfidenceHeadError::Io(err))),
    };
    if let Some(mismatch) = get_conf_head_mismatch(&config, params) {
        return Some(Err(ConfidenceHeadError::TreeMismatch(mismatch)));
    }

    Some(ConfidenceHead::new(context, &speculator_dir, &conf_head_dir, config))
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
