use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use derive_more::Debug;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
#[cfg(grammar)]
use crate::engine::language_model::grammar::Grammar;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        Context, gpu_types::trie::TrieNode as GpuTrieNode,
    },
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        confidence_head::{ConfidenceHead, ConfidenceHeadConfig, ConfidenceHeadError, file_sha256},
        dflash::{DFlash, DFlashEncodeError, DFlashNewError},
        embedding::Embedding,
        sampling::{PRng, Sampling, SamplingMethod},
        weaver::{ProposalNode, Weaver, WeaverEncodeError, WeaverNewError, WeaverTreeShape},
    },
    parameters::{ParameterLoader, ParameterLoaderError},
    trie::TrieNode,
};

#[derive(Debug, Error)]
pub enum DFlashTreeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("DFlash draft error: {0}")]
    DFlash(#[from] DFlashEncodeError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverEncodeError<B>),
    #[error("confidence head error: {0}")]
    ConfidenceHead(#[from] ConfidenceHeadError<B>),
    #[error("invalid tree shape: {0}")]
    InvalidTreeShape(String),
}

#[derive(Debug, Error)]
pub enum DFlashSpeculatorLoadError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("DFlash error: {0}")]
    DFlash(#[from] DFlashNewError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverNewError<B>),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(tag = "type")]
pub enum DFlashTfmTreeConstructionMethod {
    Argmax,
    Weaver {
        rounds: u32,
        expand_per_round: u32,
        expand_width: u32,
        /// A shape without it prunes with `DEFAULT_PRUNE_SIGMA`; `null` prunes on the model logprobs.
        #[serde(default = "default_prune_sigma")]
        prune_sigma: Option<f32>,
    },
}

/// The final pruning noise scale calibrated for DFlash + Weaver trees.
pub const DEFAULT_PRUNE_SIGMA: f32 = 1.5;

fn default_prune_sigma() -> Option<f32> {
    Some(DEFAULT_PRUNE_SIGMA)
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct DFlashTfmTreeShape {
    pub tree_budget: u32,
    pub max_tree_depth: u32,
    pub dflash_depth_override: Option<u32>,
    pub construction_method: DFlashTfmTreeConstructionMethod,
}

pub struct DFlashTfmSpeculator<B: Backend> {
    context: Arc<B::Context>,
    dflash: DFlash<B>,
    weaver: Option<Weaver<B>>,
    confidence_head: Option<ConfidenceHead<B>>,
    sampling: Sampling<B>,
    config: DFlashSpeculatorConfig,
    shape: DFlashTfmTreeShape,
}

impl<B: Backend> DFlashTfmSpeculator<B> {
    pub fn new(
        model_path: &Path,
        context: Arc<B::Context>,
    ) -> Result<Option<Self>, DFlashSpeculatorLoadError<B>> {
        let mut shapes = serde_json::from_reader::<_, HashMap<String, DFlashTfmTreeShape>>(BufReader::new(
            File::open(model_path.join("shapes.json"))?,
        ))?;

        let Some(shape) = context
            .device_name()
            .and_then(|device_name| shapes.remove(device_name))
            .or_else(|| shapes.remove("default"))
        else {
            return Ok(None);
        };

        let data_type = DataType::BF16;

        let config: SpeculatorModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*context)?;
        let speculator_tree = weight_loader.tree().subtree("speculator");

        let dflash = DFlash::new(data_type, &config.draft_config, &speculator_tree.subtree("draft_model"), &*context)?;
        let weaver = config
            .weaver_config
            .as_ref()
            .map(|weaver_config| {
                Weaver::new(
                    config.draft_config.vocab_size,
                    weaver_config,
                    &speculator_tree.subtree("weaver"),
                    &*context,
                )
            })
            .transpose()?;

        weight_loader.tree().assert_all_tensors_validated()?;

        // A confidence head in `confidence_head/` next to the weights grows and prunes the Weaver tree in place of the
        // path weights and prune_sigma. It is dropped, with a warning, when it cannot load, does not fit the tree or
        // was trained with another speculator file: the speculator then runs as it would without it.
        let confidence_head = match (&shape.construction_method, weaver.as_ref()) {
            (
                DFlashTfmTreeConstructionMethod::Weaver {
                    rounds,
                    expand_per_round,
                    expand_width,
                    ..
                },
                Some(weaver),
            ) => load_confidence_head(
                model_path,
                HeadTree {
                    rounds: *rounds,
                    expand_per_round: *expand_per_round,
                    expand_width: *expand_width,
                    max_depth: weaver.max_depth(),
                    pool: weaver.candidate_pool_size(),
                },
                &*context,
            ),
            _ => None,
        };

        let sampling = Sampling::new(config.draft_config.vocab_size, data_type);

        Ok(Some(Self {
            context,
            dflash,
            weaver,
            confidence_head,
            sampling,
            config,
            shape,
        }))
    }

    pub fn hidden_feature_layer_indices(&self) -> &[u32] {
        &self.config.draft_config.target_layer_ids
    }

    pub fn empty_state(
        &self,
        context_capacity: u32,
    ) -> Result<DFlashState<B>, B::Error> {
        self.dflash.empty_state(context_capacity, &self.context)
    }

    pub fn encode_accept(
        &self,
        state: &mut DFlashState<B>,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.dflash.encode_accept(target_features, accepted_indices, state, command_buffer)
    }

    pub fn make_shape(
        &self,
        max_depth: Option<u32>,
        sampling_method: &SamplingMethod,
    ) -> Option<DFlashTfmTreeShape> {
        if max_depth.is_some_and(|max_depth| max_depth < 2) {
            return None;
        }

        let mut shape = self.shape.clone();

        if let Some(max_depth) = max_depth {
            shape.max_tree_depth = u32::min(shape.max_tree_depth, max_depth);

            if let DFlashTfmTreeConstructionMethod::Weaver {
                rounds,
                ..
            } = &mut shape.construction_method
            {
                *rounds = u32::min(*rounds, max_depth);
            }
        }

        // Greedy verification adds no Gumbel noise to the target logits, so pruning has none to anticipate.
        if matches!(sampling_method, SamplingMethod::Greedy)
            && let DFlashTfmTreeConstructionMethod::Weaver {
                prune_sigma,
                ..
            } = &mut shape.construction_method
        {
            *prune_sigma = None;
        }

        Some(shape)
    }

    pub fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        shape: DFlashTfmTreeShape,
        sampling_method: &SamplingMethod,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
        prng: &PRng,
        allocation_pool: Arc<B::AllocationPool>,
        command_buffer_name: Option<&str>,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        assert!(shape.tree_budget >= 2, "tree budget needs at least a root and one draft token");

        let block_size = self.dflash.block_size();
        let dflash_depth = shape.dflash_depth_override.unwrap_or(block_size);
        if !(2..=block_size).contains(&dflash_depth) {
            return Err(DFlashTreeError::InvalidTreeShape(format!(
                "dflash depth {dflash_depth} is outside 2..={block_size}"
            )));
        }

        let root_position = state.context_length();

        let mut command_buffer = self
            .context
            .create_command_buffer(command_buffer_name, Some(allocation_pool))
            .map_err(DFlashTreeError::Backend)?;

        let nodes = match shape.construction_method {
            DFlashTfmTreeConstructionMethod::Argmax => {
                if shape.tree_budget > dflash_depth {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "argmax chain of {} nodes needs {} draft rows, dflash depth is {}",
                        shape.tree_budget,
                        shape.tree_budget - 1,
                        dflash_depth
                    )));
                }
                let chain_length = shape.tree_budget - 1;
                let mut nodes = Vec::with_capacity(shape.tree_budget as usize);
                nodes.push(ProposalNode {
                    token_id: target_output_token,
                    depth: 0,
                    logprob: 0.0,
                    child_indices: vec![1],
                });
                let dflash_output = self.dflash.encode_draft(
                    target_embedding,
                    target_output_token,
                    dflash_depth,
                    state,
                    &mut command_buffer,
                )?;
                let topology_nodes = (0..chain_length)
                    .map(|index| GpuTrieNode {
                        trie_start: index,
                        trie_end: chain_length - 1,
                        height: index,
                    })
                    .collect::<Box<[_]>>();
                let batch_topology = BatchTopology::new(&topology_nodes, true);
                let sampled = self
                    .sampling
                    .encode(
                        &dflash_output.logits,
                        None::<&B::ConstantBuffer>,
                        None::<&B::ConstantBuffer>,
                        None::<&B::GlobalBuffer>,
                        None::<&B::ConstantBuffer>,
                        &batch_topology,
                        (0..chain_length).into(),
                        &SamplingMethod::Greedy,
                        &mut command_buffer,
                    )
                    .map_err(DFlashTreeError::Backend)?;
                let completed =
                    command_buffer.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
                let tokens = sampled.copyout::<u32>();
                drop(completed);
                nodes.extend(tokens.into_iter().zip(1u32..).map(|(token_id, depth)| ProposalNode {
                    token_id,
                    depth,
                    logprob: 0.0,
                    child_indices: if depth < chain_length {
                        vec![depth as usize + 1]
                    } else {
                        Vec::new()
                    },
                }));
                nodes
            },
            DFlashTfmTreeConstructionMethod::Weaver {
                rounds,
                expand_per_round,
                expand_width,
                prune_sigma,
            } => {
                let weaver =
                    self.weaver.as_ref().expect("weaver tree construction requires a speculator with weaver weights");
                // With a confidence head the tree grows by the head's log r and is pruned by its edges, prune_sigma
                // unused, unless the sampler is greedy in effect: the head predicts the target's sampled choice. The
                // head was checked against the loaded shape; a request's shape only has fewer rounds.
                let head = match self.confidence_head.as_ref() {
                    Some(head) if ConfidenceHead::<B>::applies_to(sampling_method) => {
                        let tree = HeadTree {
                            rounds,
                            expand_per_round,
                            expand_width,
                            max_depth: weaver.max_depth(),
                            pool: weaver.candidate_pool_size(),
                        };
                        if let Some(problem) = head_tree_mismatch(head.config(), tree) {
                            return Err(DFlashTreeError::InvalidTreeShape(problem));
                        }
                        Some((head, head.token_table(target_embedding, &self.context)?))
                    },
                    _ => None,
                };
                // `max_depth` counts the root; the weaver's `max_depth` counts edges.
                if shape.max_tree_depth < 2 || shape.max_tree_depth > weaver.max_depth() + 1 {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "tree max_depth {} is outside 2..={}",
                        shape.max_tree_depth,
                        weaver.max_depth() + 1
                    )));
                }
                if shape.max_tree_depth > dflash_depth {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "tree of max_depth {} needs {} draft rows, dflash depth is {}",
                        shape.max_tree_depth,
                        shape.max_tree_depth - 1,
                        dflash_depth
                    )));
                }
                if let Some(prune_sigma) = prune_sigma
                    && !(prune_sigma > 0.0 && prune_sigma.is_finite() && prune_sigma.recip().is_finite())
                {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "prune sigma {prune_sigma} is not positive and finite"
                    )));
                }
                let dflash_output = self.dflash.encode_draft(
                    target_embedding,
                    target_output_token,
                    dflash_depth,
                    state,
                    &mut command_buffer,
                )?;
                let depth_seeds = (0..weaver.max_depth())
                    .map(|depth| prng.derive(root_position as u64 + depth as u64))
                    .collect::<Box<[u64]>>();
                let tree = weaver.encode_tree(
                    target_output_norm,
                    &dflash_output.draft_hidden,
                    target_embedding,
                    &dflash_output.logits,
                    &depth_seeds,
                    target_output_token,
                    WeaverTreeShape {
                        tree_budget: shape.tree_budget,
                        max_depth: shape.max_tree_depth,
                        dflash_depth,
                        rounds,
                        expand_per_round,
                        expand_width,
                        prune_noise_scale: prune_sigma.map(f32::recip),
                    },
                    head,
                    &mut command_buffer,
                )?;
                let completed =
                    command_buffer.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
                let nodes = tree.read_nodes();
                drop(completed);
                nodes
            },
        };

        fn recursive_build(
            nodes: &[ProposalNode],
            index: usize,
            root_position: u32,
            #[cfg(grammar)] mut grammar: Option<&mut Grammar>,
            prng: &PRng,
        ) -> TrieNode {
            let node = &nodes[index];
            let mut trie_node = TrieNode::new(
                node.token_id as u64,
                prng.derive(root_position as u64 + node.depth as u64),
                node.logprob,
            );
            for &child_index in &node.child_indices {
                // Grammar-illegal proposals are dropped; nothing may be proposed
                // past termination either, since only stop tokens can follow.
                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut()
                    && (grammar.is_terminated() || grammar.accept_token(nodes[child_index].token_id as u64).is_err())
                {
                    continue;
                }

                let child = recursive_build(
                    nodes,
                    child_index,
                    root_position,
                    #[cfg(grammar)]
                    grammar.as_deref_mut(),
                    prng,
                );

                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut() {
                    grammar.rollback(1);
                }

                trie_node.add(child).expect("tree children are selected without replacement");
            }
            trie_node
        }

        let mut trie = recursive_build(
            &nodes,
            0,
            root_position,
            #[cfg(grammar)]
            grammar,
            prng,
        );
        trie.prune_to_budget(shape.tree_budget as usize);
        Ok(trie)
    }
}

/// The tree a confidence head runs on: the shape's rounds and expansions, the Weaver's depth and pool.
#[derive(Debug, Clone, Copy)]
struct HeadTree {
    rounds: u32,
    expand_per_round: u32,
    expand_width: u32,
    max_depth: u32,
    pool: u32,
}

/// Why a head cannot run on a tree, if it cannot: other children per expansion, depth or pool than it was trained
/// with, or more construction slots than its attention takes.
fn head_tree_mismatch(
    config: &ConfidenceHeadConfig,
    tree: HeadTree,
) -> Option<String> {
    let slot_count = 1 + tree.rounds.saturating_sub(1) * tree.expand_per_round;
    (config.children != tree.expand_width
        || config.max_depth != tree.max_depth
        || config.pool != tree.pool
        || !config.fits(slot_count))
    .then(|| {
        format!(
            "the confidence head was trained with {} children, depth {} and a pool of {}; the tree has {} children, \
             depth {}, a pool of {} and {} slots",
            config.children, config.max_depth, config.pool, tree.expand_width, tree.max_depth, tree.pool, slot_count
        )
    })
}

/// The confidence head in `<speculator>/confidence_head/`, if there is one that loads, fits the tree and was trained
/// with this speculator's weights file (its sha256 against the head's `trained_with.speculator_sha256`). Otherwise
/// none, with a warning. The checks on the config come first and the head is built last, so a head that is dropped
/// costs at most the hash of the weights file, a fraction of a second.
fn load_confidence_head<B: Backend>(
    model_path: &Path,
    tree: HeadTree,
    context: &B::Context,
) -> Option<ConfidenceHead<B>> {
    let directory = model_path.join(CONFIDENCE_HEAD_DIRECTORY);
    if !directory.is_dir() {
        return None;
    }
    let config = match ConfidenceHead::<B>::read_config(&directory) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(%error, "confidence head not loaded; the speculator runs without it");
            return None;
        },
    };
    if let Some(problem) = head_tree_mismatch(&config, tree) {
        tracing::warn!(problem, "confidence head does not fit the tree; the speculator runs without it");
        return None;
    }
    match file_sha256(&model_path.join("model.safetensors")) {
        Ok(sha256) if sha256 == config.speculator_sha256 => {},
        Ok(sha256) => {
            tracing::warn!(
                trained_with = %config.speculator_sha256,
                speculator = %sha256,
                "confidence head was trained with another speculator; the speculator runs without it"
            );
            return None;
        },
        Err(error) => {
            tracing::warn!(%error, "speculator weights not hashed; the speculator runs without the confidence head");
            return None;
        },
    }
    tracing::debug!(sampling = ?config.sampling, "confidence head trained under this sampler");
    ConfidenceHead::with_config(config, &directory, context)
        .inspect_err(|error| tracing::warn!(%error, "confidence head not loaded; the speculator runs without it"))
        .ok()
}

/// The directory of a speculator's confidence head, next to its weights.
pub const CONFIDENCE_HEAD_DIRECTORY: &str = "confidence_head";

#[cfg(test)]
#[path = "../../unit/speculators/dflash_tfm_test.rs"]
mod tests;

#[cfg(test)]
#[path = "../../unit/speculators/dflash_tfm_head_test.rs"]
mod head_tests;
