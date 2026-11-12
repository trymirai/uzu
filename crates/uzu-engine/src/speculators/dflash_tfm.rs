use std::{
    collections::HashMap,
    io,
    path::Path,
    sync::{Arc, mpsc::Sender},
};

use derive_more::Debug;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, Context, TimestampSpan,
    },
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        confidence_head::{ConfidenceHead, ConfidenceHeadError},
        dflash::{DFlash, DFlashEncodeError, DFlashNewError},
        embedding::Embedding,
        sampling::{PRng, Sampling, SamplingMethod},
        weaver::{ProposalNode, Weaver, WeaverEncodeError, WeaverNewError, WeaverTreeShape},
    },
    engine::language_model::grammar::Grammar,
    safetensors::{ParameterLoader, ParameterLoaderError},
    speculators::tree_budget::{BudgetLatencies, LatenciesFile, TreeBudget},
    trie::TrieNode,
    utils::{
        file,
        timestamps::{record, wait},
    },
};

#[derive(Debug, Error)]
pub enum DFlashTreeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("DFlash draft error: {0}")]
    DFlash(#[from] DFlashEncodeError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverEncodeError<B>),
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
    #[error("confidence head error: {0}")]
    ConfidenceHead(#[from] ConfidenceHeadError<B>),
    #[error("tree budget: {0}")]
    TreeBudget(String),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(tag = "type")]
pub enum DFlashTfmTreeConstructionMethod {
    Argmax,
    Weaver {
        rounds: u32,
        expand_per_round: u32,
        expand_width: u32,
        #[serde(default = "default_prune_sigma")]
        prune_sigma: Option<f32>,
    },
}

fn default_prune_sigma() -> Option<f32> {
    Some(1.5)
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct DFlashTfmTreeShape {
    pub tree_budget: TreeBudget,
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
    budget_latencies: Option<BudgetLatencies>,
}

/// This device's entry of a per-device file (`shapes.json`, `latencies.json`), else its `"default"` entry.
fn for_device<T>(
    mut entries: HashMap<String, T>,
    device_name: Option<&str>,
) -> Option<T> {
    device_name.and_then(|device_name| entries.remove(device_name)).or_else(|| entries.remove("default"))
}

impl<B: Backend> DFlashTfmSpeculator<B> {
    pub async fn new(
        model_path: &Path,
        target_embedding: &Embedding<B>,
        context: Arc<B::Context>,
    ) -> Result<Option<Self>, DFlashSpeculatorLoadError<B>> {
        let shapes: HashMap<String, DFlashTfmTreeShape> =
            serde_json::from_slice(&file::read(model_path.join("shapes.json")).await?)?;

        let Some(shape) = for_device(shapes, context.device_name()) else {
            return Ok(None);
        };

        let budget_latencies = match shape.tree_budget {
            TreeBudget::Fixed(_) => None,
            TreeBudget::Dynamic => {
                if !matches!(shape.construction_method, DFlashTfmTreeConstructionMethod::Weaver { .. }) {
                    return Err(DFlashSpeculatorLoadError::TreeBudget(
                        "a dynamic tree budget needs Weaver tree construction".to_string(),
                    ));
                }
                let latencies: LatenciesFile =
                    serde_json::from_slice(&file::read(model_path.join("latencies.json")).await?)?;
                let step_ms = for_device(latencies, context.device_name()).ok_or_else(|| {
                    DFlashSpeculatorLoadError::TreeBudget(
                        "latencies.json has no entry for this device and no \"default\" one".to_string(),
                    )
                })?;
                Some(BudgetLatencies::new(&step_ms).map_err(DFlashSpeculatorLoadError::TreeBudget)?)
            },
        };

        let data_type = DataType::BF16;

        let config: SpeculatorModelConfig = serde_json::from_slice(&file::read(model_path.join("config.json")).await?)?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;

        let weight_loader = ParameterLoader::new(model_path.join("model.safetensors"), &*context).await?;
        let speculator_tree = weight_loader.tree().subtree("speculator");

        let dflash = DFlash::new(
            data_type,
            &config.draft_config,
            target_embedding.readout_input_hadamard_factors(),
            &speculator_tree.subtree("draft_model"),
            &*context,
        )
        .await?;
        let weaver = if let Some(weaver_config) = &config.weaver_config {
            Some(
                Weaver::new(
                    weaver_config,
                    target_embedding.readout_input_hadamard_factors(),
                    &speculator_tree.subtree("weaver"),
                    &*context,
                )
                .await?,
            )
        } else {
            None
        };
        let confidence_head = match &config.confidence_head_config {
            Some(head_config) => Some(
                ConfidenceHead::new(
                    head_config,
                    &speculator_tree.subtree("confidence_head"),
                    target_embedding,
                    weaver.as_ref().map_or(0, Weaver::max_depth),
                    &*context,
                )
                .await?,
            ),
            None => None,
        };

        weight_loader.assert_all_tensors_read()?;

        let sampling = Sampling::new(config.draft_config.vocab_size, data_type);

        Ok(Some(Self {
            context,
            dflash,
            weaver,
            confidence_head,
            sampling,
            config,
            shape,
            budget_latencies,
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
    ) -> Result<(), DFlashTreeError<B>> {
        self.dflash
            .encode_accept(target_features, accepted_indices, state, command_buffer)
            .map_err(DFlashTreeError::DFlash)
    }

    /// The fixed budget, or the largest a dynamic one may choose.
    fn largest_budget(
        &self,
        budget: TreeBudget,
    ) -> Result<u32, DFlashTreeError<B>> {
        match (budget, &self.budget_latencies) {
            (TreeBudget::Fixed(budget), _) => Ok(budget),
            (TreeBudget::Dynamic, Some(latencies)) => Ok(latencies.largest()),
            (TreeBudget::Dynamic, None) => Err(DFlashTreeError::InvalidTreeShape(
                "a dynamic tree budget needs the latencies.json of a speculator loaded with it".to_string(),
            )),
        }
    }

    pub fn make_shape(
        &self,
        max_depth: Option<u32>,
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

        Some(shape)
    }

    pub async fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        shape: DFlashTfmTreeShape,
        tokens_per_pass: Option<f32>,
        sampling_method: &SamplingMethod,
        grammar: Option<&mut Grammar>,
        prng: &PRng,
        allocation_pool: Arc<B::AllocationPool>,
        command_buffer_name: Option<&str>,
        timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        let largest_budget = self.largest_budget(shape.tree_budget)?;
        assert!(largest_budget >= 2, "tree budget needs at least a root and one draft token");

        let block_size = self.dflash.block_size();
        let dflash_depth = shape.dflash_depth_override.unwrap_or(block_size);
        if !(2..=block_size).contains(&dflash_depth) {
            return Err(DFlashTreeError::InvalidTreeShape(format!(
                "dflash depth {dflash_depth} is outside 2..={block_size}"
            )));
        }

        let root_position = state.context_length();

        let command_buffer = self
            .context
            .create_command_buffer(command_buffer_name, Some(allocation_pool))
            .map_err(DFlashTreeError::Backend)?;
        let mut command_buffer = record(command_buffer, timestamps);

        let nodes = match shape.construction_method {
            DFlashTfmTreeConstructionMethod::Argmax => {
                if largest_budget > dflash_depth {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "argmax chain of {} nodes needs {} draft rows, dflash depth is {}",
                        largest_budget,
                        largest_budget - 1,
                        dflash_depth
                    )));
                }
                let chain_length = largest_budget - 1;
                let mut nodes = Vec::with_capacity(largest_budget as usize);
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
                let batch_topology = BatchTopology::flat(chain_length, true);
                let sampled = self
                    .sampling
                    .encode(
                        &dflash_output.logits,
                        None::<&B::ConstantBuffer>,
                        None::<&B::ConstantBuffer>,
                        None::<&B::GlobalGpuBuffer>,
                        None::<&B::ConstantBuffer>,
                        &batch_topology,
                        (0..chain_length).into(),
                        &SamplingMethod::Greedy,
                        &mut command_buffer,
                    )
                    .map_err(DFlashTreeError::Backend)?;
                let completed =
                    wait(command_buffer.end_encoding().submit(), timestamps).await.map_err(DFlashTreeError::Backend)?;
                let tokens = self.context.copyout::<u32>(&sampled).await.map_err(DFlashTreeError::Backend)?;
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
                let (head, prune_sigma) = match sampling_method {
                    SamplingMethod::Greedy => (None, None),
                    SamplingMethod::Stochastic {
                        ..
                    } => match &self.confidence_head {
                        Some(head) => (Some(head), None),
                        None => (None, prune_sigma),
                    },
                };
                let tree = weaver.encode_tree(
                    target_output_norm,
                    &dflash_output.draft_hidden,
                    target_embedding,
                    &dflash_output.logits,
                    &depth_seeds,
                    target_output_token,
                    WeaverTreeShape {
                        tree_budget: largest_budget,
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
                    wait(command_buffer.end_encoding().submit(), timestamps).await.map_err(DFlashTreeError::Backend)?;
                let nodes = tree.read_nodes(&self.context).await.map_err(DFlashTreeError::Backend)?;
                drop(completed);
                nodes
            },
        };

        fn recursive_build(
            nodes: &[ProposalNode],
            index: usize,
            root_position: u32,
            mut grammar: Option<&mut Grammar>,
            prng: &PRng,
        ) -> TrieNode {
            let node = &nodes[index];
            let mut trie_node = TrieNode::new(
                node.token_id as u64,
                prng.derive(root_position as u64 + node.depth as u64),
                node.logprob,
            );
            for &child_index in &node.child_indices {
                if let Some(grammar) = grammar.as_mut()
                    && (grammar.is_terminated() || grammar.accept_token(nodes[child_index].token_id as u64).is_err())
                {
                    continue;
                }

                let child = recursive_build(nodes, child_index, root_position, grammar.as_deref_mut(), prng);

                if let Some(grammar) = grammar.as_mut() {
                    grammar.rollback(1);
                }

                trie_node.add(child).expect("tree children are selected without replacement");
            }
            trie_node
        }

        let mut trie = recursive_build(&nodes, 0, root_position, grammar, prng);
        let budget = match &self.budget_latencies {
            Some(latencies) if shape.tree_budget == TreeBudget::Dynamic => latencies.choose(&trie, tokens_per_pass),
            _ => largest_budget,
        };
        trie.prune_to_budget(budget as usize);
        Ok(trie)
    }
}

#[cfg(test)]
#[path = "../../unit/speculators/dflash_tfm_test.rs"]
mod tests;
