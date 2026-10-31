use std::{
    collections::HashMap,
    io,
    mem::size_of,
    path::Path,
    sync::{Arc, mpsc::Sender},
};

use derive_more::Debug;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, Context, Kernels,
        TimestampSpan, gpu_types::trie::TrieNode as GpuTrieNode, kernel::LinearizeTreeKernel,
    },
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        dflash::{DFlash, DFlashEncodeError, DFlashNewError},
        embedding::Embedding,
        sampling::{PRng, Sampling, SamplingMethod},
        weaver::{EncodedWeaverTree, Weaver, WeaverEncodeError, WeaverNewError, WeaverTreeShape},
    },
    engine::language_model::grammar::Grammar,
    safetensors::{ParameterLoader, ParameterLoaderError},
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
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
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
        #[serde(default = "default_prune_sigma")]
        prune_sigma: Option<f32>,
    },
}

fn default_prune_sigma() -> Option<f32> {
    Some(1.5)
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct DFlashTfmTreeShape {
    pub tree_budget: u32,
    pub max_tree_depth: u32,
    pub dflash_depth_override: Option<u32>,
    pub construction_method: DFlashTfmTreeConstructionMethod,
}

type Pending<B> = <<B as Backend>::CommandBuffer as CommandBuffer>::Pending;

pub struct ProposedTree<B: Backend> {
    pub token_ids: B::GlobalGpuBuffer,
    pub seeds: B::GlobalGpuBuffer,
    pub count: B::GlobalGpuBuffer,
}

enum ProposalSource<B: Backend> {
    Argmax {
        sampled: B::GlobalGpuBuffer,
        chain_length: u32,
    },
    Weaver {
        tree: EncodedWeaverTree<B>,
        tree_slot_count: u32,
        frontier_capacity: u32,
    },
}

pub struct DFlashTfmSpeculator<B: Backend> {
    context: Arc<B::Context>,
    dflash: DFlash<B>,
    weaver: Option<Weaver<B>>,
    sampling: Sampling<B>,
    linearize_tree: <B::Kernels as Kernels>::LinearizeTreeKernel,
    config: DFlashSpeculatorConfig,
    shape: DFlashTfmTreeShape,
}

impl<B: Backend> DFlashTfmSpeculator<B> {
    pub async fn new(
        model_path: &Path,
        context: Arc<B::Context>,
    ) -> Result<Option<Self>, DFlashSpeculatorLoadError<B>> {
        let mut shapes: HashMap<String, DFlashTfmTreeShape> =
            serde_json::from_slice(&file::read(model_path.join("shapes.json")).await?)?;

        let Some(shape) = context
            .device_name()
            .and_then(|device_name| shapes.remove(device_name))
            .or_else(|| shapes.remove("default"))
        else {
            return Ok(None);
        };

        let data_type = DataType::BF16;

        let config: SpeculatorModelConfig = serde_json::from_slice(&file::read(model_path.join("config.json")).await?)?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;

        let weight_loader = ParameterLoader::new(model_path.join("model.safetensors"), &*context).await?;
        let speculator_tree = weight_loader.tree().subtree("speculator");

        let dflash =
            DFlash::new(data_type, &config.draft_config, &speculator_tree.subtree("draft_model"), &*context).await?;
        let weaver = if let Some(weaver_config) = &config.weaver_config {
            Some(
                Weaver::new(
                    config.draft_config.vocab_size,
                    weaver_config,
                    &speculator_tree.subtree("weaver"),
                    &*context,
                )
                .await?,
            )
        } else {
            None
        };

        weight_loader.assert_all_tensors_read()?;

        let sampling = Sampling::new(config.draft_config.vocab_size, data_type);

        let linearize_tree =
            <B::Kernels as Kernels>::LinearizeTreeKernel::new(&*context).map_err(DFlashSpeculatorLoadError::Backend)?;

        Ok(Some(Self {
            context,
            dflash,
            weaver,
            sampling,
            linearize_tree,
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

    pub fn encode_accept_all(
        &self,
        state: &mut DFlashState<B>,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        num_tokens: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DFlashTreeError<B>> {
        self.dflash
            .encode_accept_all(target_features, num_tokens, state, command_buffer)
            .map_err(DFlashTreeError::DFlash)
    }

    pub fn encode_accept(
        &self,
        state: &mut DFlashState<B>,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        accepted_indices: impl BufferRef<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DFlashTreeError<B>> {
        self.dflash
            .encode_accept(target_features, accepted_indices, state, command_buffer)
            .map_err(DFlashTreeError::DFlash)
    }

    pub fn commit_accept(
        &self,
        state: &mut DFlashState<B>,
        accepted_len: u32,
    ) {
        self.dflash.commit_accept(state, accepted_len);
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

    fn encode_proposal(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
        target_embedding: &Embedding<B>,
        shape: &DFlashTfmTreeShape,
        sampling_method: &SamplingMethod,
        prng: &PRng,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(ProposalSource<B>, u32), DFlashTreeError<B>> {
        assert!(shape.tree_budget >= 2, "tree budget needs at least a root and one draft token");

        let block_size = self.dflash.block_size();
        let dflash_depth = shape.dflash_depth_override.unwrap_or(block_size);
        if !(2..=block_size).contains(&dflash_depth) {
            return Err(DFlashTreeError::InvalidTreeShape(format!(
                "dflash depth {dflash_depth} is outside 2..={block_size}"
            )));
        }

        let root_position = state.context_length();

        let source = match &shape.construction_method {
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
                let dflash_output = self.dflash.encode_draft(
                    target_embedding,
                    target_output_token,
                    dflash_depth,
                    state,
                    command_buffer,
                )?;
                let batch_topology = BatchTopology::flat(chain_length);
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
                        command_buffer,
                    )
                    .map_err(DFlashTreeError::Backend)?;
                ProposalSource::Argmax {
                    sampled,
                    chain_length,
                }
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
                    command_buffer,
                )?;
                let depth_seeds = (0..weaver.max_depth())
                    .map(|depth| prng.derive(root_position as u64 + depth as u64))
                    .collect::<Box<[u64]>>();
                let prune_sigma = match sampling_method {
                    SamplingMethod::Greedy => None,
                    SamplingMethod::Stochastic {
                        ..
                    } => *prune_sigma,
                };
                let weaver_shape = WeaverTreeShape {
                    tree_budget: shape.tree_budget,
                    max_depth: shape.max_tree_depth,
                    dflash_depth,
                    rounds: *rounds,
                    expand_per_round: *expand_per_round,
                    expand_width: *expand_width,
                    prune_noise_scale: prune_sigma.map(f32::recip),
                };
                let tree_slot_count = weaver_shape.slot_count();
                let frontier_capacity = tree_slot_count * weaver_shape.expand_width;
                let tree = weaver.encode_tree(
                    target_output_norm,
                    &dflash_output.draft_hidden,
                    target_embedding,
                    &dflash_output.logits,
                    &depth_seeds,
                    target_output_token,
                    weaver_shape,
                    command_buffer,
                )?;
                ProposalSource::Weaver {
                    tree,
                    tree_slot_count,
                    frontier_capacity,
                }
            },
        };

        Ok((source, root_position))
    }

    /// Proposes a speculative tree and linearizes it on the GPU: returns the
    /// trie topology, draft token ids and per-node seeds as device buffers.
    /// Non-async — the returned command buffer is submitted but not waited on;
    /// anything encoded afterwards on the same queue observes its outputs.
    pub fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
        target_embedding: &Embedding<B>,
        shape: DFlashTfmTreeShape,
        sampling_method: &SamplingMethod,
        prng: &PRng,
        allocation_pool: Arc<B::AllocationPool>,
        command_buffer_name: Option<&str>,
        timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
    ) -> Result<(ProposedTree<B>, BatchTopology<B>, Pending<B>), DFlashTreeError<B>> {
        let command_buffer = self
            .context
            .create_command_buffer(command_buffer_name, Some(allocation_pool))
            .map_err(DFlashTreeError::Backend)?;
        let mut command_buffer = record(command_buffer, timestamps);

        let (source, root_position) = self.encode_proposal(
            state,
            target_output_norm,
            target_output_token,
            target_embedding,
            &shape,
            sampling_method,
            prng,
            &mut command_buffer,
        )?;

        let budget = shape.tree_budget;
        let context = &*self.context;
        let word_bytes = size_of::<u32>();
        let mut nodes =
            context.create_buffer_gpu(budget as usize * size_of::<GpuTrieNode>()).map_err(DFlashTreeError::Backend)?;
        let mut parents =
            context.create_buffer_gpu(budget as usize * size_of::<i32>()).map_err(DFlashTreeError::Backend)?;
        let mut heights = context.create_buffer_gpu(budget as usize * word_bytes).map_err(DFlashTreeError::Backend)?;
        let mut token_ids =
            context.create_buffer_gpu(budget as usize * word_bytes).map_err(DFlashTreeError::Backend)?;
        let mut seeds =
            context.create_buffer_gpu(budget as usize * size_of::<u64>()).map_err(DFlashTreeError::Backend)?;
        let mut count = context.create_buffer_gpu(word_bytes).map_err(DFlashTreeError::Backend)?;
        // The linearize kernel's scratch: one selection-state word per weaver
        // slot, then the DFS emission stack.
        let total_slots = match &source {
            ProposalSource::Argmax {
                ..
            } => 0,
            ProposalSource::Weaver {
                tree_slot_count,
                frontier_capacity,
                ..
            } => tree_slot_count + frontier_capacity,
        };
        let mut stack = command_buffer
            .allocate_scratch((total_slots as usize + 3 * (shape.max_tree_depth as usize + 2)) * word_bytes)
            .map_err(DFlashTreeError::Backend)?;
        let dummy = context.create_buffer_gpu(word_bytes).map_err(DFlashTreeError::Backend)?;

        let prng_seed = prng.seed();
        match &source {
            ProposalSource::Argmax {
                sampled,
                chain_length,
            } => {
                self.linearize_tree.encode(
                    &dummy,
                    &dummy,
                    sampled,
                    target_output_token,
                    &mut nodes,
                    &mut parents,
                    &mut heights,
                    &mut token_ids,
                    &mut seeds,
                    &mut count,
                    &mut stack,
                    0,
                    0,
                    budget,
                    *chain_length,
                    1,
                    prng_seed,
                    root_position,
                    &mut command_buffer,
                );
            },
            ProposalSource::Weaver {
                tree,
                tree_slot_count,
                frontier_capacity,
            } => {
                self.linearize_tree.encode(
                    tree.packed_tree(),
                    tree.frontier(),
                    &dummy,
                    target_output_token,
                    &mut nodes,
                    &mut parents,
                    &mut heights,
                    &mut token_ids,
                    &mut seeds,
                    &mut count,
                    &mut stack,
                    *tree_slot_count,
                    *frontier_capacity,
                    budget,
                    0,
                    0,
                    prng_seed,
                    root_position,
                    &mut command_buffer,
                );
            },
        }

        let batch_dim = BatchTopology::Tree {
            nodes,
            parents,
            heights,
            size: budget,
        };

        let pending = command_buffer.end_encoding().submit();
        let trie = ProposedTree {
            token_ids,
            seeds,
            count,
        };

        Ok((trie, batch_dim, pending))
    }

    /// Grammar path: waits for the proposal and builds the pruned trie on the
    /// CPU, so the grammar bitmask can be computed host-side. Slower (a device
    /// sync per decode step) but only used when a grammar is configured.
    pub async fn propose_tree_with_grammar(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
        target_embedding: &Embedding<B>,
        shape: DFlashTfmTreeShape,
        sampling_method: &SamplingMethod,
        grammar: &mut Grammar,
        prng: &PRng,
        allocation_pool: Arc<B::AllocationPool>,
        command_buffer_name: Option<&str>,
        timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        let command_buffer = self
            .context
            .create_command_buffer(command_buffer_name, Some(allocation_pool))
            .map_err(DFlashTreeError::Backend)?;
        let mut command_buffer = record(command_buffer, timestamps);

        let (source, root_position) = self.encode_proposal(
            state,
            target_output_norm,
            target_output_token,
            target_embedding,
            &shape,
            sampling_method,
            prng,
            &mut command_buffer,
        )?;

        let pending = command_buffer.end_encoding().submit();
        let completed = wait(pending, timestamps).await.map_err(DFlashTreeError::Backend)?;

        let nodes = match &source {
            ProposalSource::Argmax {
                sampled,
                chain_length,
            } => {
                let (root_token, tokens) = futures::future::try_join(
                    self.context.copyout::<u32>(target_output_token),
                    self.context.copyout::<u32>(sampled),
                )
                .await
                .map_err(DFlashTreeError::Backend)?;
                let mut nodes = Vec::with_capacity(shape.tree_budget as usize);
                nodes.push(ProposalNode {
                    token_id: root_token[0],
                    depth: 0,
                    logprob: 0.0,
                    child_indices: vec![1],
                });
                nodes.extend(tokens.into_iter().zip(1u32..).map(|(token_id, depth)| ProposalNode {
                    token_id,
                    depth,
                    logprob: 0.0,
                    child_indices: if depth < *chain_length {
                        vec![depth as usize + 1]
                    } else {
                        Vec::new()
                    },
                }));
                nodes
            },
            ProposalSource::Weaver {
                tree,
                ..
            } => tree.read_nodes(&self.context).await.map_err(DFlashTreeError::Backend)?,
        };
        drop(completed);

        let mut trie = recursive_build(&nodes, 0, root_position, Some(grammar), prng);
        trie.prune_to_budget(shape.tree_budget as usize);
        Ok(trie)
    }
}

fn recursive_build(
    nodes: &[ProposalNode],
    index: usize,
    root_position: u32,
    mut grammar: Option<&mut Grammar>,
    prng: &PRng,
) -> TrieNode {
    let node = &nodes[index];
    let mut trie_node =
        TrieNode::new(node.token_id as u64, prng.derive(root_position as u64 + node.depth as u64), node.logprob);
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

#[cfg(test)]
#[path = "../../unit/speculators/dflash_tfm_test.rs"]
mod tests;
