use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
#[cfg(grammar)]
use crate::engine::language_model::grammar::Grammar;
use crate::{
    backends::common::{Allocation, AllocationPool, Backend, Encoder, gpu_types::trie::TrieNode as GpuTrieNode},
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        dflash::{DFlash, DFlashEncodeError, DFlashNewError},
        embedding::Embedding,
        sampling::{PRng, Sampling, SamplingMethod},
        weaver::{ProposalNode, Weaver, WeaverEncodeError, WeaverNewError, WeaverTreeShape},
    },
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
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
    #[error("invalid tree shape: {0}")]
    InvalidTreeShape(String),
}

#[derive(Debug, Error)]
pub enum DFlashSpeculatorLoadError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("HeaderLoading error: {0}")]
    HeaderLoading(#[from] HeaderLoadingError),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("DFlash error: {0}")]
    DFlash(#[from] DFlashNewError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverNewError<B>),
}

pub enum DFlashTfmTreeConstructionMethod {
    Argmax,
    Weaver {
        depth: u32,
        expand_per_round: u32,
        expand_width: u32,
    },
}

pub struct DFlashTfmTreeShape {
    pub budget: u32,
    pub construction_method: DFlashTfmTreeConstructionMethod,
}

pub struct DFlashTfmSpeculator<B: Backend> {
    context: Arc<B::Context>,
    dflash: DFlash<B>,
    weaver: Option<Weaver<B>>,
    sampling: Sampling<B>,
    config: DFlashSpeculatorConfig,
}

impl<B: Backend> DFlashTfmSpeculator<B> {
    pub fn new(
        model_path: &Path,
        context: Arc<B::Context>,
    ) -> Result<Self, DFlashSpeculatorLoadError<B>> {
        let data_type = DataType::BF16;

        let config: SpeculatorModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*context)?;
        let speculator_tree = weight_loader.tree().subtree("speculator");

        let dflash = DFlash::new(&*context, &config.draft_config, &speculator_tree.subtree("draft_model"), data_type)?;
        let weaver = config
            .weaver_config
            .as_ref()
            .map(|weaver_config| {
                Weaver::new(
                    &*context,
                    weaver_config,
                    config.draft_config.vocab_size,
                    &speculator_tree.subtree("weaver"),
                )
            })
            .transpose()?;

        weight_loader.tree().assert_all_tensors_validated()?;

        let sampling = Sampling::new(DataType::F32, config.draft_config.vocab_size);

        Ok(Self {
            context,
            dflash,
            weaver,
            sampling,
            config,
        })
    }

    pub fn has_weaver(&self) -> bool {
        self.weaver.is_some()
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
        target_features: &[Allocation<B>],
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.dflash.encode_accept(state, target_features, accepted_indices, encoder)
    }

    pub fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: &Allocation<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        shape: DFlashTfmTreeShape,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
        prng: &PRng,
        allocation_pool: Arc<AllocationPool<B>>,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        assert!(shape.budget >= 2, "tree budget needs at least a root and one draft token");

        let root_position = state.context_length();

        let mut encoder = Encoder::new_with_pool_name(&*self.context, allocation_pool, Some("speculator propose"))
            .map_err(DFlashTreeError::Backend)?;

        let nodes = match shape.construction_method {
            DFlashTfmTreeConstructionMethod::Argmax => {
                if shape.budget > self.dflash.block_size() {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "argmax chain of {} nodes needs {} draft rows, block size is {}",
                        shape.budget,
                        shape.budget - 1,
                        self.dflash.block_size()
                    )));
                }
                let chain_length = shape.budget - 1;
                let mut nodes = Vec::with_capacity(shape.budget as usize);
                nodes.push(ProposalNode {
                    token_id: target_output_token,
                    depth: 0,
                    child_indices: vec![1],
                });
                let dflash_output = self.dflash.encode_draft(
                    state,
                    target_output_token,
                    target_embedding,
                    shape.budget,
                    &mut encoder,
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
                        None,
                        None,
                        None,
                        None,
                        &SamplingMethod::Greedy,
                        &batch_topology,
                        0..chain_length as usize,
                        &mut encoder,
                    )
                    .map_err(DFlashTreeError::Backend)?;
                let completed =
                    encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
                let tokens = sampled.copyout::<u32>();
                drop(completed);
                nodes.extend(tokens.into_iter().enumerate().map(|(index, token_id)| {
                    let depth = index as u32 + 1;
                    ProposalNode {
                        token_id,
                        depth,
                        child_indices: if depth < chain_length {
                            vec![index + 2]
                        } else {
                            Vec::new()
                        },
                    }
                }));
                nodes
            },
            DFlashTfmTreeConstructionMethod::Weaver {
                depth,
                expand_per_round,
                expand_width,
            } => {
                let weaver =
                    self.weaver.as_ref().expect("weaver tree construction requires a speculator with weaver weights");
                if depth < 2 {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "tree depth {depth} needs at least a root and one level"
                    )));
                }
                // `depth` counts the root; the weaver's `max_depth` counts edges.
                if depth > weaver.max_depth() + 1 {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "requested tree depth {depth} exceeds weaver max depth {}",
                        weaver.max_depth() + 1
                    )));
                }
                if depth > self.dflash.block_size() {
                    return Err(DFlashTreeError::InvalidTreeShape(format!(
                        "tree of depth {depth} needs {} draft rows, block size is {}",
                        depth - 1,
                        self.dflash.block_size()
                    )));
                }
                let dflash_output =
                    self.dflash.encode_draft(state, target_output_token, target_embedding, depth, &mut encoder)?;
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
                        budget: shape.budget,
                        depth,
                        expand_per_round,
                        expand_width,
                    },
                    &mut encoder,
                )?;
                let completed =
                    encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
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
            let mut trie_node =
                TrieNode::new(node.token_id as u64, prng.derive(root_position as u64 + node.depth as u64));
            for &child_index in &node.child_indices {
                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut()
                    && grammar.accept_token(nodes[child_index].token_id as u64).is_err()
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

        Ok(recursive_build(
            &nodes,
            0,
            root_position,
            #[cfg(grammar)]
            grammar,
            prng,
        ))
    }
}
