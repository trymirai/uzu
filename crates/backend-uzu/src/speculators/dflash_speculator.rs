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
    backends::common::{Allocation, Backend, Encoder},
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        dflash::{DFlashDraft, DFlashDraftNewError, DFlashDraftOutput},
        embedding::{Embedding, EmbeddingError},
        sampling::PRng,
        weaver::{ProposalNode, Weaver, WeaverEncodeError, WeaverNewError, WeaverTreeInput},
    },
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
    trie::TrieNode,
};

#[derive(Clone, Copy, Debug)]
pub struct DFlashTreeOptions {
    pub budget: usize,
    pub frontier_width: usize,
    pub children_per_node: usize,
}

const MAX_TREE_BUDGET: usize = 4096;
const MAX_TREE_FRONTIER_WIDTH: usize = 8;
const MAX_CHILDREN_PER_NODE: usize = 8;

#[derive(Debug, Error)]
pub enum DFlashTreeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverEncodeError<B>),
    #[error("invalid tree options")]
    InvalidOptions,
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
    #[error("DFlash draft error: {0}")]
    Draft(#[from] DFlashDraftNewError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverNewError<B>),
    #[error("DFlash mask_token_id {mask_token_id} is outside vocabulary size {vocab_size}")]
    InvalidMaskTokenId {
        mask_token_id: u64,
        vocab_size: usize,
    },
}

pub struct DFlashSpeculator<B: Backend> {
    context: Arc<B::Context>,
    model: DFlashDraft<B>,
    weaver: Option<Weaver<B>>,
    config: DFlashSpeculatorConfig,
}

impl<B: Backend> DFlashSpeculator<B> {
    pub fn load(
        model_path: &Path,
        context: Arc<B::Context>,
    ) -> Result<Self, DFlashSpeculatorLoadError<B>> {
        let config: SpeculatorModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;
        let draft_config = &config.draft_config;
        if draft_config.mask_token_id >= draft_config.vocab_size as u64 {
            return Err(DFlashSpeculatorLoadError::InvalidMaskTokenId {
                mask_token_id: draft_config.mask_token_id,
                vocab_size: draft_config.vocab_size,
            });
        }

        let data_type = DataType::BF16;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*context)?;
        let speculator_tree = weight_loader.tree().subtree("speculator")?;
        let model =
            DFlashDraft::new(&*context, &config.draft_config, &speculator_tree.subtree("draft_model")?, data_type)?;
        let weaver = config
            .weaver_config
            .as_ref()
            .map(|weaver_config| Weaver::new(&*context, weaver_config, &speculator_tree.subtree("weaver")?))
            .transpose()?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(Self {
            context,
            model,
            weaver,
            config,
        })
    }

    pub fn hidden_feature_layer_indices(&self) -> &[usize] {
        &self.config.draft_config.target_layer_ids
    }

    pub fn empty_state(
        &self,
        context_capacity: usize,
    ) -> Result<DFlashState<B>, B::Error> {
        self.model.empty_state(context_capacity, &self.context)
    }

    pub fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.model.append_state(state, target_features, accepted_indices, encoder)
    }

    pub fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: &Allocation<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        prng: &PRng,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
        options: DFlashTreeOptions,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        if options.budget == 0
            || options.budget > MAX_TREE_BUDGET
            || options.frontier_width == 0
            || options.frontier_width > MAX_TREE_FRONTIER_WIDTH
            || options.children_per_node == 0
            || options.children_per_node > MAX_CHILDREN_PER_NODE
        {
            return Err(DFlashTreeError::InvalidOptions);
        }
        let block_size = self.model.block_size();
        let target_model_dim = self.config.draft_config.model_dim;
        let vocab_size = self.config.draft_config.vocab_size;
        let pool_size =
            self.config.weaver_config.as_ref().map_or(1, |config| config.candidate_pool_size.min(vocab_size));
        if target_output_token as usize >= vocab_size
            || pool_size == 0
            || options.children_per_node > pool_size
            || target_embedding.vocab_size() != self.config.draft_config.vocab_size
            || target_embedding.model_dim() != target_model_dim
            || self.config.weaver_config.as_ref().is_some_and(|config| config.target_model_dim != target_model_dim)
        {
            return Err(DFlashTreeError::InvalidOptions);
        }
        let root_position = state.context_length();
        if target_output_norm.size() != target_model_dim * DataType::BF16.size_in_bytes() {
            return Err(DFlashTreeError::InvalidOptions);
        }

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let DFlashDraftOutput {
            candidate_ids: pool_ids,
            candidate_scores: pool_scores,
            hidden: draft_hidden,
        } = self.model.encode_draft(state, target_output_token, target_embedding, pool_size, &mut encoder)?;

        if let Some(weaver) = self.weaver.as_ref() {
            let max_depth = self.config.weaver_config.as_ref().expect("a Weaver implies a Weaver config").max_depth;
            let depth_seeds =
                (0..max_depth).map(|depth| prng.derive((root_position + depth) as u64)).collect::<Box<[u64]>>();
            let tree = weaver.encode_tree(
                WeaverTreeInput {
                    target_hidden: target_output_norm,
                    draft_hidden: &draft_hidden,
                    target_embedding,
                    candidate_ids: &pool_ids,
                    candidate_scores: &pool_scores,
                    candidate_rows: block_size - 1,
                    candidates_per_row: pool_size,
                    depth_seeds: &depth_seeds,
                    root_token_id: target_output_token,
                    tree_budget: options.budget,
                    frontier_width: options.frontier_width,
                    children_per_node: options.children_per_node,
                },
                &self.context,
                &mut encoder,
            )?;
            let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
            let nodes = tree.decode();
            drop(pool_ids);
            drop(pool_scores);
            drop(draft_hidden);
            drop(completed);
            return Ok(Self::finish_tree(
                nodes,
                prng,
                root_position,
                #[cfg(grammar)]
                grammar,
            ));
        }

        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let pool_id_values = pool_ids.copyout::<u32>();
        drop(pool_ids);
        drop(pool_scores);
        drop(draft_hidden);
        drop(completed);

        let mut nodes = vec![ProposalNode {
            token: target_output_token,
            depth: 0,
            children: Vec::new(),
        }];
        for depth in 0..options.budget.min(block_size.saturating_sub(1)) {
            let token = pool_id_values[depth * pool_size];
            let parent = nodes.len() - 1;
            let index = nodes.len();
            nodes.push(ProposalNode {
                token,
                depth: depth + 1,
                children: Vec::new(),
            });
            nodes[parent].children.push(index);
        }
        Ok(Self::finish_tree(
            nodes,
            prng,
            root_position,
            #[cfg(grammar)]
            grammar,
        ))
    }

    fn finish_tree(
        nodes: Vec<ProposalNode>,
        prng: &PRng,
        root_position: usize,
        #[cfg(grammar)] mut grammar: Option<&mut Grammar>,
    ) -> TrieNode {
        fn build(
            nodes: &[ProposalNode],
            index: usize,
            prng: &PRng,
            root_position: usize,
            #[cfg(grammar)] grammar: &mut Option<&mut Grammar>,
        ) -> TrieNode {
            let node = &nodes[index];
            let mut trie_node = TrieNode::new(node.token as u64, prng.derive((root_position + node.depth) as u64));
            for &child_index in &node.children {
                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut()
                    && grammar.accept_token(nodes[child_index].token as u64).is_err()
                {
                    continue;
                }
                let child = build(
                    nodes,
                    child_index,
                    prng,
                    root_position,
                    #[cfg(grammar)]
                    grammar,
                );
                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut() {
                    grammar.rollback(1);
                }
                trie_node.add(child).expect("tree children are selected without replacement");
            }
            trie_node
        }
        build(
            &nodes,
            0,
            prng,
            root_position,
            #[cfg(grammar)]
            &mut grammar,
        )
    }
}
