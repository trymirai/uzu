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
        dflash::{DFlashDraft, DFlashDraftEncodeError, DFlashDraftNewError, DFlashDraftOutput},
        embedding::Embedding,
        sampling::PRng,
        weaver::{CandidatePool, ProposalNode, TreeShape, Weaver, WeaverEncodeError, WeaverInputs, WeaverNewError},
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
    #[error("DFlash draft error: {0}")]
    Draft(#[from] DFlashDraftEncodeError<B>),
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
        let vocab_size = self.config.draft_config.vocab_size;
        let target_model_dim = self.config.draft_config.model_dim;
        let candidates_per_row =
            self.config.weaver_config.as_ref().map_or(1, |config| config.candidate_pool_size.min(vocab_size));
        if target_output_token as usize >= vocab_size
            || candidates_per_row == 0
            || options.children_per_node > candidates_per_row
            || target_embedding.vocab_size() != vocab_size
            || target_embedding.model_dim() != target_model_dim
            || target_output_norm.size() != target_model_dim * DataType::BF16.size_in_bytes()
            || self.config.weaver_config.as_ref().is_some_and(|config| config.target_model_dim != target_model_dim)
        {
            return Err(DFlashTreeError::InvalidOptions);
        }
        let root_position = state.context_length();

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let DFlashDraftOutput {
            candidates,
            draft_hidden,
        } = self.model.encode_draft(state, target_output_token, target_embedding, candidates_per_row, &mut encoder)?;

        if let Some(weaver) = self.weaver.as_ref() {
            let max_depth = self.config.weaver_config.as_ref().expect("a Weaver implies a Weaver config").max_depth;
            let depth_seeds =
                (0..max_depth).map(|depth| prng.derive((root_position + depth) as u64)).collect::<Box<[u64]>>();
            let tree = weaver.encode_tree(
                WeaverInputs::new(
                    target_output_norm,
                    &draft_hidden,
                    target_embedding,
                    CandidatePool::new(
                        &candidates.ids,
                        &candidates.scores,
                        candidates.rows,
                        candidates.candidates_per_row,
                    ),
                    &depth_seeds,
                    target_output_token,
                ),
                TreeShape::new(options.budget, options.frontier_width, options.children_per_node),
                &self.context,
                &mut encoder,
            )?;
            let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
            let nodes = tree.read_nodes();
            drop(candidates);
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
        let candidate_rows = candidates.rows;
        let candidates_per_row = candidates.candidates_per_row;
        let candidate_id_values = candidates.ids.copyout::<u32>();
        drop(candidates);
        drop(draft_hidden);
        drop(completed);

        let mut nodes = vec![ProposalNode {
            token_id: target_output_token,
            depth: 0,
            child_indices: Vec::new(),
        }];
        for depth in 0..options.budget.min(candidate_rows) {
            let token = candidate_id_values[depth * candidates_per_row];
            let parent = nodes.len() - 1;
            let index = nodes.len();
            nodes.push(ProposalNode {
                token_id: token,
                depth: depth + 1,
                child_indices: Vec::new(),
            });
            nodes[parent].child_indices.push(index);
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
            let mut trie_node = TrieNode::new(node.token_id as u64, prng.derive((root_position + node.depth) as u64));
            for &child_index in &node.child_indices {
                #[cfg(grammar)]
                if let Some(grammar) = grammar.as_mut()
                    && grammar.accept_token(nodes[child_index].token_id as u64).is_err()
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
