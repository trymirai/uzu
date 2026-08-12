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
    backends::common::{Allocation, AllocationPool, Backend, Encoder},
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        dflash::{DFlash, DFlashEncodeError, DFlashNewError},
        embedding::Embedding,
        sampling::PRng,
        weaver::{ProposalNode, TreeShape, Weaver, WeaverEncodeError, WeaverNewError},
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

pub struct DFlashTfmSpeculator<B: Backend> {
    context: Arc<B::Context>,
    dflash: DFlash<B>,
    weaver: Weaver<B>,
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
        let weaver = Weaver::new(&*context, &config.weaver_config, &speculator_tree.subtree("weaver"))?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(Self {
            context,
            dflash,
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
        self.dflash.empty_state(context_capacity, &self.context)
    }

    pub fn encode_accept(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        accepted_indices: &[usize],
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
        shape: TreeShape,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
        prng: &PRng,
        allocation_pool: Arc<AllocationPool<B>>,
    ) -> Result<TrieNode, DFlashTreeError<B>> {
        let candidate_pool_size = self.config.weaver_config.candidate_pool_size;

        let root_position = state.context_length();

        let mut encoder = Encoder::new_with_pool_name(&*self.context, allocation_pool, Some("speculator propose"))
            .map_err(DFlashTreeError::Backend)?;

        let dflash_output = self.dflash.encode_draft(
            state,
            target_output_token,
            target_embedding,
            candidate_pool_size,
            &mut encoder,
        )?;

        let max_depth = self.config.weaver_config.max_depth;
        let depth_seeds =
            (0..max_depth).map(|depth| prng.derive((root_position + depth) as u64)).collect::<Box<[u64]>>();
        let tree = self.weaver.encode_tree(
            target_output_norm,
            &dflash_output.draft_hidden,
            target_embedding,
            &dflash_output.candidates,
            &depth_seeds,
            target_output_token,
            shape,
            &mut encoder,
        )?;
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let nodes = tree.read_nodes();
        drop(dflash_output);
        drop(completed);

        fn recursive_build(
            nodes: &[ProposalNode],
            index: usize,
            root_position: usize,
            #[cfg(grammar)] mut grammar: Option<&mut Grammar>,
            prng: &PRng,
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
