use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::weaver::{FRONTIER_MAX_SLOTS, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx},
        kernel::{WeaverFrontierScatterKernel, WeaverFrontierSelectKernel},
    },
    config::speculator::{AnySpeculatorConfig, dflash::DFlashSpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        dflash::{DFlashDraft, DFlashDraftNewError},
        embedding::{Embedding, EmbeddingError},
        sampling::PRng,
        weaver::{Weaver, WeaverEncodeError, WeaverNewError, WeaverNodeKvCache, WeaverPrefixKvCache, WeaverStepBatch},
    },
    engine::language_model::grammar::Grammar,
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

struct HostTreeNode {
    token: u32,
    depth: usize,
    children: Vec<usize>,
}

struct DFlashChainOutput<B: Backend> {
    pool_ids: Allocation<B>,
    pool_scores: Allocation<B>,
    draft_logits: Allocation<B>,
    draft_hidden: Allocation<B>,
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
    pub(crate) context: Arc<B::Context>,
    pub(crate) model: DFlashDraft<B>,
    pub(crate) weaver: Option<Weaver<B>>,
    pub(crate) config: DFlashSpeculatorConfig,
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

    pub fn propose_tree<'grammar>(
        &self,
        state: &mut DFlashState<B>,
        target_output_norm: &Allocation<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        prng: &PRng,
        grammar: Option<&mut (dyn Grammar + 'grammar)>,
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

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let DFlashChainOutput {
            pool_ids,
            pool_scores,
            draft_logits,
            draft_hidden,
        } = self.encode_dflash_chain(
            &mut encoder,
            state,
            target_output_norm,
            target_output_token,
            target_embedding,
            block_size,
            target_model_dim,
            pool_size,
        )?;

        if let Some(weaver) = self.weaver.as_ref() {
            let max_depth = self.config.weaver_config.as_ref().expect("a Weaver implies a Weaver config").max_depth;
            let lookahead_count = max_depth.min(block_size.saturating_sub(1));
            // The frontier holds one slot per (tree slot, child) pair; the select
            // kernel silently no-ops past its capacity.
            if (options.budget + 1) * options.children_per_node > FRONTIER_MAX_SLOTS {
                return Err(DFlashTreeError::InvalidOptions);
            }
            let depth_seeds =
                (0..max_depth).map(|depth| prng.derive((root_position + depth) as u64)).collect::<Box<[u64]>>();
            let prefix = weaver.build_prefix(target_output_norm, &draft_hidden, lookahead_count, &mut encoder)?;
            drop(draft_hidden);
            drop(draft_logits);
            let mut weaver_state = weaver.create_node_kv_cache(options.budget + 1, &self.context)?;
            let arguments = TreeEncodingArguments {
                weaver,
                prefix: &prefix,
                target_embedding,
                pool_ids: &pool_ids,
                pool_scores: &pool_scores,
                pool_rows: block_size - 1,
                pool_size,
                depth_seeds: &depth_seeds,
                options,
                max_depth,
                lookahead_count,
                target_output_token,
            };
            let tree = self.encode_tree(&mut encoder, &arguments, &mut weaver_state)?;
            let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
            let slots = tree.copyout::<u32>();
            drop(tree);
            drop(pool_ids);
            drop(pool_scores);
            drop(completed);
            return Ok(Self::finish_tree(tree_from_slots(&slots), prng, root_position, grammar));
        }

        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let pool_id_values = pool_ids.copyout::<u32>();
        drop(pool_ids);
        drop(pool_scores);
        drop(draft_logits);
        drop(draft_hidden);
        drop(completed);

        let mut nodes = vec![HostTreeNode {
            token: target_output_token,
            depth: 0,
            children: Vec::new(),
        }];
        for depth in 0..options.budget.min(block_size.saturating_sub(1)) {
            let token = pool_id_values[depth * pool_size];
            let parent = nodes.len() - 1;
            let index = nodes.len();
            nodes.push(HostTreeNode {
                token,
                depth: depth + 1,
                children: Vec::new(),
            });
            nodes[parent].children.push(index);
        }
        Ok(Self::finish_tree(nodes, prng, root_position, grammar))
    }

    fn encode_dflash_chain(
        &self,
        encoder: &mut Encoder<B>,
        state: &mut DFlashState<B>,
        target_output_norm: &Allocation<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        block_size: usize,
        target_model_dim: usize,
        pool_size: usize,
    ) -> Result<DFlashChainOutput<B>, DFlashTreeError<B>> {
        if target_output_norm.size() != target_model_dim * DataType::BF16.size_in_bytes() {
            return Err(DFlashTreeError::InvalidOptions);
        }

        let mut noise_ids =
            encoder.allocate_constant(block_size * DataType::U32.size_in_bytes()).map_err(DFlashTreeError::Backend)?;
        let mut noise = vec![self.config.draft_config.mask_token_id as u32; block_size];
        noise[0] = target_output_token;
        noise_ids.copyin(&noise);
        let token_embeddings = target_embedding.encode_lookup(&noise_ids, block_size, encoder)?;
        let draft_hidden =
            self.model.encode_block(state, token_embeddings, encoder).map_err(DFlashTreeError::Backend)?;
        // The first block row is the target's output token; only the lookahead rows are ranked.
        let row_bytes = target_embedding.model_dim() * DataType::BF16.size_in_bytes();
        let mut lookahead_hidden =
            encoder.allocate_scratch((block_size - 1) * row_bytes).map_err(DFlashTreeError::Backend)?;
        encoder.encode_copy(&draft_hidden, row_bytes..block_size * row_bytes, &mut lookahead_hidden, ..);
        let draft_logits =
            target_embedding.encode_readout(block_size - 1, &lookahead_hidden, DataType::F32, encoder)?;
        let (pool_ids, pool_scores) = self
            .model
            .encode_top_k(&draft_logits, block_size - 1, pool_size, encoder)
            .map_err(DFlashTreeError::Backend)?;
        drop(noise_ids);
        drop(lookahead_hidden);
        Ok(DFlashChainOutput {
            pool_ids,
            pool_scores,
            draft_logits,
            draft_hidden,
        })
    }

    fn finish_tree<'grammar>(
        nodes: Vec<HostTreeNode>,
        prng: &PRng,
        root_position: usize,
        mut grammar: Option<&mut (dyn Grammar + 'grammar)>,
    ) -> TrieNode {
        fn build<'grammar>(
            nodes: &[HostTreeNode],
            index: usize,
            prng: &PRng,
            root_position: usize,
            grammar: &mut Option<&mut (dyn Grammar + 'grammar)>,
        ) -> TrieNode {
            let node = &nodes[index];
            let mut trie_node = TrieNode::new(node.token as u64, prng.derive((root_position + node.depth) as u64));
            for &child_index in &node.children {
                if let Some(grammar) = grammar.as_deref_mut()
                    && grammar.accept_token(nodes[child_index].token as u64).is_err()
                {
                    continue;
                }
                let child = build(nodes, child_index, prng, root_position, grammar);
                if let Some(grammar) = grammar.as_deref_mut() {
                    grammar.rollback(1);
                }
                trie_node.add(child).expect("tree children are selected without replacement");
            }
            trie_node
        }
        build(&nodes, 0, prng, root_position, &mut grammar)
    }

    fn encode_tree(
        &self,
        encoder: &mut Encoder<B>,
        params: &TreeEncodingArguments<'_, B>,
        state: &mut WeaverNodeKvCache<B>,
    ) -> Result<Allocation<B>, DFlashTreeError<B>> {
        let context = &*self.context;
        let slots = params.options.budget + 1;
        let children_per_node = params.options.children_per_node;
        let capacity = slots * children_per_node;
        let width = params.options.frontier_width;
        let stride = params.max_depth;
        let pool_size = params.pool_size;

        let select =
            <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(context).map_err(DFlashTreeError::Backend)?;
        let scatter =
            <B::Kernels as Kernels>::WeaverFrontierScatterKernel::new(context).map_err(DFlashTreeError::Backend)?;

        let mut tree_values = vec![0u32; TreeIdx::COUNT * slots];
        for slot in 0..slots {
            tree_values[TreeIdx::ParentSlot as usize * slots + slot] = FRONTIER_NO_WINNER;
        }
        tree_values[TreeIdx::TokenId as usize * slots] = params.target_output_token;
        tree_values[TreeIdx::Valid as usize * slots] = 1;

        let mut tree = encoder.allocate_constant_from_slice(&tree_values).map_err(DFlashTreeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; FrontierIdx::COUNT * capacity])
            .map_err(DFlashTreeError::Backend)?;
        let mut slot_ancestors =
            encoder.allocate_constant_from_slice(&vec![0u32; slots * stride]).map_err(DFlashTreeError::Backend)?;

        let mut round_token_id_values = vec![0u32; width];
        round_token_id_values[0] = params.target_output_token;
        let mut round_valid_values = vec![0u32; width];
        round_valid_values[0] = 1;
        let mut round_token_ids =
            encoder.allocate_constant_from_slice(&round_token_id_values).map_err(DFlashTreeError::Backend)?;
        let mut round_metadata = encoder
            .allocate_constant_from_slice(&vec![0u32; MetadataIdx::COUNT * width])
            .map_err(DFlashTreeError::Backend)?;
        let mut round_ancestors =
            encoder.allocate_constant_from_slice(&vec![0u32; width * stride]).map_err(DFlashTreeError::Backend)?;
        let mut round_valid =
            encoder.allocate_constant_from_slice(&round_valid_values).map_err(DFlashTreeError::Backend)?;
        let mut round_candidate_ids =
            encoder.allocate_constant_from_slice(&vec![0u32; width * pool_size]).map_err(DFlashTreeError::Backend)?;
        let mut round_candidate_scores =
            encoder.allocate_constant_from_slice(&vec![0.0f32; width * pool_size]).map_err(DFlashTreeError::Backend)?;
        let depth_seeds = encoder.allocate_constant_from_slice(params.depth_seeds).map_err(DFlashTreeError::Backend)?;

        let mut slot_start = 0;
        while slot_start < slots {
            let rows = if slot_start == 0 {
                1
            } else {
                width.min(slots - slot_start)
            };
            if slot_start > 0 {
                select.encode(
                    &mut frontier,
                    &mut tree,
                    &mut slot_ancestors,
                    &mut round_token_ids,
                    &mut round_metadata,
                    &mut round_ancestors,
                    &mut round_valid,
                    params.pool_ids,
                    params.pool_scores,
                    &mut round_candidate_ids,
                    &mut round_candidate_scores,
                    capacity as u32,
                    slots as u32,
                    rows as u32,
                    slot_start as u32,
                    stride as u32,
                    params.max_depth as u32,
                    params.lookahead_count as u32,
                    params.pool_rows as u32,
                    pool_size as u32,
                    encoder,
                );
            }
            let (candidate_ids, candidate_scores) = if slot_start == 0 {
                (params.pool_ids, params.pool_scores)
            } else {
                (&round_candidate_ids, &round_candidate_scores)
            };
            let input = WeaverStepBatch {
                node_count: rows,
                candidates_per_node: pool_size,
                ancestor_stride: stride,
                node_token_ids: &round_token_ids,
                candidate_ids,
                candidate_logits: candidate_scores,
                ancestor_indices: &round_ancestors,
                node_metadata: &round_metadata,
                depth_seeds: &depth_seeds,
            };
            let children = params.weaver.encode_step_batch(
                params.prefix,
                &input,
                state,
                children_per_node,
                params.target_embedding,
                encoder,
            )?;
            scatter.encode(
                &tree,
                &round_metadata,
                &round_valid,
                &children.token_ids,
                &children.logprobs,
                &mut frontier,
                capacity as u32,
                slots as u32,
                rows as u32,
                children_per_node as u32,
                encoder,
            );
            drop(children);
            slot_start += rows;
        }
        Ok(tree)
    }
}

fn tree_from_slots(tree: &[u32]) -> Vec<HostTreeNode> {
    assert!(tree.len().is_multiple_of(TreeIdx::COUNT), "tree array must contain {} equal-length lanes", TreeIdx::COUNT);
    let slots = tree.len() / TreeIdx::COUNT;
    let field = |field: TreeIdx, slot: usize| tree[field as usize * slots + slot];
    let mut slot_to_node = vec![usize::MAX; slots];
    let mut nodes: Vec<HostTreeNode> = Vec::with_capacity(slots);
    for slot in 0..slots {
        if field(TreeIdx::Valid, slot) == 0 {
            continue;
        }
        let parent_slot = field(TreeIdx::ParentSlot, slot) as i32;
        let parent = (parent_slot >= 0).then(|| {
            let parent = slot_to_node[parent_slot as usize];
            assert_ne!(parent, usize::MAX, "tree slot {slot} names padding slot {parent_slot} as its parent");
            parent
        });
        let index = nodes.len();
        slot_to_node[slot] = index;
        if let Some(parent) = parent {
            nodes[parent].children.push(index);
        }
        nodes.push(HostTreeNode {
            token: field(TreeIdx::TokenId, slot),
            depth: field(TreeIdx::Depth, slot) as usize,
            children: Vec::new(),
        });
    }
    nodes
}

struct TreeEncodingArguments<'a, B: Backend> {
    weaver: &'a Weaver<B>,
    prefix: &'a WeaverPrefixKvCache<B>,
    target_embedding: &'a Embedding<B>,
    pool_ids: &'a Allocation<B>,
    pool_scores: &'a Allocation<B>,
    pool_rows: usize,
    pool_size: usize,
    depth_seeds: &'a [u64],
    options: DFlashTreeOptions,
    max_depth: usize,
    lookahead_count: usize,
    target_output_token: u32,
}
