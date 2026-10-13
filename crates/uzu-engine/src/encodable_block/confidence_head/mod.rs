//! A confidence head for the Weaver tree: it scores every node n by log r(n), its estimate of the log-probability
//! that n lies on the target's path, and the tree both grows and is pruned by that score.
//!
//! For an expanded node u the head predicts the target's choice over u's candidate pool,
//!
//!     log q_u(t) = log_softmax_t(alpha_u * l_ut + beta_u * G_t + delta_ut),
//!
//! with l = a + rho the pool logits the children were chosen by, G the target sampler's Gumbel noise of the pool
//! row, alpha_u and beta_u from the node's vector z_u and delta_ut non-zero only on u's children. log r of a child is
//! its parent's plus min(log q, 0). z_u sums the projected drafter row of u's depth and a set encoder over u's pool;
//! a child's delta comes from an edge MLP over [z_u; e_c; z_u * e_c; slot numbers], e_c the child token's projected
//! target input embedding.
//!
//! The local part (no tree attention) reads only the node's own drafter row, pool and children, so every round
//! computes it for the nodes it has just expanded (`encode_round`): the children's edges go to the frontier, which
//! adds them to the parent's path and ranks by the sum. The full head then runs attention over all nodes of the
//! tree (`encode_prune`) and rewrites every edge the final prune sums along paths.

pub mod layout;

use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::OnceLock,
};

use derive_more::Debug;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use self::layout::{LocalLayout, NODE_OUTPUTS, POOL_INPUTS, SLOT_NUMBERS};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels,
        gpu_types::activation_type::ActivationType,
        kernel::{
            ActivationKernel, ConfidenceHeadPoolNoiseKernel, ConfidenceHeadProjectRowsKernel,
            ConfidenceHeadPruneKernel, ConfidenceHeadRoundKernel, ConfidenceHeadTreeAttentionKernel,
            ConfidenceHeadTreeTokensKernel,
        },
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        sampling::SamplingMethod,
    },
    parameters::{ParameterLoader, ParameterLoaderError, ParameterTree},
};

pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const FORMAT: u32 = 2;
/// Tree relations the attention biases: self, parent, child, ancestor two or more edges up, descendant likewise,
/// sibling.
pub const RELATIONS: u32 = 6;
/// Limits of the fused kernels (one threadgroup per node, two pool candidates per thread).
pub const CANDIDATES_MAX: u32 = 512;
pub const HIDDEN_MAX: u32 = 64;
pub const CHILDREN_MAX: u32 = 4;
pub const POOL_HIDDEN_MAX: u32 = 16;
pub const EDGE_HIDDEN_MAX: u32 = 128;
pub const TOKENS_MAX: u32 = 512;
/// The tree attention keeps a tile of keys per head in threadgroup memory.
pub const HEAD_DIM_MAX: u32 = 64;
pub const LEVELS_MAX: u32 = 32;
/// Rows of the target embedding projected per pass when the token table is built.
const TABLE_CHUNK: u32 = 4096;

/// The tree attention's sizes (`attention` in `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct ConfidenceHeadTreeConfig {
    pub layers: u32,
    pub heads: u32,
    pub width: u32,
    pub mlp_hidden: u32,
    pub relations: u32,
}

/// The target's sampler of the rollouts the head learned from (`trained_with.sampling`): where its gain was measured.
/// It does not gate the head; a sampler that is greedy in effect does (`ConfidenceHead::applies_to`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ConfidenceHeadSampling {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

/// A head's configuration. `config.json` (format 2) groups it: what the head was trained with (`trained_with`: the
/// sha256 of the speculator weights file, the tree shape, the sampler), then its own sizes (`target_width`, `local`,
/// `attention`, `layer_norm_eps`).
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "ConfidenceHeadFile")]
pub struct ConfidenceHeadConfig {
    pub format: u32,
    /// sha256 of the speculator's `model.safetensors` the head was trained with, 64 lowercase hex digits.
    pub speculator_sha256: String,
    pub sampling: ConfidenceHeadSampling,
    pub hidden: u32,
    pub pool_hidden: u32,
    pub edge_hidden: u32,
    pub children: u32,
    pub pool: u32,
    pub max_depth: u32,
    pub model_width: u32,
    pub cap: f32,
    pub layer_norm_eps: f32,
    pub tree: Option<ConfidenceHeadTreeConfig>,
}

#[derive(Deserialize)]
struct ConfidenceHeadFile {
    #[serde(rename = "type")]
    _type: monostate::MustBe!("ConfidenceHeadConfig"),
    format: u32,
    trained_with: TrainedWith,
    target_width: u32,
    local: LocalSizes,
    attention: Option<ConfidenceHeadTreeConfig>,
    layer_norm_eps: f32,
}

#[derive(Deserialize)]
struct TrainedWith {
    speculator_sha256: String,
    tree: TrainedTree,
    sampling: ConfidenceHeadSampling,
}

#[derive(Deserialize)]
struct TrainedTree {
    children: u32,
    pool: u32,
    max_depth: u32,
}

#[derive(Deserialize)]
struct LocalSizes {
    hidden: u32,
    pool_hidden: u32,
    edge_hidden: u32,
    logit_cap: f32,
}

impl From<ConfidenceHeadFile> for ConfidenceHeadConfig {
    fn from(file: ConfidenceHeadFile) -> Self {
        Self {
            format: file.format,
            speculator_sha256: file.trained_with.speculator_sha256,
            sampling: file.trained_with.sampling,
            hidden: file.local.hidden,
            pool_hidden: file.local.pool_hidden,
            edge_hidden: file.local.edge_hidden,
            children: file.trained_with.tree.children,
            pool: file.trained_with.tree.pool,
            max_depth: file.trained_with.tree.max_depth,
            model_width: file.target_width,
            cap: file.local.logit_cap,
            layer_norm_eps: file.layer_norm_eps,
            tree: file.attention,
        }
    }
}

impl ConfidenceHeadConfig {
    /// Whether the head runs its tree attention on a tree of `slot_count` construction slots (one token per node).
    pub fn fits(
        &self,
        slot_count: u32,
    ) -> bool {
        self.tree.is_none() || slot_count * self.children < TOKENS_MAX
    }
}

/// sha256 of a file's bytes as 64 lowercase hex digits, read in chunks.
pub fn file_sha256(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

#[derive(Debug, Error)]
pub enum ConfidenceHeadError<B: Backend> {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("config error: {0}")]
    Config(#[from] serde_json::Error),
    #[error("parameter error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("unsupported confidence head: {0}")]
    Unsupported(String),
}

struct TreeLayer<B: Backend> {
    attention_norm: Normalization<B>,
    qkv: Box<dyn Linear<B>>,
    out: Box<dyn Linear<B>>,
    mlp_norm: Normalization<B>,
    up: Box<dyn Linear<B>>,
    down: Box<dyn Linear<B>>,
}

struct Tree<B: Backend> {
    config: ConfidenceHeadTreeConfig,
    node_in: Box<dyn Linear<B>>,
    child_in: Box<dyn Linear<B>>,
    layers: Box<[TreeLayer<B>]>,
    final_norm: Normalization<B>,
    couple_node: Box<dyn Linear<B>>,
    couple_child: Box<dyn Linear<B>>,
    depth_embedding: B::GlobalBuffer,
    relation_weights: B::GlobalBuffer,
    tree_tokens: <B::Kernels as Kernels>::ConfidenceHeadTreeTokensKernel,
    tree_attention: <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel,
}

pub struct ConfidenceHead<B: Backend> {
    config: ConfidenceHeadConfig,
    draft_norm_scales: B::GlobalBuffer,
    draft_norm_biases: B::GlobalBuffer,
    draft_weights: B::GlobalBuffer,
    draft_biases: B::GlobalBuffer,
    token_norm_scales: B::GlobalBuffer,
    token_norm_biases: B::GlobalBuffer,
    token_weights: B::GlobalBuffer,
    token_biases: B::GlobalBuffer,
    local_weights: B::GlobalBuffer,
    tree: Option<Tree<B>>,
    token_table: OnceLock<B::GlobalBuffer>,
    pool_noise: <B::Kernels as Kernels>::ConfidenceHeadPoolNoiseKernel,
    project_rows: <B::Kernels as Kernels>::ConfidenceHeadProjectRowsKernel,
    project_table: <B::Kernels as Kernels>::ConfidenceHeadProjectRowsKernel,
    round: <B::Kernels as Kernels>::ConfidenceHeadRoundKernel,
    prune: <B::Kernels as Kernels>::ConfidenceHeadPruneKernel,
    activation: <B::Kernels as Kernels>::ActivationKernel,
}

/// One step's state of the head: what every slot reads once per step and what the rounds keep for the prune, one row
/// per construction slot.
pub struct ConfidenceHeadStep<B: Backend> {
    /// [pool rows, pool] f32: the target sampler's Gumbel noise on every pool candidate.
    noise: B::ScratchBuffer,
    /// [pool rows, hidden] f32: the drafter-row term of z_u for a slot of each depth.
    draft_rows: B::ScratchBuffer,
    /// [slots, hidden] f32: z_u.
    nodes: B::ScratchBuffer,
    /// [slots * C, hidden + 4] f32: every child's token vector and standardised slot numbers.
    child_inputs: B::ScratchBuffer,
    /// [slots * C] u32: every child's index in its parent's pool.
    child_indices: B::ScratchBuffer,
    /// [slots * C] u32: whether the child is real.
    child_real: B::ScratchBuffer,
    /// [slots, pool] f32: l = a + rho of every slot.
    slot_logits: B::ScratchBuffer,
    /// [nodes per round, C] f32: the last round's children's edges, min(log q, 0), rows of valid nodes only (the
    /// frontier holds them too; the tests read them here).
    child_edges: B::ScratchBuffer,
    slot_count: u32,
    pool_rows: u32,
}

impl<B: Backend> ConfidenceHeadStep<B> {
    #[cfg(test)]
    pub fn child_edges(&self) -> &B::ScratchBuffer {
        &self.child_edges
    }
}

fn layer_norm_config(epsilon: f32) -> NormalizationConfig {
    NormalizationConfig {
        epsilon,
        scale_offset: None,
        upcast_mode: UpcastMode::FullLayer,
        subtract_mean: true,
        has_scale: true,
        has_biases: true,
    }
}

fn read<B: Backend>(
    tree: &ParameterTree<B>,
    name: &str,
    shape: &[u32],
) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
    tree.leaf(name)?.validate(shape, DataType::F32)?.read_buffer()
}

fn read_slice<B: Backend>(
    tree: &ParameterTree<B>,
    name: &str,
    shape: &[u32],
) -> Result<Box<[f32]>, ParameterLoaderError<B>> {
    tree.leaf(name)?.validate(shape, DataType::F32)?.read_slice::<f32>()
}

impl<B: Backend> ConfidenceHead<B> {
    /// Loads `<directory>/config.json` and `<directory>/model.safetensors` (every weight f32). The speculator reads the
    /// config first (`read_config`) and builds the head only if it applies (`with_config`).
    #[cfg(test)]
    pub fn new(
        directory: &Path,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let config = Self::read_config(directory)?;
        Self::with_config(config, directory, context)
    }

    /// `<directory>/config.json`, checked: what the loader can run, read without building the head.
    pub fn read_config(directory: &Path) -> Result<ConfidenceHeadConfig, ConfidenceHeadError<B>> {
        let config: ConfidenceHeadConfig =
            serde_json::from_reader(BufReader::new(File::open(directory.join(CONFIG_FILE))?))?;
        Self::check(&config)?;
        Ok(config)
    }

    /// The head of `config` (from `read_config`) with the weights of `<directory>/model.safetensors`.
    pub fn with_config(
        config: ConfidenceHeadConfig,
        directory: &Path,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let weights_file = File::open(directory.join(WEIGHTS_FILE))?;
        let loader = ParameterLoader::new(&weights_file, context)?;
        let tree = loader.tree();
        let (hidden, pool_hidden, edge_hidden) = (config.hidden, config.pool_hidden, config.edge_hidden);
        let width = config.model_width;
        let edge_input = LocalLayout::edge_input(hidden as usize) as u32;

        let local = [
            read_slice(&tree, "pool_set.entry.weights.weights", &[pool_hidden, POOL_INPUTS as u32])?,
            read_slice(&tree, "pool_set.entry.biases", &[pool_hidden])?,
            read_slice(&tree, "pool_set.hidden.weights.weights", &[pool_hidden, pool_hidden])?,
            read_slice(&tree, "pool_set.hidden.biases", &[pool_hidden])?,
            read_slice(&tree, "pool_set.out.weights.weights", &[hidden, 2 * pool_hidden])?,
            read_slice(&tree, "pool_set.out.biases", &[hidden])?,
            read_slice(&tree, "slot.means", &[SLOT_NUMBERS as u32])?,
            read_slice(&tree, "slot.stds", &[SLOT_NUMBERS as u32])?,
            read_slice(&tree, "slot.keep", &[SLOT_NUMBERS as u32])?,
            read_slice(&tree, "edge.input_norm.scales", &[edge_input])?,
            read_slice(&tree, "edge.input_norm.biases", &[edge_input])?,
            read_slice(&tree, "edge.entry.weights.weights", &[hidden, edge_input])?,
            read_slice(&tree, "edge.entry.biases", &[hidden])?,
            read_slice(&tree, "edge.block_norm.scales", &[hidden])?,
            read_slice(&tree, "edge.block_norm.biases", &[hidden])?,
            read_slice(&tree, "edge.up.weights.weights", &[edge_hidden, hidden])?,
            read_slice(&tree, "edge.up.biases", &[edge_hidden])?,
            read_slice(&tree, "edge.down.weights.weights", &[hidden, edge_hidden])?,
            read_slice(&tree, "edge.down.biases", &[hidden])?,
            read_slice(&tree, "edge.final_norm.scales", &[hidden])?,
            read_slice(&tree, "edge.final_norm.biases", &[hidden])?,
            read_slice(&tree, "edge.output.weights.weights", &[1, hidden])?,
            read_slice(&tree, "edge.output.biases", &[1])?,
            read_slice(&tree, "output.weights.weights", &[NODE_OUTPUTS as u32, hidden])?,
            read_slice(&tree, "output.biases", &[NODE_OUTPUTS as u32])?,
        ]
        .concat();
        let layout = LocalLayout::new(hidden as usize, pool_hidden as usize, edge_hidden as usize);
        assert_eq!(local.len(), layout.total, "the packed local weights follow LocalLayout");
        let mut local_weights =
            context.create_buffer(local.len() * size_of::<f32>()).map_err(ConfidenceHeadError::Backend)?;
        local_weights.copyin(&local);

        let tree_part =
            config.tree.clone().map(|tree_config| Self::load_tree(&config, tree_config, &tree, context)).transpose()?;
        let head = Self {
            draft_norm_scales: read(&tree, "draft_norm.scales", &[width])?,
            draft_norm_biases: read(&tree, "draft_norm.biases", &[width])?,
            draft_weights: read(&tree, "draft.weights.weights", &[hidden, width])?,
            draft_biases: read(&tree, "draft.biases", &[hidden])?,
            token_norm_scales: read(&tree, "token_norm.scales", &[width])?,
            token_norm_biases: read(&tree, "token_norm.biases", &[width])?,
            token_weights: read(&tree, "token_projection.weights.weights", &[hidden, width])?,
            token_biases: read(&tree, "token_projection.biases", &[hidden])?,
            local_weights,
            tree: tree_part,
            token_table: OnceLock::new(),
            pool_noise: <B::Kernels as Kernels>::ConfidenceHeadPoolNoiseKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            // The kernels are specialised to the head's dimensions (pipeline constants), so their loops have fixed
            // bounds and their weight offsets are constants.
            project_rows: <B::Kernels as Kernels>::ConfidenceHeadProjectRowsKernel::new(
                context,
                DataType::BF16,
                false,
                width,
                hidden,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            project_table: <B::Kernels as Kernels>::ConfidenceHeadProjectRowsKernel::new(
                context,
                DataType::BF16,
                true,
                width,
                hidden,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            round: <B::Kernels as Kernels>::ConfidenceHeadRoundKernel::new(
                context,
                config.pool,
                config.children,
                hidden,
                pool_hidden,
                edge_hidden,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            prune: <B::Kernels as Kernels>::ConfidenceHeadPruneKernel::new(
                context,
                config.pool,
                config.children,
                hidden,
                pool_hidden,
                edge_hidden,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            activation: <B::Kernels as Kernels>::ActivationKernel::new(context, DataType::F32, true)
                .map_err(ConfidenceHeadError::Backend)?,
            config,
        };
        tree.assert_all_tensors_validated()?;
        Ok(head)
    }

    fn check(config: &ConfidenceHeadConfig) -> Result<(), ConfidenceHeadError<B>> {
        let mut problems = Vec::new();
        if config.format != FORMAT {
            problems.push(format!("format {} (expected {FORMAT})", config.format));
        }
        if config.hidden == 0 || config.hidden > HIDDEN_MAX {
            problems.push(format!("hidden {} outside 1..={HIDDEN_MAX}", config.hidden));
        }
        if config.children == 0 || config.children > CHILDREN_MAX || config.children * config.hidden > 256 {
            problems.push(format!("{} children of width {}", config.children, config.hidden));
        }
        if config.pool == 0 || config.pool > CANDIDATES_MAX {
            problems.push(format!("pool {} outside 1..={CANDIDATES_MAX}", config.pool));
        }
        if config.pool_hidden == 0 || config.pool_hidden > POOL_HIDDEN_MAX {
            problems.push(format!("pool width {} outside 1..={POOL_HIDDEN_MAX}", config.pool_hidden));
        }
        if config.edge_hidden == 0 || config.edge_hidden > EDGE_HIDDEN_MAX {
            problems.push(format!("edge width {} outside 1..={EDGE_HIDDEN_MAX}", config.edge_hidden));
        }
        // The Metal dense layers read four adjacent inputs at once: every layer's input width is a multiple of four
        // (hidden, 2 * pool width, 3 * hidden + 4, edge width).
        if !config.hidden.is_multiple_of(4)
            || !config.pool_hidden.is_multiple_of(2)
            || !config.edge_hidden.is_multiple_of(4)
        {
            problems.push(format!(
                "widths hidden {}, pool {}, edge {} not multiples of 4, 2, 4",
                config.hidden, config.pool_hidden, config.edge_hidden
            ));
        }
        if config.max_depth + 1 > LEVELS_MAX {
            problems.push(format!("max depth {}", config.max_depth));
        }
        if config.speculator_sha256.len() != 64
            || !config.speculator_sha256.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            problems.push(format!("speculator sha256 {:?} is not 64 lowercase hex digits", config.speculator_sha256));
        }
        if let Some(tree) = &config.tree {
            if tree.relations != RELATIONS {
                problems.push(format!("{} tree relations (expected {RELATIONS})", tree.relations));
            }
            if tree.heads == 0
                || tree.width % tree.heads != 0
                || 256 % (tree.width / tree.heads) != 0
                || tree.width / tree.heads > HEAD_DIM_MAX
                || (tree.width / tree.heads) % 8 != 0
            {
                problems.push(format!("{} heads over width {}", tree.heads, tree.width));
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(ConfidenceHeadError::Unsupported(problems.join("; ")))
        }
    }

    fn load_tree(
        config: &ConfidenceHeadConfig,
        tree_config: ConfidenceHeadTreeConfig,
        parameters: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Tree<B>, ConfidenceHeadError<B>> {
        let tree = parameters.subtree("tree");
        let width = tree_config.width;
        let norm = layer_norm_config(config.layer_norm_eps);
        let linear = |name: &str, inputs: u32, outputs: u32, parameters: &ParameterTree<B>| {
            <dyn Linear<B>>::new(inputs, outputs, DataType::F32, true, &parameters.subtree(name), context)
        };
        let normalization = |name: &str, mode: ShortcutMode, parameters: &ParameterTree<B>| {
            Normalization::new(
                width,
                DataType::F32,
                None,
                mode,
                PostLayerScalar::None,
                &norm,
                &parameters.subtree(name),
                context,
            )
        };
        let layers = (0..tree_config.layers)
            .map(|index| {
                let layer = tree.subtree("layers").subtree(&index.to_string());
                Ok(TreeLayer {
                    // The residual stream lives in the norms' shortcut: the first layer copies its input there, later
                    // norms add their input to it first.
                    attention_norm: normalization(
                        "attention_norm",
                        if index == 0 {
                            ShortcutMode::Copy
                        } else {
                            ShortcutMode::Add
                        },
                        &layer,
                    )?,
                    qkv: linear("qkv", width, 3 * width, &layer)?,
                    out: linear("out", width, width, &layer)?,
                    mlp_norm: normalization("mlp_norm", ShortcutMode::Add, &layer)?,
                    up: linear("up", width, tree_config.mlp_hidden, &layer)?,
                    down: linear("down", tree_config.mlp_hidden, width, &layer)?,
                })
            })
            .collect::<Result<Box<[_]>, ConfidenceHeadError<B>>>()?;
        Ok(Tree {
            node_in: linear("node_in", config.hidden, width, &tree)?,
            child_in: linear("child_in", config.hidden + SLOT_NUMBERS as u32, width, &tree)?,
            layers,
            final_norm: normalization("final_norm", ShortcutMode::Add, &tree)?,
            couple_node: linear("couple_node", width, config.hidden, &tree)?,
            couple_child: linear("couple_child", width, config.hidden, &tree)?,
            depth_embedding: read(&tree, "depth_embedding", &[config.max_depth + 1, width])?,
            relation_weights: read(&tree, "relation_weight", &[tree_config.heads, tree_config.relations])?,
            tree_tokens: <B::Kernels as Kernels>::ConfidenceHeadTreeTokensKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            tree_attention: <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel::new(
                context,
                tree_config.heads,
                width / tree_config.heads,
                tree_config.relations,
                config.max_depth,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            config: tree_config,
        })
    }

    pub fn config(&self) -> &ConfidenceHeadConfig {
        &self.config
    }

    /// Whether the head applies under a sampler. It predicts the target's sampled choice, so it does not under greedy
    /// decoding or a sampler that is greedy in effect (top-k 1, temperature 0).
    pub fn applies_to(sampling_method: &SamplingMethod) -> bool {
        match sampling_method {
            SamplingMethod::Greedy => false,
            SamplingMethod::Stochastic {
                temperature,
                top_k,
                ..
            } => *top_k != Some(1) && *temperature != Some(0.0),
        }
    }

    /// The projected token table, W_e LN(E_t) of every token of the target's input embedding in bf16, built on the
    /// first call (one pass over the vocabulary) and kept.
    pub fn token_table(
        &self,
        embedding: &Embedding<B>,
        context: &B::Context,
    ) -> Result<&B::GlobalBuffer, ConfidenceHeadError<B>> {
        if let Some(table) = self.token_table.get() {
            return Ok(table);
        }
        if embedding.model_dim() != self.config.model_width {
            return Err(ConfidenceHeadError::Unsupported(format!(
                "the target embedding is {} wide, the head reads {}",
                embedding.model_dim(),
                self.config.model_width
            )));
        }
        let vocab = embedding.vocab_size();
        let hidden = self.config.hidden;
        let mut table = context
            .create_buffer(vocab as usize * hidden as usize * DataType::BF16.size_in_bytes())
            .map_err(ConfidenceHeadError::Backend)?;
        let mut command_buffer = context
            .create_command_buffer(Some("confidence head token table"), None)
            .map_err(ConfidenceHeadError::Backend)?;
        let row_bytes = hidden as usize * DataType::BF16.size_in_bytes();
        for start in (0..vocab).step_by(TABLE_CHUNK as usize) {
            let count = TABLE_CHUNK.min(vocab - start);
            let ids = (start..start + count).collect::<Box<[u32]>>();
            let ids = command_buffer.allocate_constant_from_slice(&ids).map_err(ConfidenceHeadError::Backend)?;
            let rows = embedding.encode_lookup(&ids, count, &mut command_buffer)?;
            self.project_table.encode(
                &rows,
                &self.token_norm_scales,
                &self.token_norm_biases,
                &self.token_weights,
                &self.token_biases,
                None::<&mut B::ScratchBuffer>,
                Some((&mut table).subrange_mut(start as usize * row_bytes..(start + count) as usize * row_bytes)),
                count,
                self.config.layer_norm_eps,
                &mut command_buffer,
            );
        }
        command_buffer.end_encoding().submit().wait_until_completed().map_err(ConfidenceHeadError::Backend)?;
        Ok(self.token_table.get_or_init(|| table))
    }

    /// Allocates the step's state and computes what every slot reads once per step: the pool rows' noise and the
    /// drafter rows. `draft_hidden` is the drafter's [depth + 1, model width] output (row d + 1 is behind pool row d),
    /// `candidate_ids` the [pool rows, pool] candidate pool, `depth_seeds` the target sampler's seed of every depth.
    pub fn encode_step(
        &self,
        draft_hidden: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        depth_seeds: impl BufferRef<Backend = B>,
        pool_rows: u32,
        slot_count: u32,
        nodes_per_round: u32,
        vocab_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<ConfidenceHeadStep<B>, B::Error> {
        let ConfidenceHeadConfig {
            hidden,
            pool,
            children,
            model_width,
            ..
        } = self.config;
        // Every buffer but one is written before it is read: the rounds fill the rows of the slots they expand, and
        // only those rows are read. child_real must say "no child" for slots never expanded.
        let mut child_real = command_buffer.allocate_scratch_for_shape(&[slot_count * children], DataType::U32)?;
        command_buffer.encode_fill(&mut child_real, 0);
        let mut scratch =
            |shape: &[u32], data_type: DataType| command_buffer.allocate_scratch_for_shape(shape, data_type);
        let mut step = ConfidenceHeadStep {
            noise: scratch(&[pool_rows, pool], DataType::F32)?,
            draft_rows: scratch(&[pool_rows, hidden], DataType::F32)?,
            nodes: scratch(&[slot_count, hidden], DataType::F32)?,
            child_inputs: scratch(&[slot_count * children, hidden + SLOT_NUMBERS as u32], DataType::F32)?,
            child_indices: scratch(&[slot_count * children], DataType::U32)?,
            child_real,
            slot_logits: scratch(&[slot_count, pool], DataType::F32)?,
            child_edges: scratch(&[nodes_per_round, children], DataType::F32)?,
            slot_count,
            pool_rows,
        };
        self.pool_noise.encode(
            candidate_ids,
            depth_seeds,
            &mut step.noise,
            pool_rows * pool,
            pool,
            vocab_size,
            command_buffer,
        );
        let row_bytes = model_width as usize * DataType::BF16.size_in_bytes();
        self.project_rows.encode(
            draft_hidden.subrange(row_bytes..),
            &self.draft_norm_scales,
            &self.draft_norm_biases,
            &self.draft_weights,
            &self.draft_biases,
            Some(&mut step.draft_rows),
            None::<&mut B::ScratchBuffer>,
            pool_rows,
            self.config.layer_norm_eps,
            command_buffer,
        );
        Ok(step)
    }

    /// The local part on the `rows` nodes a round has just expanded, after Weaver's top_children put their children
    /// into the frontier: the children's edges min(log q, 0) into `step.child_edges()` and into their frontier entries
    /// (edge, path = the parent's path in the packed tree + edge, and its priority key), and the nodes' rows of the
    /// step state. The tree's expand width must be the head's children.
    pub fn encode_round(
        &self,
        token_table: &B::GlobalBuffer,
        residual_logits: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        node_metadata: impl BufferRef<Backend = B>,
        node_valid: impl BufferRef<Backend = B>,
        packed_tree: impl BufferRef<Backend = B>,
        rows: u32,
        frontier_capacity: u32,
        tree_slot_count: u32,
        frontier: impl BufferMut<Backend = B>,
        step: &mut ConfidenceHeadStep<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.round.encode(
            residual_logits,
            candidate_logits,
            candidate_ids,
            &step.noise,
            &step.draft_rows,
            node_metadata,
            node_valid,
            packed_tree,
            token_table,
            &self.local_weights,
            &mut step.child_edges,
            frontier,
            &mut step.nodes,
            &mut step.child_inputs,
            &mut step.child_indices,
            &mut step.child_real,
            &mut step.slot_logits,
            rows,
            step.pool_rows,
            self.config.cap,
            self.config.layer_norm_eps,
            frontier_capacity,
            tree_slot_count,
            command_buffer,
        );
    }

    /// The full head on the grown tree: attention over all its nodes, then every real child's edge rewritten in the
    /// lane the final prune sums along paths (the packed tree's for a child expanded into a slot, the frontier's
    /// otherwise). A head without tree attention prunes by its local edges, which the rounds already wrote.
    pub fn encode_prune(
        &self,
        step: &ConfidenceHeadStep<B>,
        mut packed_tree: impl BufferMut<Backend = B>,
        mut frontier: impl BufferMut<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let Some(tree) = &self.tree else {
            return Ok(());
        };
        let command_buffer = &mut *command_buffer.debug_group("confidence head prune");
        let ConfidenceHeadConfig {
            children,
            max_depth,
            ..
        } = self.config;
        let slots = step.slot_count;
        let token_count = 1 + slots * children;
        let width = tree.config.width;
        let copy = |source: &B::ScratchBuffer,
                    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
         -> Result<B::ScratchBuffer, B::Error> {
            let mut target = command_buffer.allocate_scratch(source.size())?;
            command_buffer.encode_copy(source, &mut target);
            Ok(target)
        };
        let node_inputs = tree.node_in.encode(copy(&step.nodes, command_buffer)?, slots, command_buffer)?;
        let child_inputs =
            tree.child_in.encode(copy(&step.child_inputs, command_buffer)?, slots * children, command_buffer)?;
        let mut tokens = command_buffer.allocate_scratch_for_shape(&[token_count, width], DataType::F32)?;
        let mut token_parents = command_buffer.allocate_scratch_for_shape(&[token_count], DataType::U32)?;
        let mut token_depths = command_buffer.allocate_scratch_for_shape(&[token_count], DataType::U32)?;
        let mut token_present = command_buffer.allocate_scratch_for_shape(&[token_count], DataType::U32)?;
        let mut token_slots = command_buffer.allocate_scratch_for_shape(&[token_count], DataType::U32)?;
        let mut token_ancestors =
            command_buffer.allocate_scratch_for_shape(&[token_count, max_depth + 1], DataType::U32)?;
        let mut slot_tokens = command_buffer.allocate_scratch_for_shape(&[slots], DataType::U32)?;
        command_buffer.encode_fill(&mut slot_tokens, 0);
        tree.tree_tokens.encode(
            packed_tree.as_ref(),
            frontier.as_ref(),
            &step.child_real,
            &node_inputs,
            &child_inputs,
            &tree.depth_embedding,
            &mut tokens,
            &mut token_parents,
            &mut token_depths,
            &mut token_present,
            &mut token_slots,
            &mut token_ancestors,
            &mut slot_tokens,
            slots,
            children,
            width,
            max_depth,
            command_buffer,
        );

        let mut residual = command_buffer.allocate_scratch_for_shape(&[token_count, width], DataType::F32)?;
        let mut layer_input = tokens;
        for layer in &tree.layers {
            let attention_input =
                layer.attention_norm.encode(&layer_input, 0, token_count, Some(&mut residual), command_buffer)?;
            let qkv = layer.qkv.encode(attention_input, token_count, command_buffer)?;
            let mut attention = command_buffer.allocate_scratch_for_shape(&[token_count, width], DataType::F32)?;
            tree.tree_attention.encode(
                &qkv,
                &tree.relation_weights,
                &token_parents,
                &token_depths,
                &token_present,
                &token_ancestors,
                &mut attention,
                token_count,
                command_buffer,
            );
            let projected = layer.out.encode(attention, token_count, command_buffer)?;
            let mlp_input = layer.mlp_norm.encode(&projected, 0, token_count, Some(&mut residual), command_buffer)?;
            let mut raised = layer.up.encode(mlp_input, token_count, command_buffer)?;
            self.activation.encode(
                None::<&B::ScratchBuffer>,
                &mut raised,
                token_count * tree.config.mlp_hidden,
                ActivationType::SILU,
                command_buffer,
            );
            layer_input = layer.down.encode(raised, token_count, command_buffer)?;
        }
        let normed = tree.final_norm.encode(&layer_input, 0, token_count, Some(&mut residual), command_buffer)?;
        let normed_copy = copy(&normed, command_buffer)?;
        let node_shifts = tree.couple_node.encode(normed, token_count, command_buffer)?;
        let child_shifts = tree.couple_child.encode(normed_copy, token_count, command_buffer)?;
        self.prune.encode(
            &step.nodes,
            &step.child_inputs,
            &step.child_indices,
            &step.child_real,
            &step.slot_logits,
            &step.noise,
            &node_shifts,
            &child_shifts,
            &slot_tokens,
            &token_slots,
            &self.local_weights,
            packed_tree.reborrow(),
            frontier.reborrow(),
            slots,
            step.pool_rows,
            self.config.cap,
            self.config.layer_norm_eps,
            command_buffer,
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/confidence_head_config_test.rs"]
mod config_tests;

#[cfg(test)]
#[path = "../../../unit/encodable_block/confidence_head_test.rs"]
pub(crate) mod tests;
