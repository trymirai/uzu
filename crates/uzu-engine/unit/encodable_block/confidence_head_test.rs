//! The confidence head against a float64 reference written from its definition (this module's doc): a small head with
//! pseudo-random weights and one step of a Weaver tree grown here, every round's local edges and the prune's final
//! edges, on the CPU and on every other backend. The test keeps no data of its own: the head is written to a temporary
//! directory and loaded as the speculator loads it.
//!
//! The reference is independent of the code under test: plain loops over the definitions in float64, sharing with the
//! engine only the bf16 and f32 inputs both read.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::ConfidenceHead;
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
            CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::{
                ConfidenceHeadPoolNoiseKernel, ConfidenceHeadProjectRowsKernel, ConfidenceHeadTreeAttentionKernel,
                ConfidenceHeadTreeTokensKernel,
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::sampling::{gumbel_float, revidx},
    tests::helpers::{
        buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend,
        submit_command_buffer,
    },
};

/// The engine's f32 against the reference's f64: per edge a few hundred-term dot products, layer norms, a softmax over
/// the pool and (for the prune) two attention layers, so f32 rounding stays near 1e-5 of values of order 10.
const TOLERANCE: f32 = 2e-4;
/// The rounds' edges on a GPU backend: the round kernels are the engine's own f32 kernels (no matmul), and Metal's fast
/// exp and log stay within a few f32 units, as the CPU's do (observed: 2e-7), so the CPU's tolerance holds.
const GPU_ROUND_TOLERANCE: f32 = TOLERANCE;
/// The full head's edges on a GPU backend: its dense layers run on the engine's matmuls, which on Apple GPUs round f32
/// operands to a 10-bit mantissa (confidence_head_blocks_against_the_cpu: median 7.8e-4 relative per layer). The final
/// edges pass through 17 of them (four attention layers of four, the coupling layer), hence 2^-10 x 8.
const GPU_FINAL_TOLERANCE: f32 = 7.8e-3;
/// Pool log-probabilities are clamped at this floor, as in training.
const LOG_PROB_FLOOR: f64 = -30.0;
const LAYER_NORM_EPS: f64 = 1e-5;
const LOGIT_CAP: f64 = 50.0;
/// Tree relations: self, parent, child, ancestor two or more edges up, descendant likewise, sibling.
const RELATIONS: usize = 6;
const SLOT_NUMBERS: usize = 4;
const NO_PARENT: u32 = u32::MAX;

/// Sizes of a head and of the step it runs on. The first round expands the root alone, every later one
/// `round_width` slots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    children: usize,
    pool: usize,
    pool_rows: usize,
    max_depth: usize,
    model: usize,
    vocab: usize,
    hidden: usize,
    pool_hidden: usize,
    edge_hidden: usize,
    layers: usize,
    heads: usize,
    width: usize,
    mlp_hidden: usize,
    rounds: usize,
    round_width: usize,
}

impl Shape {
    fn slots(&self) -> usize {
        1 + (self.rounds - 1) * self.round_width
    }
}

/// The tested head: three children, three pool rows below a depth cap of four (slots at depths 3 and 4 share the last
/// row), ten rounds (19 slots).
pub(crate) const SMALL: Shape = Shape {
    children: 3,
    pool: 64,
    pool_rows: 3,
    max_depth: 4,
    model: 48,
    vocab: 300,
    hidden: 16,
    pool_hidden: 4,
    edge_hidden: 32,
    layers: 2,
    heads: 2,
    width: 32,
    mlp_hidden: 64,
    rounds: 10,
    round_width: 2,
};

/// A deployed head's sizes (the timing test): 16 rounds of 4 slots over pools of 512, Qwen3.8's width.
const DEPLOYED: Shape = Shape {
    children: 4,
    pool: 512,
    pool_rows: 15,
    max_depth: 15,
    model: 5120,
    vocab: 4096,
    hidden: 64,
    pool_hidden: 16,
    edge_hidden: 128,
    layers: 4,
    heads: 4,
    width: 256,
    mlp_hidden: 512,
    rounds: 16,
    round_width: 4,
};

/// A deployed head's sizes for the reference check: the kernels are specialised to these, only the vocabulary is
/// smaller (it sizes the token table alone).
const DEPLOYED_CHECK: Shape = Shape {
    vocab: 1024,
    ..DEPLOYED
};

/// SplitMix64: deterministic pseudo-random numbers for the test's weights and step.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform in [-1, 1) on a grid of 2^-23: every value is an f32.
    fn uniform(&mut self) -> f64 {
        (self.next() >> 40) as f64 / (1u64 << 23) as f64 - 1.0
    }

    /// Uniform in (0, 1).
    fn open_unit(&mut self) -> f64 {
        ((self.next() >> 40) as f64 + 0.5) / (1u64 << 24) as f64
    }
}

fn to_f32(value: f64) -> f64 {
    f64::from(value as f32)
}

/// A head's weights in f64 (each an f32 value), by the names of uzu's head directory.
pub(crate) struct Weights {
    shape: Shape,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f64>)>,
    dense: Vec<String>,
}

impl Weights {
    pub(crate) fn random(
        shape: Shape,
        seed: u64,
    ) -> Self {
        let mut weights = Self {
            shape,
            tensors: BTreeMap::new(),
            dense: Vec::new(),
        };
        let mut rng = Rng(seed);
        let (hidden, model, width) = (shape.hidden, shape.model, shape.width);
        let edge_input = 3 * hidden + SLOT_NUMBERS;
        weights.add_norm(&mut rng, "draft_norm", model);
        weights.add_dense(&mut rng, "draft", hidden, model);
        weights.add_norm(&mut rng, "token_norm", model);
        weights.add_dense(&mut rng, "token_projection", hidden, model);
        weights.add_dense(&mut rng, "pool_set.entry", shape.pool_hidden, 3);
        weights.add_dense(&mut rng, "pool_set.hidden", shape.pool_hidden, shape.pool_hidden);
        weights.add_dense(&mut rng, "pool_set.out", hidden, 2 * shape.pool_hidden);
        weights.add("slot.means", vec![SLOT_NUMBERS], (0..SLOT_NUMBERS).map(|_| 0.3 * rng.uniform()).collect());
        weights.add(
            "slot.stds",
            vec![SLOT_NUMBERS],
            (0..SLOT_NUMBERS).map(|_| 1.0 + 0.5 * rng.uniform().abs()).collect(),
        );
        weights.add("slot.keep", vec![SLOT_NUMBERS], vec![1.0; SLOT_NUMBERS]);
        weights.add_norm(&mut rng, "edge.input_norm", edge_input);
        weights.add_dense(&mut rng, "edge.entry", hidden, edge_input);
        weights.add_norm(&mut rng, "edge.block_norm", hidden);
        weights.add_dense(&mut rng, "edge.up", shape.edge_hidden, hidden);
        weights.add_dense(&mut rng, "edge.down", hidden, shape.edge_hidden);
        weights.add_norm(&mut rng, "edge.final_norm", hidden);
        weights.add_dense(&mut rng, "edge.output", 1, hidden);
        // The node output layer's third row is unused (its readers read alpha and beta); the export writes zeros.
        weights.add_dense(&mut rng, "output", 3, hidden);
        let (_, output) = weights.tensors.get_mut("output.weights.weights").unwrap();
        output[2 * hidden..].fill(0.0);
        weights.tensors.get_mut("output.biases").unwrap().1[2] = 0.0;
        // The tree path is made strong (large relation biases, inputs and couplings): every part of the attention then
        // moves the final edges well above a GPU's matmul rounding, so a defect in any of them shows on every backend.
        weights.add_dense_scaled(&mut rng, "tree.node_in", width, hidden, 2.0, 1.0);
        weights.add_dense_scaled(&mut rng, "tree.child_in", width, hidden + SLOT_NUMBERS, 2.0, 1.0);
        weights.add_dense_scaled(&mut rng, "tree.couple_node", hidden, width, 3.0, 0.1);
        weights.add_dense_scaled(&mut rng, "tree.couple_child", hidden, width, 3.0, 0.1);
        let depths = shape.max_depth + 1;
        weights.add(
            "tree.depth_embedding",
            vec![depths, width],
            (0..depths * width).map(|_| to_f32(rng.uniform())).collect(),
        );
        weights.add(
            "tree.relation_weight",
            vec![shape.heads, RELATIONS],
            (0..shape.heads * RELATIONS).map(|_| to_f32(3.0 * rng.uniform())).collect(),
        );
        for layer in 0..shape.layers {
            let prefix = format!("tree.layers.{layer}");
            weights.add_norm(&mut rng, &format!("{prefix}.attention_norm"), width);
            weights.add_dense_scaled(&mut rng, &format!("{prefix}.qkv"), 3 * width, width, 2.0, 0.1);
            weights.add_dense(&mut rng, &format!("{prefix}.out"), width, width);
            weights.add_norm(&mut rng, &format!("{prefix}.mlp_norm"), width);
            weights.add_dense(&mut rng, &format!("{prefix}.up"), shape.mlp_hidden, width);
            weights.add_dense(&mut rng, &format!("{prefix}.down"), width, shape.mlp_hidden);
        }
        weights.add_norm(&mut rng, "tree.final_norm", width);
        weights
    }

    fn add(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        self.tensors.insert(name.to_string(), (shape, values));
    }

    fn add_dense(
        &mut self,
        rng: &mut Rng,
        name: &str,
        outputs: usize,
        inputs: usize,
    ) {
        self.add_dense_scaled(rng, name, outputs, inputs, 1.0, 0.1);
    }

    /// A dense layer with weights uniform in gain * [-1, 1) / sqrt(inputs) and biases in bias * [-1, 1).
    fn add_dense_scaled(
        &mut self,
        rng: &mut Rng,
        name: &str,
        outputs: usize,
        inputs: usize,
        gain: f64,
        bias: f64,
    ) {
        let scale = gain / (inputs as f64).sqrt();
        self.add(
            &format!("{name}.weights.weights"),
            vec![outputs, inputs],
            (0..outputs * inputs).map(|_| to_f32(scale * rng.uniform())).collect(),
        );
        self.add(
            &format!("{name}.biases"),
            vec![outputs],
            (0..outputs).map(|_| to_f32(bias * rng.uniform())).collect(),
        );
        self.dense.push(name.to_string());
    }

    fn add_norm(
        &mut self,
        rng: &mut Rng,
        name: &str,
        size: usize,
    ) {
        self.add(&format!("{name}.scales"), vec![size], (0..size).map(|_| to_f32(1.0 + 0.2 * rng.uniform())).collect());
        self.add(&format!("{name}.biases"), vec![size], (0..size).map(|_| to_f32(0.1 * rng.uniform())).collect());
    }

    fn get(
        &self,
        name: &str,
    ) -> &[f64] {
        &self.tensors.get(name).unwrap_or_else(|| panic!("no tensor {name}")).1
    }

    /// W x + b of a dense layer, W [outputs, inputs] row-major.
    fn dense(
        &self,
        name: &str,
        input: &[f64],
    ) -> Vec<f64> {
        let weights = self.get(&format!("{name}.weights.weights"));
        let biases = self.get(&format!("{name}.biases"));
        assert_eq!(weights.len(), biases.len() * input.len(), "{name}");
        biases
            .iter()
            .enumerate()
            .map(|(output, bias)| {
                bias + weights[output * input.len()..(output + 1) * input.len()]
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Layer norm: (x - mean) / sqrt(variance + eps) * scale + bias.
    fn norm(
        &self,
        name: &str,
        input: &[f64],
    ) -> Vec<f64> {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance = input.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / input.len() as f64;
        let scale = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
        input
            .iter()
            .zip(self.get(&format!("{name}.scales")).iter().zip(self.get(&format!("{name}.biases"))))
            .map(|(value, (gain, bias))| (value - mean) * scale * gain + bias)
            .collect()
    }

    /// The head as its directory: config.json (format 2) and model.safetensors (f32, a full-precision spec for every
    /// dense layer).
    pub(crate) fn write(
        &self,
        directory: &Path,
    ) {
        let shape = self.shape;
        std::fs::create_dir_all(directory).unwrap();
        let config = serde_json::json!({
            "type": "ConfidenceHeadConfig",
            "format": 2,
            "trained_with": {
                "speculator_sha256": "0".repeat(64),
                "tree": {"children": shape.children, "pool": shape.pool, "max_depth": shape.max_depth},
                "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": null}
            },
            "target_width": shape.model,
            "local": {
                "hidden": shape.hidden,
                "pool_hidden": shape.pool_hidden,
                "edge_hidden": shape.edge_hidden,
                "logit_cap": LOGIT_CAP
            },
            "attention": {
                "layers": shape.layers,
                "heads": shape.heads,
                "width": shape.width,
                "mlp_hidden": shape.mlp_hidden,
                "relations": RELATIONS
            },
            "layer_norm_eps": LAYER_NORM_EPS
        });
        std::fs::write(directory.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        let mut header = serde_json::Map::new();
        let spec = r#"{"type":"FullPrecisionSpec","layout":"output_input"}"#;
        header.insert(
            "__metadata__".to_string(),
            serde_json::Value::Object(
                self.dense.iter().map(|name| (format!("{name}.weights.spec"), serde_json::json!(spec))).collect(),
            ),
        );
        let mut data = Vec::new();
        for (name, (dims, values)) in &self.tensors {
            let start = data.len();
            data.extend(values.iter().flat_map(|&value| (value as f32).to_le_bytes()));
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": dims, "data_offsets": [start, data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(directory.join("model.safetensors"), file).unwrap();
    }
}

/// A head written to a temporary directory, removed when dropped.
struct HeadDirectory(PathBuf);

impl HeadDirectory {
    fn new(weights: &Weights) -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "confidence_head_test_{}_{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        weights.write(&directory);
        Self(directory)
    }

    fn load<B: Backend>(
        &self,
        context: &B::Context,
    ) -> ConfidenceHead<B> {
        ConfidenceHead::<B>::new(&self.0, context).unwrap()
    }
}

impl Drop for HeadDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One step of a Weaver tree as the head reads it: the drafter's pools and rows, every slot's Weaver residual and the
/// tree in uzu's layouts (packed tree [TreeIdx, slot], frontier [FrontierIdx, slot * children + child]).
struct Step {
    shape: Shape,
    /// (first slot, slots) of every round.
    rounds: Vec<(usize, usize)>,
    depths: Vec<u32>,
    expanded: Vec<bool>,
    packed_tree: Vec<u32>,
    frontier: Vec<u32>,
    /// [pool rows, pool]: the drafter's candidates, their logits a and the target sampler's Gumbel noise G.
    candidate_ids: Vec<u32>,
    candidate_logits: Vec<f32>,
    noise: Vec<f32>,
    /// [slots, pool]: rho of every slot (l = a + rho), zero where not expanded.
    residuals: Vec<bf16>,
    /// [pool rows + 1, model]: drafter row d + 1 behind pool row d (row 0 unused).
    draft_hidden: Vec<bf16>,
    /// [vocab, model]: the target's input embedding.
    embedding: Vec<bf16>,
}

impl Step {
    /// A tree grown round by round: every slot takes a frontier entry of a slot of an earlier round, the deepest one
    /// for the first slot of a pair and the shallowest for the second, so rounds mix depths and some slots reach the
    /// depth cap (expanded into, no children of their own). An expanded slot's children are its pool's top candidates
    /// by l.
    fn grow(
        shape: Shape,
        seed: u64,
    ) -> Self {
        let mut rng = Rng(seed);
        let (slots, children, pool, rows) = (shape.slots(), shape.children, shape.pool, shape.pool_rows);
        let candidate_ids =
            (0..rows * pool).map(|index| ((index / pool * 131 + index % pool * 7 + 3) % shape.vocab) as u32).collect();
        let candidate_logits = (0..rows * pool).map(|_| (3.0 * rng.uniform()) as f32).collect::<Vec<_>>();
        let noise = (0..rows * pool).map(|_| (-(-rng.open_unit().ln()).ln()) as f32).collect();
        let draft_hidden = (0..(rows + 1) * shape.model).map(|_| bf16::from_f64(rng.uniform())).collect();
        let embedding = (0..shape.vocab * shape.model).map(|_| bf16::from_f64(rng.uniform())).collect();
        let mut step = Self {
            shape,
            rounds: std::iter::once((0, 1))
                .chain((1..shape.rounds).map(|round| (1 + (round - 1) * shape.round_width, shape.round_width)))
                .collect(),
            depths: vec![0; slots],
            expanded: vec![false; slots],
            packed_tree: vec![0; TreeIdx::COUNT * slots],
            frontier: vec![0; FrontierIdx::COUNT * slots * children],
            candidate_ids,
            candidate_logits,
            noise,
            residuals: vec![bf16::ZERO; slots * pool],
            draft_hidden,
            embedding,
        };
        let mut taken = vec![false; slots * children];
        let mut open = Vec::<usize>::new();
        for (round, (start, width)) in step.rounds.clone().into_iter().enumerate() {
            let available = open.clone();
            for slot in start..start + width {
                let (depth, token, parent) = if slot == 0 {
                    (0, 1, NO_PARENT)
                } else {
                    let candidates = available.iter().copied().filter(|&entry| !taken[entry]);
                    let depth_of = |entry: &usize| step.frontier_field(FrontierIdx::Depth, *entry);
                    let entry = if (slot - start) % 2 == 0 {
                        candidates.max_by_key(|entry| (depth_of(entry), std::cmp::Reverse(*entry)))
                    } else {
                        candidates.min_by_key(|entry| (depth_of(entry), *entry))
                    }
                    .expect("the grown tree has frontier entries left");
                    taken[entry] = true;
                    (depth_of(&entry), step.frontier_field(FrontierIdx::TokenId, entry), (entry / children) as u32)
                };
                for (field, value) in [
                    (TreeIdx::TokenId, token),
                    (TreeIdx::ParentSlot, parent),
                    (TreeIdx::Depth, depth),
                    (TreeIdx::Valid, 1),
                ] {
                    step.packed_tree[field as usize * slots + slot] = value;
                }
                step.depths[slot] = depth;
                if depth as usize >= shape.max_depth {
                    continue;
                }
                step.expanded[slot] = true;
                for index in 0..pool {
                    step.residuals[slot * pool + index] = bf16::from_f64(0.5 * rng.uniform());
                }
                let logits = step.logits(slot);
                let mut order = (0..pool).collect::<Vec<_>>();
                order.sort_by(|&one, &two| logits[two].total_cmp(&logits[one]).then(one.cmp(&two)));
                let row = step.row(slot);
                for (child, &index) in order[..children].iter().enumerate() {
                    let entry = slot * children + child;
                    for (field, value) in [
                        (FrontierIdx::TokenId, step.candidate_ids[row * pool + index]),
                        (FrontierIdx::ParentSlot, slot as u32),
                        (FrontierIdx::Depth, depth + 1),
                        (FrontierIdx::Active, 1),
                    ] {
                        step.frontier[field as usize * slots * children + entry] = value;
                    }
                    if round + 1 < step.rounds.len() {
                        open.push(entry);
                    }
                }
            }
        }
        step
    }

    fn slots(&self) -> usize {
        self.depths.len()
    }

    fn row(
        &self,
        slot: usize,
    ) -> usize {
        (self.depths[slot] as usize).min(self.shape.pool_rows - 1)
    }

    fn tree_field(
        &self,
        field: TreeIdx,
        slot: usize,
    ) -> u32 {
        self.packed_tree[field as usize * self.slots() + slot]
    }

    fn frontier_field(
        &self,
        field: FrontierIdx,
        entry: usize,
    ) -> u32 {
        self.frontier[field as usize * self.slots() * self.shape.children + entry]
    }

    /// l = a + rho of a slot, in f64 (both terms are exact).
    fn logits(
        &self,
        slot: usize,
    ) -> Vec<f64> {
        let (pool, row) = (self.shape.pool, self.row(slot));
        (0..pool)
            .map(|index| {
                f64::from(self.candidate_logits[row * pool + index]) + self.residuals[slot * pool + index].to_f64()
            })
            .collect()
    }

    /// The children's tokens of every slot, in frontier order (zero where the slot was not expanded).
    fn child_tokens(&self) -> Vec<u32> {
        (0..self.slots() * self.shape.children).map(|entry| self.frontier_field(FrontierIdx::TokenId, entry)).collect()
    }
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|value| (value - max).exp()).sum::<f64>().ln()
}

/// log_softmax clamped at LOG_PROB_FLOOR.
fn log_probs(values: &[f64]) -> Vec<f64> {
    let total = log_sum_exp(values);
    values.iter().map(|value| (value - total).max(LOG_PROB_FLOOR)).collect()
}

/// The token table, W_e LN(E_t) + b_e of every token, in f64.
fn reference_table(
    weights: &Weights,
    step: &Step,
) -> Vec<f64> {
    step.embedding
        .chunks(step.shape.model)
        .flat_map(|row| {
            let row = row.iter().map(|value| value.to_f64()).collect::<Vec<_>>();
            weights.dense("token_projection", &weights.norm("token_norm", &row))
        })
        .collect()
}

/// The local part of an expanded slot: z_u, and per child its token vector, standardised slot numbers and pool index.
struct Local {
    node: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    numbers: Vec<Vec<f64>>,
    indices: Vec<usize>,
    logits: Vec<f64>,
    gumbels: Vec<f64>,
}

fn local(
    weights: &Weights,
    step: &Step,
    table: &[f64],
    slot: usize,
) -> Local {
    let Shape {
        pool,
        hidden,
        children,
        model,
        ..
    } = step.shape;
    let row = step.row(slot);
    let draft = (0..pool).map(|index| f64::from(step.candidate_logits[row * pool + index])).collect::<Vec<_>>();
    let logits = step.logits(slot);
    let gumbels = (0..pool).map(|index| f64::from(step.noise[row * pool + index])).collect::<Vec<_>>();
    let log_draft = log_probs(&draft);
    let log_logits = log_probs(&logits);
    // The pool's set encoder: per candidate (log pi_a, log pi_l - log pi_a, G) through two layers, then the
    // pi_l-weighted mean and the maximum through one.
    let encoded = (0..pool)
        .map(|index| {
            let features = [log_draft[index], log_logits[index] - log_draft[index], gumbels[index]];
            let entry = weights.dense("pool_set.entry", &features).into_iter().map(silu).collect::<Vec<_>>();
            weights.dense("pool_set.hidden", &entry)
        })
        .collect::<Vec<_>>();
    let total = log_sum_exp(&logits);
    let mut summary = vec![0.0; 2 * encoded[0].len()];
    let width = encoded[0].len();
    summary[width..].fill(f64::NEG_INFINITY);
    for (index, values) in encoded.iter().enumerate() {
        let weight = (logits[index] - total).exp();
        for unit in 0..width {
            summary[unit] += weight * values[unit];
            summary[width + unit] = summary[width + unit].max(values[unit]);
        }
    }
    let row_hidden =
        step.draft_hidden[(row + 1) * model..(row + 2) * model].iter().map(|value| value.to_f64()).collect::<Vec<_>>();
    let drafter = weights.dense("draft", &weights.norm("draft_norm", &row_hidden));
    let pooled = weights.dense("pool_set.out", &summary);
    let node = drafter.iter().zip(&pooled).map(|(one, two)| one + two).collect::<Vec<_>>();
    let tokens = &step.candidate_ids[row * pool..(row + 1) * pool];
    let (mut vectors, mut numbers, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for child in 0..children {
        let token = step.frontier_field(FrontierIdx::TokenId, slot * children + child);
        let index = tokens.iter().position(|&candidate| candidate == token).expect("children are pool candidates");
        vectors.push(table[token as usize * hidden..(token as usize + 1) * hidden].to_vec());
        let rank = draft.iter().filter(|&&value| value > draft[index]).count() as f64;
        let raw = [log_draft[index], log_logits[index], gumbels[index], rank.ln_1p()];
        let (means, stds, keep) = (weights.get("slot.means"), weights.get("slot.stds"), weights.get("slot.keep"));
        numbers.push(
            (0..SLOT_NUMBERS).map(|number| (raw[number] - means[number]) / stds[number] * keep[number]).collect(),
        );
        indices.push(index);
    }
    Local {
        node,
        vectors,
        numbers,
        indices,
        logits,
        gumbels,
    }
}

/// delta of a child: the edge MLP (input norm, entry, one residual SiLU block, final norm, output) over
/// [z_u; e_c; z_u * e_c; slot numbers].
fn delta(
    weights: &Weights,
    node: &[f64],
    vector: &[f64],
    numbers: &[f64],
) -> f64 {
    let input = node
        .iter()
        .copied()
        .chain(vector.iter().copied())
        .chain(node.iter().zip(vector).map(|(one, two)| one * two))
        .chain(numbers.iter().copied())
        .collect::<Vec<_>>();
    let mut hidden = weights.dense("edge.entry", &weights.norm("edge.input_norm", &input));
    let raised =
        weights.dense("edge.up", &weights.norm("edge.block_norm", &hidden)).into_iter().map(silu).collect::<Vec<_>>();
    for (value, residual) in hidden.iter_mut().zip(weights.dense("edge.down", &raised)) {
        *value += residual;
    }
    weights.dense("edge.output", &weights.norm("edge.final_norm", &hidden))[0]
}

/// min(log q, 0) of every child: log q = log_softmax over the pool of alpha l + beta G + delta at the real children,
/// alpha and beta the softplus of the output layer at z_u capped at LOGIT_CAP.
fn edges(
    weights: &Weights,
    node: &[f64],
    vectors: &[Vec<f64>],
    local: &Local,
    real: &[bool],
) -> Vec<f64> {
    let output = weights.dense("output", node);
    let (alpha, beta) = (softplus(output[0]).min(LOGIT_CAP), softplus(output[1]).min(LOGIT_CAP));
    let mut scores = local
        .logits
        .iter()
        .zip(&local.gumbels)
        .map(|(logit, gumbel)| alpha * logit + beta * gumbel)
        .collect::<Vec<_>>();
    for child in (0..vectors.len()).filter(|&child| real[child]) {
        scores[local.indices[child]] += delta(weights, node, &vectors[child], &local.numbers[child]);
    }
    let total = log_sum_exp(&scores);
    local.indices.iter().map(|&index| (scores[index] - total).min(0.0)).collect()
}

/// What the reference gives for a step: every expanded slot's round edges and final edges, by frontier entry (NaN
/// where the slot was not expanded or the child is not real).
struct Reference {
    round_edges: Vec<f64>,
    final_edges: Vec<f64>,
}

/// The head on a step: the local part of every expanded slot, then attention over the tree's tokens (the root and one
/// per frontier entry: a slot's token is the entry it was expanded from) and the final edges. `real` marks the real
/// children by frontier entry; `table` is the token table both sides read.
fn reference(
    weights: &Weights,
    step: &Step,
    table: &[f64],
    real: &[bool],
) -> Reference {
    let Shape {
        children,
        hidden,
        width,
        heads,
        max_depth,
        layers,
        ..
    } = step.shape;
    let (slots, entries) = (step.slots(), step.slots() * children);
    let locals =
        (0..slots).map(|slot| step.expanded[slot].then(|| local(weights, step, table, slot))).collect::<Vec<_>>();
    let mut round_edges = vec![f64::NAN; entries];
    for (slot, part) in locals.iter().enumerate() {
        if let Some(part) = part {
            let all = vec![true; children];
            for (child, edge) in edges(weights, &part.node, &part.vectors, part, &all).into_iter().enumerate() {
                round_edges[slot * children + child] = edge;
            }
        }
    }

    // Tokens: 0 is the root, 1 + e the child at frontier entry e; slot s > 0 is the token of its parent's entry with
    // its token.
    let tokens = 1 + entries;
    let slot_token = (0..slots)
        .map(|slot| {
            if slot == 0 {
                return Some(0);
            }
            let parent = step.tree_field(TreeIdx::ParentSlot, slot) as usize;
            (0..children)
                .map(|child| parent * children + child)
                .find(|&entry| {
                    real[entry]
                        && step.frontier_field(FrontierIdx::TokenId, entry) == step.tree_field(TreeIdx::TokenId, slot)
                })
                .map(|entry| 1 + entry)
        })
        .collect::<Vec<_>>();
    let present = (0..tokens)
        .map(|token| {
            if token == 0 {
                step.tree_field(TreeIdx::Valid, 0) != 0
            } else {
                real[token - 1]
            }
        })
        .collect::<Vec<_>>();
    let parent_token =
        (0..tokens).map(|token| (token > 0).then(|| slot_token[(token - 1) / children]).flatten()).collect::<Vec<_>>();
    let depth = (0..tokens)
        .map(|token| {
            if token == 0 {
                0
            } else {
                step.frontier_field(FrontierIdx::Depth, token - 1) as usize
            }
        })
        .collect::<Vec<_>>();
    let mut token_slot = vec![None; tokens];
    for (slot, token) in slot_token.iter().enumerate() {
        if let Some(token) = token {
            token_slot[*token] = Some(slot);
        }
    }
    let embedding = weights.get("tree.depth_embedding");
    let mut values = (0..tokens)
        .map(|token| {
            if !present[token] {
                return vec![0.0; width];
            }
            let level = depth[token].min(max_depth);
            let mut value = embedding[level * width..(level + 1) * width].to_vec();
            if token > 0 {
                let entry = token - 1;
                let part = locals[entry / children].as_ref().unwrap();
                let own = part.vectors[entry % children]
                    .iter()
                    .chain(&part.numbers[entry % children])
                    .copied()
                    .collect::<Vec<_>>();
                for (sum, add) in value.iter_mut().zip(weights.dense("tree.child_in", &own)) {
                    *sum += add;
                }
            }
            if let Some(slot) = token_slot[token]
                && let Some(part) = &locals[slot]
                && (0..children).any(|child| real[slot * children + child])
            {
                for (sum, add) in value.iter_mut().zip(weights.dense("tree.node_in", &part.node)) {
                    *sum += add;
                }
            }
            value
        })
        .collect::<Vec<_>>();

    // Relations of every pair of present tokens (query t, key u): self, u the parent of t, u a child of t, u an
    // ancestor of t two or more edges up, u a descendant likewise, u a sibling of t.
    let ancestors = |token: usize| {
        let mut chain = Vec::new();
        let mut walk = parent_token[token];
        while let Some(up) = walk {
            chain.push(up);
            walk = parent_token[up];
        }
        chain
    };
    let chains = (0..tokens).map(ancestors).collect::<Vec<_>>();
    let relation = |query: usize, key: usize| -> [bool; RELATIONS] {
        [
            query == key,
            parent_token[query] == Some(key),
            parent_token[key] == Some(query),
            chains[query].iter().skip(1).any(|&up| up == key),
            chains[key].iter().skip(1).any(|&up| up == query),
            query != key && parent_token[query].is_some() && parent_token[query] == parent_token[key],
        ]
    };
    let relation_weight = weights.get("tree.relation_weight");
    let size = width / heads;
    for layer in 0..layers {
        let prefix = format!("tree.layers.{layer}");
        let qkv = values
            .iter()
            .map(|value| {
                weights.dense(&format!("{prefix}.qkv"), &weights.norm(&format!("{prefix}.attention_norm"), value))
            })
            .collect::<Vec<_>>();
        let mut next = values.clone();
        for query in (0..tokens).filter(|&token| present[token]) {
            let mut mixed = vec![0.0; width];
            for head in 0..heads {
                let keys = (0..tokens).filter(|&token| present[token]).collect::<Vec<_>>();
                let logits = keys
                    .iter()
                    .map(|&key| {
                        let dot = (0..size)
                            .map(|unit| qkv[query][head * size + unit] * qkv[key][width + head * size + unit])
                            .sum::<f64>();
                        let bias = relation(query, key)
                            .iter()
                            .enumerate()
                            .filter(|(_, on)| **on)
                            .map(|(kind, _)| relation_weight[head * RELATIONS + kind])
                            .sum::<f64>();
                        dot / (size as f64).sqrt() + bias
                    })
                    .collect::<Vec<_>>();
                let total = log_sum_exp(&logits);
                for (&key, logit) in keys.iter().zip(&logits) {
                    let probability = (logit - total).exp();
                    for unit in 0..size {
                        mixed[head * size + unit] += probability * qkv[key][2 * width + head * size + unit];
                    }
                }
            }
            let mut value = values[query].clone();
            for (sum, add) in value.iter_mut().zip(weights.dense(&format!("{prefix}.out"), &mixed)) {
                *sum += add;
            }
            let raised = weights
                .dense(&format!("{prefix}.up"), &weights.norm(&format!("{prefix}.mlp_norm"), &value))
                .into_iter()
                .map(silu)
                .collect::<Vec<_>>();
            for (sum, add) in value.iter_mut().zip(weights.dense(&format!("{prefix}.down"), &raised)) {
                *sum += add;
            }
            next[query] = value;
        }
        values = next;
    }
    let finals = values.iter().map(|value| weights.norm("tree.final_norm", value)).collect::<Vec<_>>();

    let mut final_edges = vec![f64::NAN; entries];
    for (slot, part) in locals.iter().enumerate() {
        let Some(part) = part else {
            continue;
        };
        let own = slot_token[slot].expect("an expanded slot has a token");
        let node = part
            .node
            .iter()
            .zip(weights.dense("tree.couple_node", &finals[own]))
            .map(|(one, two)| one + two)
            .collect::<Vec<_>>();
        let vectors = (0..children)
            .map(|child| {
                let shift = weights.dense("tree.couple_child", &finals[1 + slot * children + child]);
                part.vectors[child].iter().zip(shift).map(|(one, two)| one + two).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let real_children = (0..children).map(|child| real[slot * children + child]).collect::<Vec<_>>();
        for (child, edge) in edges(weights, &node, &vectors, part, &real_children).into_iter().enumerate() {
            if real_children[child] {
                final_edges[slot * children + child] = edge;
            }
        }
    }
    let _ = hidden;
    Reference {
        round_edges,
        final_edges,
    }
}

/// The token table rounded to bf16, as the engine keeps it and as both sides read it.
fn table_bf16(table: &[f64]) -> Vec<bf16> {
    table.iter().map(|&value| bf16::from_f64(value)).collect()
}

fn close(
    actual: f32,
    expected: f64,
    tolerance: f32,
    what: &str,
) {
    assert!(
        (f64::from(actual) - expected).abs() <= f64::from(tolerance) * (1.0 + expected.abs()),
        "{what}: {actual} against the reference's {expected}"
    );
}

/// What one run of the head on a step leaves: every round's child edges and the trees after the prune.
struct Run {
    round_edges: Vec<Vec<f32>>,
    packed_tree: Vec<u32>,
    frontier: Vec<u32>,
}

/// The head on a step as the Weaver tree runs it: the step state, every round, the prune. `not_real` marks one frontier
/// child as not real before the prune, with the given pool index.
fn run<B: Backend>(
    head: &HeadDirectory,
    step: &Step,
    table: &[bf16],
    not_real: Option<(usize, u32)>,
) -> Run {
    let Shape {
        children,
        pool,
        pool_rows,
        vocab,
        ..
    } = step.shape;
    let slots = step.slots();
    let entries = slots * children;
    let context = create_context::<B>();
    let head = head.load::<B>(&context);
    assert!(head.tree.is_some());
    let mut frontier = step.frontier.clone();
    if let Some((entry, _)) = not_real {
        frontier[FrontierIdx::Active as usize * entries + entry] = 0;
    }
    let table = create_buffer_with_data::<B, bf16>(&context, table);
    let draft_hidden = create_buffer_with_data::<B, bf16>(&context, &step.draft_hidden);
    let candidate_ids = create_buffer_with_data::<B, u32>(&context, &step.candidate_ids);
    let seeds = create_buffer_with_data::<B, u64>(&context, &vec![0u64; pool_rows]);
    let noise = create_buffer_with_data::<B, f32>(&context, &step.noise);
    let mut packed_tree_buffer = create_buffer_with_data::<B, u32>(&context, &step.packed_tree);
    let mut frontier_buffer = create_buffer_with_data::<B, u32>(&context, &frontier);
    let width_max = step.rounds.iter().map(|&(_, width)| width).max().unwrap() as u32;

    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut state = head
        .encode_step(
            &draft_hidden,
            &candidate_ids,
            &seeds,
            pool_rows as u32,
            slots as u32,
            width_max,
            vocab as u32,
            &mut command_buffer,
        )
        .unwrap();
    // The step's noise replaces the target sampler's, which the pool-noise test checks.
    command_buffer.encode_copy(&noise, &mut state.noise);
    let mut inputs = Vec::new();
    let mut readbacks = Vec::new();
    for &(start, width) in &step.rounds {
        let buffers = round_inputs::<B>(&context, step, &(start..start + width).collect::<Vec<_>>(), None);
        head.encode_round(
            &table,
            &buffers.0,
            &buffers.1,
            &buffers.2,
            &buffers.3,
            &buffers.4,
            &packed_tree_buffer,
            width as u32,
            entries as u32,
            slots as u32,
            &mut frontier_buffer,
            &mut state,
            &mut command_buffer,
        );
        let mut readback = create_buffer::<B, f32>(&context, width_max as usize * children);
        command_buffer.encode_copy(state.child_edges(), &mut readback);
        inputs.push(buffers);
        readbacks.push(readback);
    }
    let patch = not_real.map(|(_, index)| {
        (create_buffer_with_data::<B, u32>(&context, &[0u32]), create_buffer_with_data::<B, u32>(&context, &[index]))
    });
    if let (Some((entry, _)), Some((zero, index))) = (not_real, &patch) {
        let word = size_of::<u32>();
        command_buffer.encode_copy(zero, (&mut state.child_real).subrange_mut(entry * word..(entry + 1) * word));
        command_buffer.encode_copy(index, (&mut state.child_indices).subrange_mut(entry * word..(entry + 1) * word));
    }
    head.encode_prune(&state, &mut packed_tree_buffer, &mut frontier_buffer, &mut command_buffer).unwrap();
    submit_command_buffer(command_buffer);
    let _ = pool;
    Run {
        round_edges: readbacks.iter().map(|readback| buffer_to_vec::<B, f32>(readback)).collect(),
        packed_tree: buffer_to_vec::<B, u32>(&packed_tree_buffer),
        frontier: buffer_to_vec::<B, u32>(&frontier_buffer),
    }
}

/// A round's inputs for the given slots: their residuals, their pool rows' logits and candidates, their metadata and
/// whether each is a valid node: whether it is expanded, or `valid` for all of them.
fn round_inputs<B: Backend>(
    context: &B::Context,
    step: &Step,
    slots: &[usize],
    valid: Option<bool>,
) -> (B::GlobalBuffer, B::GlobalBuffer, B::GlobalBuffer, B::GlobalBuffer, B::GlobalBuffer) {
    let pool = step.shape.pool;
    let row = |slot: usize| step.row(slot) * pool..(step.row(slot) + 1) * pool;
    let width = slots.len();
    let mut metadata = vec![0u32; MetadataIdx::COUNT * width];
    for (index, &slot) in slots.iter().enumerate() {
        metadata[MetadataIdx::Depth as usize * width + index] = step.depths[slot];
        metadata[MetadataIdx::AncestorCount as usize * width + index] = step.depths[slot];
        metadata[MetadataIdx::TreeSlot as usize * width + index] = slot as u32;
    }
    (
        create_buffer_with_data::<B, bf16>(
            context,
            &slots.iter().flat_map(|&slot| step.residuals[slot * pool..(slot + 1) * pool].to_vec()).collect::<Vec<_>>(),
        ),
        create_buffer_with_data::<B, f32>(
            context,
            &slots.iter().flat_map(|&slot| step.candidate_logits[row(slot)].to_vec()).collect::<Vec<_>>(),
        ),
        create_buffer_with_data::<B, u32>(
            context,
            &slots.iter().flat_map(|&slot| step.candidate_ids[row(slot)].to_vec()).collect::<Vec<_>>(),
        ),
        create_buffer_with_data::<B, u32>(context, &metadata),
        create_buffer_with_data::<B, u32>(
            context,
            &slots.iter().map(|&slot| u32::from(valid.unwrap_or(step.expanded[slot]))).collect::<Vec<_>>(),
        ),
    )
}

/// The final edge of every real child of an expanded slot: in the packed tree when the child was expanded into a slot,
/// in the frontier otherwise.
fn final_edges(
    step: &Step,
    run: &Run,
) -> Vec<(usize, f32)> {
    let (slots, children) = (step.slots(), step.shape.children);
    let entries = slots * children;
    let child_tokens = step.child_tokens();
    let packed_tree = &run.packed_tree;
    let mut edges = Vec::new();
    for slot in (0..slots).filter(|&slot| step.expanded[slot]) {
        for child in 0..children {
            let entry = slot * children + child;
            let expanded_into = (1..slots).find(|&candidate| {
                packed_tree[TreeIdx::Valid as usize * slots + candidate] != 0
                    && packed_tree[TreeIdx::ParentSlot as usize * slots + candidate] == slot as u32
                    && packed_tree[TreeIdx::TokenId as usize * slots + candidate] == child_tokens[entry]
            });
            let bits = match expanded_into {
                Some(target) => packed_tree[TreeIdx::EdgeLogprobBits as usize * slots + target],
                None => run.frontier[FrontierIdx::EdgeLogprobBits as usize * entries + entry],
            };
            edges.push((entry, f32::from_bits(bits)));
        }
    }
    edges
}

/// The largest deviations from the reference, relative to 1 + |value|, on stderr: a failure then shows how far off the
/// whole set of edges is, not only the first edge past the tolerance.
fn report(
    what: &str,
    edges: &[(usize, f32, f64)],
) {
    let mut deviations = edges
        .iter()
        .map(|&(entry, actual, expected)| {
            ((f64::from(actual) - expected).abs() / (1.0 + expected.abs()), entry, actual, expected)
        })
        .collect::<Vec<_>>();
    deviations.sort_by(|left, right| right.0.total_cmp(&left.0));
    let median = deviations[deviations.len() / 2].0;
    eprintln!("{what} edges: {} edges, median relative deviation {median:.2e}, largest:", deviations.len());
    for (deviation, entry, actual, expected) in deviations.iter().take(5) {
        eprintln!("  entry {entry}: {actual} against {expected} ({deviation:.2e})");
    }
}

/// Every round's edges and every final edge of a run against the reference, within `tolerance` relative to 1 + |value|.
fn check_against_reference(
    step: &Step,
    run: &Run,
    expected: &Reference,
    tolerance: f32,
    final_tolerance: f32,
) {
    let children = step.shape.children;
    let mut local = Vec::new();
    for (&(start, width), edges) in step.rounds.iter().zip(&run.round_edges) {
        for index in 0..width {
            let slot = start + index;
            if !step.expanded[slot] {
                continue;
            }
            for child in 0..children {
                let entry = slot * children + child;
                local.push((entry, edges[index * children + child], expected.round_edges[entry]));
            }
        }
    }
    let full = final_edges(step, run)
        .into_iter()
        .filter(|&(entry, _)| !expected.final_edges[entry].is_nan())
        .map(|(entry, edge)| (entry, edge, expected.final_edges[entry]))
        .collect::<Vec<_>>();
    assert!(!local.is_empty() && !full.is_empty());
    report("round", &local);
    report("final", &full);
    for (entry, actual, expected) in local {
        close(actual, expected, tolerance, &format!("round edge of entry {entry}"));
    }
    for (entry, actual, expected) in full {
        close(actual, expected, final_tolerance, &format!("final edge of entry {entry}"));
    }
}

/// The tested head, its step, the reference's token table (bf16, as both sides read it) and the reference's edges with
/// every child of an expanded slot real.
fn small() -> (Weights, HeadDirectory, Step, Vec<bf16>, Reference) {
    let weights = Weights::random(SMALL, 7);
    let directory = HeadDirectory::new(&weights);
    let step = Step::grow(SMALL, 11);
    let table = table_bf16(&reference_table(&weights, &step));
    let as_read = table.iter().map(|value| value.to_f64()).collect::<Vec<_>>();
    let real = all_real(&step);
    let expected = reference(&weights, &step, &as_read, &real);
    (weights, directory, step, table, expected)
}

fn all_real(step: &Step) -> Vec<bool> {
    (0..step.slots() * step.shape.children).map(|entry| step.expanded[entry / step.shape.children]).collect()
}

/// Spec of the grown step: rounds that mix depths, slots at the depth cap (expanded into, no children of their own),
/// and every child of an expanded slot a distinct candidate of its pool row: the cases the head must handle.
#[uzu_test]
fn grown_step_covers_capped_slots_and_mixed_rounds() {
    let step = Step::grow(SMALL, 11);
    assert!((0..step.slots()).any(|slot| !step.expanded[slot] && step.depths[slot] as usize == SMALL.max_depth));
    assert!(step.rounds.iter().any(|&(start, width)| {
        (start..start + width).map(|slot| step.depths[slot]).collect::<std::collections::HashSet<_>>().len() > 1
    }));
    for slot in (0..step.slots()).filter(|&slot| step.expanded[slot]) {
        let tokens = (0..SMALL.children)
            .map(|child| step.child_tokens()[slot * SMALL.children + child])
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(tokens.len(), SMALL.children, "slot {slot}");
    }
}

/// Every round's local edges and the full head's final edges, as the Weaver tree would run them, against the reference.
#[uzu_test]
fn confidence_head_rounds_and_prune_match_the_reference() {
    let (_, directory, step, table, expected) = small();
    check_against_reference(&step, &run::<Cpu>(&directory, &step, &table, None), &expected, TOLERANCE, TOLERANCE);
}

/// The same run on every backend other than the CPU (Metal on Apple hardware), against the reference: the rounds' edges
/// within the CPU's tolerance, the final edges within the engine's matmul rounding (GPU_FINAL_TOLERANCE).
#[uzu_test]
fn confidence_head_matches_the_reference_on_every_backend() {
    let (_, directory, step, table, expected) = small();
    for_each_non_cpu_backend!(|B| {
        check_against_reference(
            &step,
            &run::<B>(&directory, &step, &table, None),
            &expected,
            GPU_ROUND_TOLERANCE,
            GPU_FINAL_TOLERANCE,
        );
    });
}

/// The same against the reference at a deployed head's sizes (DEPLOYED_CHECK: pools of 512, four children, depth 15,
/// 61 slots, four attention layers of width 256 over 245 tokens, Qwen3.8's width), on every backend: the kernels'
/// specialisation to these sizes.
#[uzu_test]
fn confidence_head_matches_the_reference_at_deployed_sizes() {
    let weights = Weights::random(DEPLOYED_CHECK, 23);
    let directory = HeadDirectory::new(&weights);
    let step = Step::grow(DEPLOYED_CHECK, 29);
    assert!((0..step.slots()).any(|slot| !step.expanded[slot]));
    let table = table_bf16(&reference_table(&weights, &step));
    let as_read = table.iter().map(|value| value.to_f64()).collect::<Vec<_>>();
    let expected = reference(&weights, &step, &as_read, &all_real(&step));
    check_against_reference(&step, &run::<Cpu>(&directory, &step, &table, None), &expected, TOLERANCE, TOLERANCE);
    for_each_non_cpu_backend!(|B| {
        check_against_reference(
            &step,
            &run::<B>(&directory, &step, &table, None),
            &expected,
            GPU_ROUND_TOLERANCE,
            GPU_FINAL_TOLERANCE,
        );
    });
}

/// A child that is not real adds no delta and has no token: the final edges are the reference's with that child not
/// real (spec), and its pool index may be anything: two different values leave every other edge bitwise the same (MR
/// from the prune's masks).
#[uzu_test]
fn confidence_head_prune_ignores_the_pool_index_of_a_child_that_is_not_real() {
    let (weights, directory, step, table, _) = small();
    let children = SMALL.children;
    let child_tokens = step.child_tokens();
    // A child still in the frontier: dropping it leaves no node without its parent.
    let entry = (0..step.slots() * children)
        .find(|&entry| {
            step.expanded[entry / children]
                && !(1..step.slots()).any(|slot| {
                    step.tree_field(TreeIdx::ParentSlot, slot) as usize == entry / children
                        && step.tree_field(TreeIdx::TokenId, slot) == child_tokens[entry]
                })
        })
        .unwrap();
    let first = final_edges(&step, &run::<Cpu>(&directory, &step, &table, Some((entry, 0))));
    let second = final_edges(&step, &run::<Cpu>(&directory, &step, &table, Some((entry, SMALL.pool as u32 - 1))));
    for ((at, one), (_, two)) in first.iter().zip(&second).filter(|((at, _), _)| *at != entry) {
        assert_eq!(one.to_bits(), two.to_bits(), "entry {at}");
    }
    let mut real = all_real(&step);
    real[entry] = false;
    let as_read = table.iter().map(|value| value.to_f64()).collect::<Vec<_>>();
    let expected = reference(&weights, &step, &as_read, &real);
    for (at, edge) in first.into_iter().filter(|&(at, _)| at != entry) {
        close(edge, expected.final_edges[at], TOLERANCE, &format!("final edge of entry {at}"));
    }
}

/// Spec (encode_round): after its round, every child of an expanded node has in its frontier entry (slot * children +
/// child) the path of its parent in the packed tree plus the head's edge of that child, bit for bit, and the priority
/// key of that path; the token, parent, depth and activity Weaver's top_children wrote stay, and the entries of nodes
/// the rounds did not expand are not touched. Every slot gets its own parent path here. MR on the side: the edges do not
/// depend on the parents' paths. Catches a round that leaves the model's path (the head would not steer the growth),
/// adds the edge to another slot's path, keys the edge instead of the path, or writes another node's entries.
#[uzu_test]
fn confidence_head_round_puts_the_heads_paths_into_the_frontier() {
    let (_, directory, step, table, _) = small();
    check_frontier_paths::<Cpu>(&directory, &step, &table);
    for_each_non_cpu_backend!(|B| {
        check_frontier_paths::<B>(&directory, &step, &table);
    });
}

fn check_frontier_paths<B: Backend>(
    directory: &HeadDirectory,
    step: &Step,
    table: &[bf16],
) {
    let Shape {
        children,
        pool_rows,
        vocab,
        ..
    } = step.shape;
    let slots = step.slots();
    let entries = slots * children;
    let context = create_context::<B>();
    let head = directory.load::<B>(&context);
    // Distinct negative paths, none equal to another's or to an edge's scale: -0.37 per slot below -0.11.
    let parent_path = |slot: usize| -0.11 - 0.37 * slot as f32;
    let mut packed_tree = step.packed_tree.clone();
    for slot in 0..slots {
        packed_tree[TreeIdx::PathLogprobBits as usize * slots + slot] = parent_path(slot).to_bits();
    }
    let before = &step.frontier;
    let table_buffer = create_buffer_with_data::<B, bf16>(&context, table);
    let draft_hidden = create_buffer_with_data::<B, bf16>(&context, &step.draft_hidden);
    let candidate_ids = create_buffer_with_data::<B, u32>(&context, &step.candidate_ids);
    let seeds = create_buffer_with_data::<B, u64>(&context, &vec![0u64; pool_rows]);
    let noise = create_buffer_with_data::<B, f32>(&context, &step.noise);
    let packed_tree_buffer = create_buffer_with_data::<B, u32>(&context, &packed_tree);
    let mut frontier_buffer = create_buffer_with_data::<B, u32>(&context, before);
    let width_max = step.rounds.iter().map(|&(_, width)| width).max().unwrap() as u32;
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut state = head
        .encode_step(
            &draft_hidden,
            &candidate_ids,
            &seeds,
            pool_rows as u32,
            slots as u32,
            width_max,
            vocab as u32,
            &mut command_buffer,
        )
        .unwrap();
    command_buffer.encode_copy(&noise, &mut state.noise);
    let mut inputs = Vec::new();
    let mut readbacks = Vec::new();
    for &(start, width) in &step.rounds {
        let buffers = round_inputs::<B>(&context, step, &(start..start + width).collect::<Vec<_>>(), None);
        head.encode_round(
            &table_buffer,
            &buffers.0,
            &buffers.1,
            &buffers.2,
            &buffers.3,
            &buffers.4,
            &packed_tree_buffer,
            width as u32,
            entries as u32,
            slots as u32,
            &mut frontier_buffer,
            &mut state,
            &mut command_buffer,
        );
        let mut readback = create_buffer::<B, f32>(&context, width_max as usize * children);
        command_buffer.encode_copy(state.child_edges(), &mut readback);
        inputs.push(buffers);
        readbacks.push(readback);
    }
    submit_command_buffer(command_buffer);
    let round_edges = readbacks.iter().map(|readback| buffer_to_vec::<B, f32>(readback)).collect::<Vec<_>>();
    let frontier = buffer_to_vec::<B, u32>(&frontier_buffer);
    let reference_run = run::<B>(directory, step, table, None);

    let field = |frontier: &[u32], index: FrontierIdx, entry: usize| frontier[index as usize * entries + entry];
    // The frontier's key, from its definition: an unsigned order equal to the f32 order (negative values reversed by
    // complementing every bit, non-negative ones above them by setting the sign bit).
    let key = |value: f32| {
        let bits = value.to_bits();
        if bits >> 31 == 1 {
            !bits
        } else {
            bits | (1 << 31)
        }
    };
    let mut written = vec![false; entries];
    let mut paths = Vec::new();
    for ((&(start, width), edges), reference_edges) in
        step.rounds.iter().zip(&round_edges).zip(&reference_run.round_edges)
    {
        for index in 0..width {
            let slot = start + index;
            if !step.expanded[slot] {
                continue;
            }
            for child in 0..children {
                let entry = slot * children + child;
                let edge = edges[index * children + child];
                assert_eq!(edge.to_bits(), reference_edges[index * children + child].to_bits(), "entry {entry}");
                let path = parent_path(slot) + edge;
                assert_eq!(field(&frontier, FrontierIdx::PathLogprobBits, entry), path.to_bits(), "entry {entry}");
                assert_eq!(field(&frontier, FrontierIdx::EdgeLogprobBits, entry), edge.to_bits(), "entry {entry}");
                assert_eq!(field(&frontier, FrontierIdx::PathScoreKey, entry), key(path), "entry {entry}");
                for kept in [FrontierIdx::TokenId, FrontierIdx::ParentSlot, FrontierIdx::Depth, FrontierIdx::Active] {
                    assert_eq!(field(&frontier, kept, entry), field(before, kept, entry), "entry {entry}");
                }
                written[entry] = true;
                paths.push(path);
            }
        }
    }
    // The capped slots' entries are never written.
    assert!(written.iter().any(|&entry| entry) && written.iter().any(|&entry| !entry));
    for entry in (0..entries).filter(|&entry| !written[entry]) {
        for index in 0..FrontierIdx::COUNT {
            assert_eq!(frontier[index * entries + entry], before[index * entries + entry], "entry {entry}");
        }
    }
    // The key orders paths as f32 does.
    for &one in &paths {
        for &two in &paths {
            assert_eq!(one.total_cmp(&two), key(one).cmp(&key(two)), "{one} and {two}");
        }
    }
}

/// MR from the round kernel's contract: a node's edges read only its own row, pool and children, so the expanded slots
/// of every round launched together in one round give bitwise the edges of the rounds as the tree ran them, on every
/// backend. The launch mixes depths, so a node reading another node's depth (and so its pool row) changes its edges.
#[uzu_test]
fn confidence_head_round_does_not_depend_on_the_other_nodes_of_its_launch() {
    let (_, directory, step, table, _) = small();
    launch_independence::<Cpu>(&directory, &step, &table);
    for_each_non_cpu_backend!(|B| {
        launch_independence::<B>(&directory, &step, &table);
    });
}

fn launch_independence<B: Backend>(
    directory: &HeadDirectory,
    step: &Step,
    table_values: &[bf16],
) {
    let Shape {
        children,
        pool_rows,
        vocab,
        ..
    } = step.shape;
    let slots = step.slots();
    let context = create_context::<B>();
    let head = directory.load::<B>(&context);
    let packed_tree = create_buffer_with_data::<B, u32>(&context, &step.packed_tree);
    let mut frontier = create_buffer_with_data::<B, u32>(&context, &step.frontier);
    let nodes = (0..slots).filter(|&slot| step.expanded[slot]).collect::<Vec<_>>();
    assert!(nodes.iter().map(|&slot| step.depths[slot]).collect::<std::collections::HashSet<_>>().len() > 1);
    let buffers = round_inputs::<B>(&context, step, &nodes, Some(true));
    let table = create_buffer_with_data::<B, bf16>(&context, table_values);
    let draft_hidden = create_buffer_with_data::<B, bf16>(&context, &step.draft_hidden);
    let candidate_ids = create_buffer_with_data::<B, u32>(&context, &step.candidate_ids);
    let seeds = create_buffer_with_data::<B, u64>(&context, &vec![0u64; pool_rows]);
    let noise = create_buffer_with_data::<B, f32>(&context, &step.noise);
    let rows = nodes.len() as u32;
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut state = head
        .encode_step(
            &draft_hidden,
            &candidate_ids,
            &seeds,
            pool_rows as u32,
            slots as u32,
            rows,
            vocab as u32,
            &mut command_buffer,
        )
        .unwrap();
    command_buffer.encode_copy(&noise, &mut state.noise);
    head.encode_round(
        &table,
        &buffers.0,
        &buffers.1,
        &buffers.2,
        &buffers.3,
        &buffers.4,
        &packed_tree,
        rows,
        (slots * children) as u32,
        slots as u32,
        &mut frontier,
        &mut state,
        &mut command_buffer,
    );
    let mut together = create_buffer::<B, f32>(&context, nodes.len() * children);
    command_buffer.encode_copy(state.child_edges(), &mut together);
    submit_command_buffer(command_buffer);
    let together = buffer_to_vec::<B, f32>(&together);

    let apart = run::<B>(directory, step, table_values, None);
    for (index, &slot) in nodes.iter().enumerate() {
        let (round, &(start, _)) = step
            .rounds
            .iter()
            .enumerate()
            .find(|(_, (start, width))| (*start..*start + *width).contains(&slot))
            .unwrap();
        for child in 0..children {
            assert_eq!(
                together[index * children + child].to_bits(),
                apart.round_edges[round][(slot - start) * children + child].to_bits(),
                "slot {slot} child {child}"
            );
        }
    }
}

/// The token table the head builds from the target's input embedding: W_e LN(E_t) of every token in bf16, on every
/// backend, against the reference in f64. The engine sums `model` terms in f32 and rounds to bf16: within one bf16 unit
/// of the value plus a few f32 units of the terms' absolute sum (an entry near zero whose terms cancel).
#[uzu_test]
fn confidence_head_token_table_matches_the_reference() {
    let (weights, directory, step, _, _) = small();
    let expected = reference_table(&weights, &step);
    let magnitudes = term_magnitudes(&weights, &step);
    check_table::<Cpu>(&directory, &step, &expected, &magnitudes);
    for_each_non_cpu_backend!(|B| {
        check_table::<B>(&directory, &step, &expected, &magnitudes);
    });
}

fn check_table<B: Backend>(
    directory: &HeadDirectory,
    step: &Step,
    expected: &[f64],
    magnitudes: &[f64],
) {
    let (vocab, hidden) = (step.shape.vocab as u32, step.shape.hidden as u32);
    let context = create_context::<B>();
    let head = directory.load::<B>(&context);
    let embedding = create_buffer_with_data::<B, bf16>(&context, &step.embedding);
    let mut table = create_buffer::<B, bf16>(&context, (vocab * hidden) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    head.project_table.encode(
        &embedding,
        &head.token_norm_scales,
        &head.token_norm_biases,
        &head.token_weights,
        &head.token_biases,
        None::<&mut B::ScratchBuffer>,
        Some(&mut table),
        vocab,
        head.config.layer_norm_eps,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    for (index, actual) in buffer_to_vec::<B, bf16>(&table).into_iter().enumerate() {
        let (actual, expected) = (actual.to_f64(), expected[index]);
        let tolerance = expected.abs() * 2f64.powi(-7) + 4.0 * f64::from(f32::EPSILON) * magnitudes[index] + 1e-6;
        assert!((actual - expected).abs() <= tolerance, "table element {index}: {actual} against {expected}");
    }
}

/// sum_i |W_ui LN(E_t)_i| + |b_u| of every table entry (t, u): the scale of the f32 rounding of its sum.
fn term_magnitudes(
    weights: &Weights,
    step: &Step,
) -> Vec<f64> {
    let (model, hidden) = (step.shape.model, step.shape.hidden);
    let projection = weights.get("token_projection.weights.weights");
    let biases = weights.get("token_projection.biases");
    step.embedding
        .chunks(model)
        .flat_map(|row| {
            let normed = weights.norm("token_norm", &row.iter().map(|value| value.to_f64()).collect::<Vec<_>>());
            (0..hidden)
                .map(|unit| {
                    projection[unit * model..(unit + 1) * model]
                        .iter()
                        .zip(&normed)
                        .map(|(weight, value)| (weight * value).abs())
                        .sum::<f64>()
                        + biases[unit].abs()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The pool noise of row d is the target sampler's at the seed of depth d: Gumbel of the 24-bit uniform at the
/// token's place in the sampler's Philox stream.
#[uzu_test]
fn confidence_head_pool_noise_is_the_target_samplers() {
    const VOCAB: u32 = 248_320;
    let rows = 3usize;
    let pool = 5usize;
    let ids = (0..rows * pool).map(|index| (index as u32 * 7_919 + 13) % VOCAB).collect::<Vec<_>>();
    let seeds = [0x9E37_79B9_7F4A_7C15u64, 0xD1B5_4A32_D192_ED03, 0x2545_F491_4F6C_DD1D];
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as crate::backends::common::Backend>::Kernels as crate::backends::common::Kernels>::ConfidenceHeadPoolNoiseKernel::new(&context).unwrap();
    let ids_buffer = create_buffer_with_data::<Cpu, u32>(&context, &ids);
    let seeds_buffer = create_buffer_with_data::<Cpu, u64>(&context, &seeds);
    let mut noise = create_buffer::<Cpu, f32>(&context, rows * pool);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &ids_buffer,
        &seeds_buffer,
        &mut noise,
        (rows * pool) as u32,
        pool as u32,
        VOCAB,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    let noise = buffer_to_vec::<Cpu, f32>(&noise);
    for row in 0..rows {
        for index in 0..pool {
            let position = row * pool + index;
            assert_eq!(noise[position].to_bits(), gumbel_float(seeds[row], revidx(ids[position], VOCAB)).to_bits());
        }
    }
    // Different seeds per row: the same token draws different noise in different rows.
    assert!((0..pool).any(|index| noise[index] != noise[pool + index]));
}

/// A head of a deployed head's sizes with pseudo-random weights, a step of its sizes and a pseudo-random token table:
/// the timing test's inputs (their values do not matter, only their sizes).
fn deployed() -> (HeadDirectory, Step, Vec<bf16>) {
    let weights = Weights::random(DEPLOYED, 13);
    let step = Step::grow(DEPLOYED, 17);
    let mut rng = Rng(19);
    let table = (0..DEPLOYED.vocab * DEPLOYED.hidden).map(|_| bf16::from_f64(rng.uniform())).collect();
    (HeadDirectory::new(&weights), step, table)
}

/// GPU time of the head's parts on every backend other than the CPU: the step state, all rounds, the prune. Each part is
/// encoded `REPEATS` times into one command buffer and its GPU time divided by that; the median over `SAMPLES` buffers
/// is printed. A head of a deployed head's sizes with pseudo-random weights (DEPLOYED); run by hand:
/// `cargo test ... confidence_head_timing -- --ignored --nocapture`.
#[uzu_test]
#[ignore]
fn confidence_head_timing() {
    let (directory, step, table) = deployed();
    for_each_non_cpu_backend!(|B| {
        timing::<B>(&directory, &step, &table);
    });
}

fn timing<B: Backend>(
    directory: &HeadDirectory,
    step: &Step,
    table_values: &[bf16],
) {
    const REPEATS: u32 = 8;
    const SAMPLES: usize = 25;
    let shape = step.shape;
    let (slots, children, pool, pool_rows, model, vocab, hidden) = (
        step.slots() as u32,
        shape.children as u32,
        shape.pool as u32,
        shape.pool_rows as u32,
        shape.model as u32,
        shape.vocab as u32,
        shape.hidden as u32,
    );
    let rounds = step.rounds.iter().map(|&(start, width)| (start as u32, width as u32)).collect::<Vec<_>>();
    let context = create_context::<B>();
    let head = directory.load::<B>(&context);
    let table = create_buffer_with_data::<B, bf16>(&context, table_values);
    let draft_hidden = create_buffer_with_data::<B, bf16>(&context, &step.draft_hidden);
    let candidate_ids_buffer = create_buffer_with_data::<B, u32>(&context, &step.candidate_ids);
    let seeds = create_buffer_with_data::<B, u64>(&context, &vec![0u64; pool_rows as usize]);
    let mut packed_tree = create_buffer_with_data::<B, u32>(&context, &step.packed_tree);
    let mut frontier = create_buffer_with_data::<B, u32>(&context, &step.frontier);
    let width_max = rounds.iter().map(|&(_, width)| width).max().unwrap();
    let round_inputs = |start: u32, width: u32, valid_nodes: bool| {
        let slots = (start as usize..(start + width) as usize).collect::<Vec<_>>();
        let (residual, logits, ids, metadata, valid) =
            round_inputs::<B>(&context, step, &slots, (!valid_nodes).then_some(false));
        (residual, logits, ids, metadata, valid, width)
    };
    let inputs = rounds.iter().map(|&(start, width)| round_inputs(start, width, true)).collect::<Vec<_>>();
    // One launch over every slot (the local part computed once, after the rounds), and a launch whose nodes are all
    // invalid (each threadgroup returns at once: the cost of a launch itself).
    let all_slots = round_inputs(0, slots, true);
    let no_nodes = round_inputs(rounds[1].0, rounds[1].1, false);

    let encode_step = |command_buffer: &mut _| {
        head.encode_step(
            &draft_hidden,
            &candidate_ids_buffer,
            &seeds,
            pool_rows,
            slots,
            width_max.max(slots),
            vocab,
            command_buffer,
        )
        .unwrap()
    };
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut step = encode_step(&mut command_buffer);
    submit_command_buffer(command_buffer);
    let measure = |encode: &mut dyn FnMut(&mut <B::CommandBuffer as CommandBuffer>::Encoding)| -> Duration {
        let mut samples = (0..=SAMPLES)
            .map(|_| {
                let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                for _ in 0..REPEATS {
                    encode(&mut command_buffer);
                }
                command_buffer.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time() / REPEATS
            })
            .skip(1)
            .collect::<Vec<_>>();
        samples.sort_unstable();
        samples[SAMPLES / 2]
    };
    // Whole steps of the head for half a second first: the GPU's clock settles before anything is measured.
    let warmup = std::time::Instant::now();
    while warmup.elapsed() < Duration::from_millis(500) {
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        let mut warm = encode_step(&mut command_buffer);
        for (residual, logits, ids, metadata, valid, width) in &inputs {
            head.encode_round(
                &table,
                residual,
                logits,
                ids,
                metadata,
                valid,
                &packed_tree,
                *width,
                slots * children,
                slots,
                &mut frontier,
                &mut warm,
                &mut command_buffer,
            );
        }
        head.encode_prune(&warm, &mut packed_tree, &mut frontier, &mut command_buffer).unwrap();
        submit_command_buffer(command_buffer);
    }
    let step_time = measure(&mut |command_buffer| {
        encode_step(command_buffer);
    });
    let rounds_time = measure(&mut |command_buffer| {
        for (residual, logits, ids, metadata, valid, width) in &inputs {
            head.encode_round(
                &table,
                residual,
                logits,
                ids,
                metadata,
                valid,
                &packed_tree,
                *width,
                slots * children,
                slots,
                &mut frontier,
                &mut step,
                command_buffer,
            );
        }
    });
    let prune_time = measure(&mut |command_buffer| {
        head.encode_prune(&step, &mut packed_tree, &mut frontier, command_buffer).unwrap();
    });
    let round_once = |command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
                      step: &mut super::ConfidenceHeadStep<B>,
                      (residual, logits, ids, metadata, valid, width): &(
        B::GlobalBuffer,
        B::GlobalBuffer,
        B::GlobalBuffer,
        B::GlobalBuffer,
        B::GlobalBuffer,
        u32,
    ),
                      frontier: &mut B::GlobalBuffer| {
        head.encode_round(
            &table,
            residual,
            logits,
            ids,
            metadata,
            valid,
            &packed_tree,
            *width,
            slots * children,
            slots,
            frontier,
            step,
            command_buffer,
        );
    };
    let all_slots_time =
        measure(&mut |command_buffer| round_once(command_buffer, &mut step, &all_slots, &mut frontier));
    let no_nodes_time = measure(&mut |command_buffer| round_once(command_buffer, &mut step, &no_nodes, &mut frontier));
    eprintln!(
        "confidence head: {slots} slots, {} rounds, pool {pool}: step {step_time:?}, rounds {rounds_time:?} ({:?} per round), \
         prune {prune_time:?}; one launch over all slots {all_slots_time:?}, a launch of invalid nodes {no_nodes_time:?}",
        rounds.len(),
        rounds_time / rounds.len() as u32
    );

    // The prune with its first k attention layers only: the slope is a layer's cost, the rest the fixed part.
    let layer_count = head.tree.as_ref().unwrap().layers.len();
    for kept in [0, 1, 2, layer_count] {
        let mut ablated = directory.load::<B>(&context);
        let tree = ablated.tree.as_mut().unwrap();
        let mut layers = std::mem::take(&mut tree.layers).into_vec();
        layers.truncate(kept);
        tree.layers = layers.into_boxed_slice();
        let time = measure(&mut |command_buffer| {
            ablated.encode_prune(&step, &mut packed_tree, &mut frontier, command_buffer).unwrap();
        });
        eprintln!("  prune with {kept} attention layers: {time:?}");
    }

    // Every block on its own, on inputs of its shapes in the deployed step: what the head's time is made of.
    use crate::backends::common::kernel::{ActivationKernel, ConfidenceHeadPruneKernel};
    let tree = head.tree.as_ref().unwrap();
    let width = tree.config.width;
    let token_count = 1 + slots * children;
    let mut report = Vec::<(&str, Duration)>::new();

    // The step's constants.
    let row_bytes = model as usize * size_of::<bf16>();
    let mut noise_out = create_buffer::<B, f32>(&context, (pool_rows * pool) as usize);
    let mut rows_out = create_buffer::<B, f32>(&context, (pool_rows * hidden) as usize);
    report.push((
        "step: zero fill",
        measure(&mut |command_buffer| {
            let mut buffer = command_buffer.allocate_scratch_for_shape(&[slots * children], DataType::U32).unwrap();
            command_buffer.encode_fill(&mut buffer, 0);
        }),
    ));
    report.push((
        "step: pool noise",
        measure(&mut |command_buffer| {
            head.pool_noise.encode(
                &candidate_ids_buffer,
                &seeds,
                &mut noise_out,
                pool_rows * pool,
                pool,
                vocab,
                command_buffer,
            );
        }),
    ));
    report.push((
        "step: drafter rows",
        measure(&mut |command_buffer| {
            head.project_rows.encode(
                (&draft_hidden).subrange(row_bytes..),
                &head.draft_norm_scales,
                &head.draft_norm_biases,
                &head.draft_weights,
                &head.draft_biases,
                Some(&mut rows_out),
                None::<&mut B::ScratchBuffer>,
                pool_rows,
                head.config.layer_norm_eps,
                command_buffer,
            );
        }),
    ));

    // The prune's fixed part.
    let mut tokens = create_buffer::<B, f32>(&context, (token_count * width) as usize);
    let mut parents = create_buffer::<B, u32>(&context, token_count as usize);
    let mut token_depths = create_buffer::<B, u32>(&context, token_count as usize);
    let mut present = create_buffer::<B, u32>(&context, token_count as usize);
    let mut token_slots = create_buffer::<B, u32>(&context, token_count as usize);
    let mut ancestors = create_buffer::<B, u32>(&context, (token_count * (head.config.max_depth + 1)) as usize);
    let mut slot_tokens = create_buffer::<B, u32>(&context, slots as usize);
    let node_inputs = create_buffer::<B, f32>(&context, (slots * width) as usize);
    let child_inputs = create_buffer::<B, f32>(&context, (slots * children * width) as usize);
    let linear_time = |linear: &dyn crate::encodable_block::linear::Linear<B>, rows: u32, inputs: u32| {
        measure(&mut |command_buffer| {
            let input = command_buffer.allocate_scratch_for_shape(&[rows, inputs], DataType::F32).unwrap();
            linear.encode(input, rows, command_buffer).unwrap();
        })
    };
    report.push((
        "prune: copy of z_u and child inputs (x2)",
        measure(&mut |command_buffer| {
            let mut target = command_buffer
                .allocate_scratch_for_shape(
                    &[slots * children, hidden + super::layout::SLOT_NUMBERS as u32],
                    DataType::F32,
                )
                .unwrap();
            command_buffer.encode_copy(&step.child_inputs, &mut target);
        }),
    ));
    report.push(("prune: node_in dense (61 x 64 -> 256)", linear_time(tree.node_in.as_ref(), slots, hidden)));
    report.push((
        "prune: child_in dense (244 x 68 -> 256)",
        linear_time(tree.child_in.as_ref(), slots * children, hidden + super::layout::SLOT_NUMBERS as u32),
    ));
    report.push((
        "prune: tree tokens",
        measure(&mut |command_buffer| {
            tree.tree_tokens.encode(
                &packed_tree,
                &frontier,
                &step.child_real,
                &node_inputs,
                &child_inputs,
                &tree.depth_embedding,
                &mut tokens,
                &mut parents,
                &mut token_depths,
                &mut present,
                &mut token_slots,
                &mut ancestors,
                &mut slot_tokens,
                slots,
                children,
                width,
                head.config.max_depth,
                command_buffer,
            );
        }),
    ));
    let norm_time = |norm: &crate::encodable_block::normalization::Normalization<B>| {
        measure(&mut |command_buffer| {
            let input = command_buffer.allocate_scratch_for_shape(&[token_count, width], DataType::F32).unwrap();
            let mut residual = command_buffer.allocate_scratch_for_shape(&[token_count, width], DataType::F32).unwrap();
            norm.encode(&input, 0, token_count, Some(&mut residual), command_buffer).unwrap();
        })
    };

    // One attention layer.
    let layer = &tree.layers[0];
    let qkv = create_buffer_with_data::<B, f32>(&context, &vec![0.01f32; (token_count * 3 * width) as usize]);
    let mut attention = create_buffer::<B, f32>(&context, (token_count * width) as usize);
    report.push(("layer: attention norm", norm_time(&layer.attention_norm)));
    report.push(("layer: qkv dense (245 x 256 -> 768)", linear_time(layer.qkv.as_ref(), token_count, width)));
    report.push((
        "layer: tree attention",
        measure(&mut |command_buffer| {
            tree.tree_attention.encode(
                &qkv,
                &tree.relation_weights,
                &parents,
                &token_depths,
                &present,
                &ancestors,
                &mut attention,
                token_count,
                command_buffer,
            );
        }),
    ));
    report.push(("layer: out dense (245 x 256 -> 256)", linear_time(layer.out.as_ref(), token_count, width)));
    report.push(("layer: mlp norm", norm_time(&layer.mlp_norm)));
    report.push(("layer: up dense (245 x 256 -> 512)", linear_time(layer.up.as_ref(), token_count, width)));
    report.push((
        "layer: silu",
        measure(&mut |command_buffer| {
            let mut buffer = command_buffer
                .allocate_scratch_for_shape(&[token_count, tree.config.mlp_hidden], DataType::F32)
                .unwrap();
            head.activation.encode(
                None::<&B::ScratchBuffer>,
                &mut buffer,
                token_count * tree.config.mlp_hidden,
                crate::backends::common::gpu_types::activation_type::ActivationType::SILU,
                command_buffer,
            );
        }),
    ));
    report.push((
        "layer: down dense (245 x 512 -> 256)",
        linear_time(layer.down.as_ref(), token_count, tree.config.mlp_hidden),
    ));

    // The prune's end.
    report.push(("prune: final norm", norm_time(&tree.final_norm)));
    report.push((
        "prune: couple dense (245 x 256 -> 64, x2)",
        linear_time(tree.couple_node.as_ref(), token_count, width),
    ));
    let node_shifts = create_buffer::<B, f32>(&context, (token_count * hidden) as usize);
    let child_shifts = create_buffer::<B, f32>(&context, (token_count * hidden) as usize);
    report.push((
        "prune: edge kernel (61 slots)",
        measure(&mut |command_buffer| {
            head.prune.encode(
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
                &head.local_weights,
                &mut packed_tree,
                &mut frontier,
                slots,
                pool_rows,
                head.config.cap,
                head.config.layer_norm_eps,
                command_buffer,
            );
        }),
    ));

    // The whole head in one command buffer, as in a step, and the CPU's time to encode it.
    let whole = measure(&mut |command_buffer| {
        let mut state = encode_step(command_buffer);
        for (residual, logits, ids, metadata, valid, width) in &inputs {
            head.encode_round(
                &table,
                residual,
                logits,
                ids,
                metadata,
                valid,
                &packed_tree,
                *width,
                slots * children,
                slots,
                &mut frontier,
                &mut state,
                command_buffer,
            );
        }
        head.encode_prune(&state, &mut packed_tree, &mut frontier, command_buffer).unwrap();
    });
    let mut encodings = (0..SAMPLES)
        .map(|_| {
            let mut command_buffer = context.create_command_buffer(None, None).unwrap();
            let started = std::time::Instant::now();
            let mut state = encode_step(&mut command_buffer);
            for (residual, logits, ids, metadata, valid, width) in &inputs {
                head.encode_round(
                    &table,
                    residual,
                    logits,
                    ids,
                    metadata,
                    valid,
                    &packed_tree,
                    *width,
                    slots * children,
                    slots,
                    &mut frontier,
                    &mut state,
                    &mut command_buffer,
                );
            }
            head.encode_prune(&state, &mut packed_tree, &mut frontier, &mut command_buffer).unwrap();
            let elapsed = started.elapsed();
            submit_command_buffer(command_buffer);
            elapsed
        })
        .collect::<Vec<_>>();
    encodings.sort_unstable();

    eprintln!("  breakdown ({token_count} tokens, width {width}):");
    for (name, time) in &report {
        eprintln!("    {name:<45} {:>8.2} us", time.as_secs_f64() * 1e6);
    }
    eprintln!("    {:<45} {:>8.2} us", "whole head, one command buffer (GPU)", whole.as_secs_f64() * 1e6);
    eprintln!("    {:<45} {:>8.2} us", "whole head, encoding (CPU)", encodings[SAMPLES / 2].as_secs_f64() * 1e6);
}

/// How far each block of the first attention layer is, on every backend other than the CPU, from the same block on the
/// CPU, on one pseudo-random input: largest |difference| over the largest |CPU output| and the median relative
/// difference, at a deployed head's sizes. A diagnostic, run by hand with `--ignored --nocapture`.
#[uzu_test]
#[ignore]
fn confidence_head_blocks_against_the_cpu() {
    let (directory, step, _) = deployed();
    let reference = block_outputs::<Cpu>(&directory, &step);
    for_each_non_cpu_backend!(|B| {
        for ((name, expected), (_, actual)) in reference.iter().zip(block_outputs::<B>(&directory, &step)) {
            let scale = expected.iter().fold(0.0f32, |top, value| top.max(value.abs()));
            let largest = expected.iter().zip(&actual).fold(0.0f32, |top, (e, a)| top.max((a - e).abs()));
            let mut relative =
                expected.iter().zip(&actual).map(|(e, a)| (a - e).abs() / (e.abs() + 1e-3 * scale)).collect::<Vec<_>>();
            relative.sort_by(f32::total_cmp);
            eprintln!(
                "{name}: largest difference {:.2e} of the output scale {scale:.3}, median relative {:.2e}",
                largest / scale,
                relative[relative.len() / 2]
            );
        }
    });
}

fn block_outputs<B: Backend>(
    directory: &HeadDirectory,
    step: &Step,
) -> Vec<(&'static str, Vec<f32>)> {
    use crate::backends::common::kernel::ActivationKernel;
    let context = create_context::<B>();
    let head = directory.load::<B>(&context);
    let tree = head.tree.as_ref().unwrap();
    let layer = &tree.layers[0];
    let rows = (1 + step.slots() * step.shape.children) as u32;
    let (width, raised) = (tree.config.width, tree.config.mlp_hidden);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let input = (0..rows * width.max(raised))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        })
        .collect::<Vec<_>>();
    let source = create_buffer_with_data::<B, f32>(&context, &input);
    let run =
        |encode: &dyn Fn(B::ScratchBuffer, &mut <B::CommandBuffer as CommandBuffer>::Encoding) -> B::ScratchBuffer,
         inputs: u32,
         outputs: u32|
         -> Vec<f32> {
            let mut command_buffer = context.create_command_buffer(None, None).unwrap();
            let mut scratch = command_buffer.allocate_scratch_for_shape(&[rows, inputs], DataType::F32).unwrap();
            let bytes = (rows * inputs) as usize * size_of::<f32>();
            command_buffer.encode_copy((&source).subrange(0..bytes), &mut scratch);
            let output = encode(scratch, &mut command_buffer);
            let mut readback = create_buffer::<B, f32>(&context, (rows * outputs) as usize);
            command_buffer.encode_copy(&output, &mut readback);
            submit_command_buffer(command_buffer);
            buffer_to_vec::<B, f32>(&readback)
        };
    vec![
        (
            "layer norm",
            run(
                &|input, command_buffer| {
                    // The head's norms add the residual stream first: a zero residual leaves the input's norm.
                    let mut residual =
                        command_buffer.allocate_scratch_for_shape(&[rows, width], DataType::F32).unwrap();
                    command_buffer.encode_fill(&mut residual, 0);
                    layer.attention_norm.encode(&input, 0, rows, Some(&mut residual), command_buffer).unwrap()
                },
                width,
                width,
            ),
        ),
        ("qkv", run(&|input, command_buffer| layer.qkv.encode(input, rows, command_buffer).unwrap(), width, 3 * width)),
        ("up", run(&|input, command_buffer| layer.up.encode(input, rows, command_buffer).unwrap(), width, raised)),
        ("down", run(&|input, command_buffer| layer.down.encode(input, rows, command_buffer).unwrap(), raised, width)),
        (
            "silu",
            run(
                &|mut input, command_buffer| {
                    head.activation.encode(
                        None::<&B::ScratchBuffer>,
                        &mut input,
                        rows * raised,
                        crate::backends::common::gpu_types::activation_type::ActivationType::SILU,
                        command_buffer,
                    );
                    input
                },
                raised,
                raised,
            ),
        ),
    ]
}
