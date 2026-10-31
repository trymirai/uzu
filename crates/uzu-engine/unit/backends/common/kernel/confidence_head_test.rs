use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::{
                confidence_head::ConfidenceHeadParams,
                weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            },
            kernel::{
                ConfidenceHeadEdgesKernel, ConfidenceHeadNodesKernel, ConfidenceHeadPoolsKernel,
                ConfidenceHeadTreeAttentionKernel, ConfidenceHeadTreeInputKernel, ConfidenceHeadTreeRelationsKernel,
            },
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const NONE: u32 = u32::MAX;
const SLOTS: usize = 5;
const CHILDREN: usize = 4;
const CAPACITY: usize = SLOTS * CHILDREN;
const ROWS: usize = 1 + CAPACITY;
const STRIDE: usize = 4;
const CANDIDATES: u32 = 300;
const VOCAB: u32 = 65_536;
const WIDTH: u32 = 16;
const POOL_HIDDEN: u32 = 8;
const EDGE_HIDDEN: u32 = 32;
const SLOT_ENTRY: [u32; SLOTS] = [NONE, 1, 2, 5, 13];
const EXPANDED: [bool; SLOTS] = [true, true, false, true, true];
const CHILD_INDEX: [u32; CHILDREN] = [5, 290, 0, 17];
const DEPTHS: usize = 4;

fn values(
    count: usize,
    salt: f32,
    scale: f32,
) -> Vec<f32> {
    (0..count).map(|index| ((index as f32 * 0.731 + salt).sin() * 43758.547).fract() * scale).collect()
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn floats(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn assert_close(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{name}: lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{name}[{index}]: {actual} != {expected}"
        );
    }
}

fn pool_token(
    slot: usize,
    index: u32,
) -> u32 {
    1000 + slot as u32 * 7919 + index * 37
}

fn depth(slot: usize) -> usize {
    match SLOT_ENTRY[slot] {
        NONE => 0,
        entry => 1 + depth(entry as usize / CHILDREN),
    }
}

fn entry_token(entry: usize) -> u32 {
    let (slot, child) = (entry / CHILDREN, entry % CHILDREN);
    pool_token(slot, CHILD_INDEX[child] + slot as u32)
}

struct Tree {
    frontier: Vec<u32>,
    packed_tree: Vec<u32>,
    slot_ancestors: Vec<u32>,
    slot_rows: Vec<u32>,
    row_slots: Vec<u32>,
}

fn tree() -> Tree {
    let mut frontier = vec![0; FrontierIdx::COUNT * CAPACITY];
    for entry in (0..CAPACITY).filter(|entry| EXPANDED[entry / CHILDREN]) {
        let slot = entry / CHILDREN;
        frontier[FrontierIdx::TokenId as usize * CAPACITY + entry] = entry_token(entry);
        frontier[FrontierIdx::ParentSlot as usize * CAPACITY + entry] = slot as u32;
        frontier[FrontierIdx::Depth as usize * CAPACITY + entry] = depth(slot) as u32 + 1;
        frontier[FrontierIdx::Active as usize * CAPACITY + entry] = u32::from(!SLOT_ENTRY.contains(&(entry as u32)));
    }
    let mut packed_tree = vec![0; TreeIdx::COUNT * SLOTS];
    let mut slot_ancestors = vec![0; SLOTS * STRIDE];
    let mut slot_rows = vec![NONE; SLOTS];
    let mut row_slots = vec![NONE; ROWS];
    for slot in 0..SLOTS {
        let entry = SLOT_ENTRY[slot];
        let field = |index: TreeIdx| index as usize * SLOTS + slot;
        packed_tree[field(TreeIdx::TokenId)] = if entry == NONE {
            7
        } else {
            entry_token(entry as usize)
        };
        packed_tree[field(TreeIdx::ParentSlot)] = if entry == NONE {
            NONE
        } else {
            entry / CHILDREN as u32
        };
        packed_tree[field(TreeIdx::Depth)] = depth(slot) as u32;
        packed_tree[field(TreeIdx::PathLogprobBits)] = (-0.3 * slot as f32).to_bits();
        packed_tree[field(TreeIdx::Valid)] = 1;
        let mut ancestor = slot;
        while SLOT_ENTRY[ancestor] != NONE {
            ancestor = SLOT_ENTRY[ancestor] as usize / CHILDREN;
            slot_ancestors[slot * STRIDE + depth(ancestor)] = ancestor as u32;
        }
        if EXPANDED[slot] {
            let row = if entry == NONE {
                0
            } else {
                1 + entry
            };
            slot_rows[slot] = row;
            row_slots[row as usize] = slot as u32;
        }
    }
    Tree {
        frontier,
        packed_tree,
        slot_ancestors,
        slot_rows,
        row_slots,
    }
}

fn params() -> ConfidenceHeadParams {
    ConfidenceHeadParams {
        candidates: CANDIDATES,
        children: CHILDREN as u32,
        tree_slot_count: SLOTS as u32,
        vocab_size: VOCAB,
        logit_cap: 50.0,
        epsilon: 1e-5,
    }
}

fn weights<const N: usize>(
    sizes: [usize; N],
    salt: f32,
) -> [Vec<f32>; N] {
    std::array::from_fn(|index| values(sizes[index], salt + index as f32, 0.6))
}

fn node_weights() -> [Vec<f32>; 8] {
    let (width, pool) = (WIDTH as usize, POOL_HIDDEN as usize);
    let mut weights = weights([3 * pool, pool, pool * pool, pool, width * 2 * pool, width, 4, 4], 0.7);
    weights[7] = vec![1.0, 1.5, 2.0, 2.5];
    weights
}

fn edge_weights() -> [Vec<f32>; 16] {
    let (width, hidden) = (WIDTH as usize, EDGE_HIDDEN as usize);
    let inputs = 3 * width + 4;
    weights(
        [
            inputs,
            inputs,
            width * inputs,
            width,
            width,
            width,
            hidden * width,
            hidden,
            width * hidden,
            width,
            width,
            width,
            width,
            1,
            2 * width,
            2,
        ],
        3.3,
    )
}

fn run_pools<B: Backend>() -> Vec<f32> {
    let context = create_context::<B>();
    let candidate_logits = create_buffer_with_data::<B, f32>(&context, &values(DEPTHS * CANDIDATES as usize, 0.3, 6.0));
    let candidate_ids = create_buffer_with_data::<B, u32>(
        &context,
        &(0..DEPTHS).flat_map(|depth| (0..CANDIDATES).map(move |index| pool_token(depth, index))).collect::<Vec<_>>(),
    );
    let depth_seeds = create_buffer_with_data::<B, u64>(&context, &[0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 7, 9]);
    let mut pool_features = create_buffer_with_data::<B, f32>(&context, &vec![0.0; DEPTHS * 3 * CANDIDATES as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &candidate_logits,
        &candidate_ids,
        &depth_seeds,
        &mut pool_features,
        params(),
        DEPTHS as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, f32>(&context, &pool_features)
}

#[uzu_test]
fn pools_match_cpu() {
    let expected = run_pools::<Cpu>();
    for_each_non_cpu_backend!(|B| assert_close("pool features", &run_pools::<B>(), &expected, 1e-4));
}

fn run_nodes<B: Backend>() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>) {
    let context = create_context::<B>();
    let tree = tree();
    let mut metadata = vec![0; MetadataIdx::COUNT * SLOTS];
    for slot in 0..SLOTS {
        metadata[MetadataIdx::Depth as usize * SLOTS + slot] = depth(slot) as u32;
        metadata[MetadataIdx::AncestorCount as usize * SLOTS + slot] = depth(slot) as u32;
        metadata[MetadataIdx::TreeSlot as usize * SLOTS + slot] = slot as u32;
    }
    let ids = (0..DEPTHS)
        .flat_map(|depth| {
            let slot = (0..SLOTS).find(|&slot| self::depth(slot) == depth).unwrap_or(0);
            (0..CANDIDATES).map(move |index| pool_token(slot, index))
        })
        .collect::<Vec<_>>();
    let mut features = values(DEPTHS * 3 * CANDIDATES as usize, 4.1, 3.0);
    for depth in 0..DEPTHS {
        for index in 0..CANDIDATES as usize {
            features[depth * 3 * CANDIDATES as usize + index] =
                -features[depth * 3 * CANDIDATES as usize + index].abs();
        }
    }
    let u32s = |data: &[u32]| create_buffer_with_data::<B, u32>(&context, data);
    let f32s = |data: &[f32]| create_buffer_with_data::<B, f32>(&context, data);
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let residual_logits = bf16_buffer(&values(SLOTS * CANDIDATES as usize, 1.7, 2.0));
    let candidate_logits = f32s(&values(DEPTHS * CANDIDATES as usize, 0.3, 6.0));
    let candidate_ids = u32s(&ids);
    let pool_features = f32s(&features);
    let node_metadata = u32s(&metadata);
    let node_valid = u32s(&EXPANDED.map(u32::from));
    let packed_tree = u32s(&tree.packed_tree);
    let frontier = u32s(&tree.frontier);
    let draft = bf16_buffer(&values(DEPTHS * WIDTH as usize, 2.3, 1.0));
    let [
        pool_entry_weights,
        pool_entry_biases,
        pool_hidden_weights,
        pool_hidden_biases,
        pool_out_weights,
        pool_out_biases,
        slot_mean,
        slot_std,
    ] = node_weights().map(|values| bf16_buffer(&values));
    let mut scores = f32s(&vec![0.0; SLOTS * 2 * CANDIDATES as usize]);
    let mut nodes = bf16_buffer(&vec![0.0; SLOTS * WIDTH as usize]);
    let mut node_inputs = bf16_buffer(&vec![0.0; SLOTS * WIDTH as usize]);
    let mut child_pools = u32s(&[0; CAPACITY]);
    let mut child_numbers = f32s(&vec![0.0; CAPACITY * 4]);
    let mut slot_rows = u32s(&[NONE; SLOTS]);
    let mut row_slots = u32s(&[NONE; ROWS]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadNodesKernel::new(&context, WIDTH, POOL_HIDDEN).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &residual_logits,
        &candidate_logits,
        &candidate_ids,
        &pool_features,
        &node_metadata,
        &node_valid,
        &packed_tree,
        &frontier,
        &draft,
        &pool_entry_weights,
        &pool_entry_biases,
        &pool_hidden_weights,
        &pool_hidden_biases,
        &pool_out_weights,
        &pool_out_biases,
        &slot_mean,
        &slot_std,
        &mut scores,
        &mut nodes,
        &mut node_inputs,
        &mut child_pools,
        &mut child_numbers,
        &mut slot_rows,
        &mut row_slots,
        params(),
        SLOTS as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (
        buffer_to_vec::<B, f32>(&context, &scores),
        floats(&buffer_to_vec::<B, bf16>(&context, &nodes)),
        floats(&buffer_to_vec::<B, bf16>(&context, &node_inputs)),
        buffer_to_vec::<B, u32>(&context, &child_pools),
        buffer_to_vec::<B, f32>(&context, &child_numbers),
        buffer_to_vec::<B, u32>(&context, &slot_rows),
        buffer_to_vec::<B, u32>(&context, &row_slots),
    )
}

#[uzu_test]
fn nodes_match_cpu() {
    let (scores, nodes, node_inputs, child_pools, child_numbers, slot_rows, row_slots) = run_nodes::<Cpu>();
    let tree = tree();
    assert_eq!(slot_rows, tree.slot_rows);
    assert_eq!(row_slots, tree.row_slots);
    assert_eq!(node_inputs, nodes);
    for_each_non_cpu_backend!(|B| {
        let actual = run_nodes::<B>();
        assert_close("scores", &actual.0, &scores, 1e-4);
        assert_close("nodes", &actual.1, &nodes, 2e-2);
        assert_eq!(actual.2, actual.1);
        assert_eq!(actual.3, child_pools);
        assert_close("child numbers", &actual.4, &child_numbers, 1e-4);
        assert_eq!(actual.5, slot_rows);
        assert_eq!(actual.6, row_slots);
    });
}

fn frontier_floats(
    frontier: &[u32],
    field: FrontierIdx,
) -> Vec<f32> {
    frontier[field as usize * CAPACITY..(field as usize + 1) * CAPACITY]
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect()
}

fn packed_edges(packed_tree: &[u32]) -> Vec<f32> {
    packed_tree[TreeIdx::EdgeLogprobBits as usize * SLOTS..][..SLOTS].iter().map(|bits| f32::from_bits(*bits)).collect()
}

fn run_edges<B: Backend>(refine: bool) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let tree = tree();
    let mut metadata = vec![0; MetadataIdx::COUNT * SLOTS];
    for slot in 0..SLOTS {
        metadata[MetadataIdx::TreeSlot as usize * SLOTS + slot] = slot as u32;
    }
    let pools =
        (0..CAPACITY).map(|entry| CHILD_INDEX[entry % CHILDREN] + (entry / CHILDREN) as u32).collect::<Vec<_>>();
    let u32s = |data: &[u32]| create_buffer_with_data::<B, u32>(&context, data);
    let f32s = |data: &[f32]| create_buffer_with_data::<B, f32>(&context, data);
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let node_metadata = u32s(&metadata);
    let slot_rows = u32s(&tree.slot_rows);
    let nodes = bf16_buffer(&values(SLOTS * WIDTH as usize, 1.1, 1.0));
    let node_shifts = bf16_buffer(&values(ROWS * WIDTH as usize, 1.3, 0.5));
    let child_shifts = bf16_buffer(&values(ROWS * WIDTH as usize, 1.9, 0.5));
    let token_features = bf16_buffer(&values(VOCAB as usize * WIDTH as usize, 3.1, 1.0));
    let child_numbers = f32s(&values(CAPACITY * 4, 2.7, 1.0));
    let child_pools = u32s(&pools);
    let scores = f32s(&values(SLOTS * 2 * CANDIDATES as usize, 0.5, 4.0));
    let [
        input_norm_scales,
        input_norm_biases,
        entry_weights,
        entry_biases,
        block_norm_scales,
        block_norm_biases,
        up_weights,
        up_biases,
        down_weights,
        down_biases,
        final_norm_scales,
        final_norm_biases,
        output_weights,
        output_biases,
        alpha_beta_weights,
        alpha_beta_biases,
    ] = edge_weights().map(|values| bf16_buffer(&values));
    let mut frontier = u32s(&tree.frontier);
    let mut packed_tree = u32s(&tree.packed_tree);
    let mut child_inputs = bf16_buffer(&vec![0.0; CAPACITY * (WIDTH as usize + 4)]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel::new(&context, WIDTH, EDGE_HIDDEN, refine).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        (!refine).then_some(&node_metadata),
        &slot_rows,
        &nodes,
        refine.then_some(&node_shifts),
        refine.then_some(&child_shifts),
        &token_features,
        &child_numbers,
        &child_pools,
        &scores,
        &mut frontier,
        &mut packed_tree,
        (!refine).then_some(&mut child_inputs),
        &input_norm_scales,
        &input_norm_biases,
        &entry_weights,
        &entry_biases,
        &block_norm_scales,
        &block_norm_biases,
        &up_weights,
        &up_biases,
        &down_weights,
        &down_biases,
        &final_norm_scales,
        &final_norm_biases,
        &output_weights,
        &output_biases,
        &alpha_beta_weights,
        &alpha_beta_biases,
        params(),
        SLOTS as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (
        buffer_to_vec::<B, u32>(&context, &frontier),
        buffer_to_vec::<B, u32>(&context, &packed_tree),
        floats(&buffer_to_vec::<B, bf16>(&context, &child_inputs)),
    )
}

#[uzu_test]
fn edges_match_cpu() {
    for refine in [false, true] {
        let (frontier, packed_tree, child_inputs) = run_edges::<Cpu>(refine);
        let edges = frontier_floats(&frontier, FrontierIdx::EdgeLogprobBits);
        assert!(edges[..8].iter().chain(&edges[12..]).all(|edge| *edge < 0.0));
        if refine {
            assert!(packed_edges(&packed_tree)[1..].iter().all(|edge| *edge < 0.0));
        }
        for_each_non_cpu_backend!(|B| {
            let actual = run_edges::<B>(refine);
            for field in [FrontierIdx::EdgeLogprobBits, FrontierIdx::PathLogprobBits] {
                assert_close("frontier", &frontier_floats(&actual.0, field), &frontier_floats(&frontier, field), 1e-3);
            }
            assert_close("tree edges", &packed_edges(&actual.1), &packed_edges(&packed_tree), 1e-3);
            assert_close("child inputs", &actual.2, &child_inputs, 2e-2);
        });
    }
}

fn run_tree_input<B: Backend>() -> Vec<f32> {
    let context = create_context::<B>();
    let tree = tree();
    let width = 32u32;
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let depth_embedding = bf16_buffer(&values((STRIDE + 1) * width as usize, 0.2, 1.0));
    let child_in = bf16_buffer(&values(CAPACITY * width as usize, 0.4, 1.0));
    let node_in = bf16_buffer(&values(SLOTS * width as usize, 0.6, 1.0));
    let frontier = create_buffer_with_data::<B, u32>(&context, &tree.frontier);
    let row_slots = create_buffer_with_data::<B, u32>(&context, &tree.row_slots);
    let mut output = bf16_buffer(&vec![0.0; ROWS * width as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeInputKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &depth_embedding,
        &child_in,
        &node_in,
        &frontier,
        &row_slots,
        &mut output,
        ROWS as u32,
        width,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    floats(&buffer_to_vec::<B, bf16>(&context, &output))
}

#[uzu_test]
fn tree_input_matches_cpu() {
    let expected = run_tree_input::<Cpu>();
    assert!(expected[9 * 32..13 * 32].iter().all(|value| *value == 0.0));
    for_each_non_cpu_backend!(|B| assert_close("tree_input", &run_tree_input::<B>(), &expected, 1e-2));
}

fn run_tree_relations<B: Backend>() -> Vec<u32> {
    let context = create_context::<B>();
    let tree = tree();
    let frontier = create_buffer_with_data::<B, u32>(&context, &tree.frontier);
    let slot_ancestors = create_buffer_with_data::<B, u32>(&context, &tree.slot_ancestors);
    let slot_rows = create_buffer_with_data::<B, u32>(&context, &tree.slot_rows);
    let mut relations = create_buffer_with_data::<B, u32>(&context, &[0; ROWS * ROWS]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeRelationsKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &frontier,
        &slot_ancestors,
        &slot_rows,
        &mut relations,
        ROWS as u32,
        STRIDE as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, u32>(&context, &relations)
}

#[uzu_test]
fn tree_relations_match_cpu() {
    let expected = run_tree_relations::<Cpu>();
    let relation = |query: usize, key: usize| expected[query * ROWS + key];
    assert_eq!([relation(6, 6), relation(6, 2), relation(2, 6), relation(6, 0), relation(0, 14)], [1, 2, 3, 4, 5]);
    assert_eq!([relation(6, 7), relation(9, 0), relation(6, 3)], [6, 7, 0]);
    for_each_non_cpu_backend!(|B| assert_eq!(run_tree_relations::<B>(), expected));
}

fn run_tree_attention<B: Backend>() -> Vec<f32> {
    let (heads, head_dim) = (2u32, 32u32);
    let context = create_context::<B>();
    let codes = (0..ROWS * ROWS)
        .map(|index| match (index / ROWS, index % ROWS) {
            (row, _) | (_, row) if (9..13).contains(&row) => 7,
            (query, key) if query == key => 1,
            _ => (index as u32 * 5 + (index / ROWS) as u32) % 7,
        })
        .collect::<Vec<_>>();
    let qkv =
        create_buffer_with_data::<B, bf16>(&context, &bf16s(&values(ROWS * 3 * (heads * head_dim) as usize, 0.6, 2.0)));
    let relations = create_buffer_with_data::<B, u32>(&context, &codes);
    let relation_bias = create_buffer_with_data::<B, bf16>(&context, &bf16s(&values(12, 2.2, 3.0)));
    let mut output =
        create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; ROWS * (heads * head_dim) as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel::new(&context, heads, head_dim).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&qkv, &relations, &relation_bias, &mut output, ROWS as u32, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    floats(&buffer_to_vec::<B, bf16>(&context, &output))
}

#[uzu_test]
fn tree_attention_matches_cpu() {
    let expected = run_tree_attention::<Cpu>();
    assert!(expected[9 * 64..13 * 64].iter().all(|value| *value == 0.0));
    for_each_non_cpu_backend!(|B| assert_close("tree_attention", &run_tree_attention::<B>(), &expected, 1e-2));
}
