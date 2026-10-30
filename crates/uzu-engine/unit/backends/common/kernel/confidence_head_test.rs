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
        packed_tree[field(TreeIdx::TokenId)] = if entry == NONE { 7 } else { entry_token(entry as usize) };
        packed_tree[field(TreeIdx::ParentSlot)] = if entry == NONE { NONE } else { entry / CHILDREN as u32 };
        packed_tree[field(TreeIdx::Depth)] = depth(slot) as u32;
        packed_tree[field(TreeIdx::PathLogprobBits)] = (-0.3 * slot as f32).to_bits();
        packed_tree[field(TreeIdx::Valid)] = 1;
        let mut ancestor = slot;
        while SLOT_ENTRY[ancestor] != NONE {
            ancestor = SLOT_ENTRY[ancestor] as usize / CHILDREN;
            slot_ancestors[slot * STRIDE + depth(ancestor)] = ancestor as u32;
        }
        if EXPANDED[slot] {
            let row = if entry == NONE { 0 } else { 1 + entry };
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


const DEPTHS: usize = 4;

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

fn weights(
    sizes: &[usize],
    salt: f32,
) -> Vec<Vec<f32>> {
    sizes.iter().enumerate().map(|(index, size)| values(*size, salt + index as f32, 0.6)).collect()
}

fn node_weights() -> Vec<Vec<f32>> {
    let (width, pool) = (WIDTH as usize, POOL_HIDDEN as usize);
    let mut weights = weights(&[3 * pool, pool, pool * pool, pool, width * 2 * pool, width, 4, 4], 0.7);
    weights[7] = vec![1.0, 1.5, 2.0, 2.5];
    weights
}

fn edge_weights() -> Vec<Vec<f32>> {
    let (width, hidden) = (WIDTH as usize, EDGE_HIDDEN as usize);
    let inputs = 3 * width + 4;
    weights(
        &[
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

fn depth_ids() -> Vec<u32> {
    (0..DEPTHS).flat_map(|depth| (0..CANDIDATES).map(move |index| pool_token(depth, index))).collect()
}

fn run_pools<B: Backend>() -> Vec<f32> {
    let context = create_context::<B>();
    let inputs = (
        create_buffer_with_data::<B, f32>(&context, &values(DEPTHS * CANDIDATES as usize, 0.3, 6.0)),
        create_buffer_with_data::<B, u32>(&context, &depth_ids()),
        create_buffer_with_data::<B, u64>(&context, &[0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 7, 9]),
    );
    let mut features = create_buffer_with_data::<B, f32>(&context, &vec![0.0; DEPTHS * 3 * CANDIDATES as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&inputs.0, &inputs.1, &inputs.2, &mut features, params(), DEPTHS as u32, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, f32>(&context, &features)
}

#[uzu_test]
fn pools_match_cpu() {
    let expected = run_pools::<Cpu>();
    for_each_non_cpu_backend!(|B| assert_close("pool features", &run_pools::<B>(), &expected, 1e-4));
}

fn run_nodes<B: Backend>() -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>) {
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
            features[depth * 3 * CANDIDATES as usize + index] = -features[depth * 3 * CANDIDATES as usize + index].abs();
        }
    }
    let u32s = |data: &[u32]| create_buffer_with_data::<B, u32>(&context, data);
    let f32s = |data: &[f32]| create_buffer_with_data::<B, f32>(&context, data);
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let inputs = (
        bf16_buffer(&values(SLOTS * CANDIDATES as usize, 1.7, 2.0)),
        f32s(&values(DEPTHS * CANDIDATES as usize, 0.3, 6.0)),
        u32s(&ids),
        f32s(&features),
        u32s(&metadata),
        u32s(&EXPANDED.map(u32::from)),
        u32s(&tree.packed_tree),
        u32s(&tree.frontier),
        bf16_buffer(&values(DEPTHS * WIDTH as usize, 2.3, 1.0)),
    );
    let weights = node_weights().iter().map(|values| bf16_buffer(values)).collect::<Vec<_>>();
    let mut outputs = (
        f32s(&vec![0.0; SLOTS * 2 * CANDIDATES as usize]),
        bf16_buffer(&vec![0.0; SLOTS * WIDTH as usize]),
        bf16_buffer(&vec![0.0; SLOTS * WIDTH as usize]),
        u32s(&[0; CAPACITY]),
        f32s(&vec![0.0; CAPACITY * 4]),
        u32s(&[NONE; SLOTS]),
        u32s(&[NONE; ROWS]),
    );
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadNodesKernel::new(&context, WIDTH, POOL_HIDDEN).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &inputs.0,
        &inputs.1,
        &inputs.2,
        &inputs.3,
        &inputs.4,
        &inputs.5,
        &inputs.6,
        &inputs.7,
        &inputs.8,
        &weights[0],
        &weights[1],
        &weights[2],
        &weights[3],
        &weights[4],
        &weights[5],
        &weights[6],
        &weights[7],
        &mut outputs.0,
        &mut outputs.1,
        &mut outputs.2,
        &mut outputs.3,
        &mut outputs.4,
        &mut outputs.5,
        &mut outputs.6,
        params(),
        SLOTS as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (
        buffer_to_vec::<B, f32>(&context, &outputs.0),
        floats(&buffer_to_vec::<B, bf16>(&context, &outputs.1)),
        buffer_to_vec::<B, u32>(&context, &outputs.3),
        buffer_to_vec::<B, f32>(&context, &outputs.4),
        buffer_to_vec::<B, u32>(&context, &outputs.5),
        buffer_to_vec::<B, u32>(&context, &outputs.6),
    )
}

#[uzu_test]
fn nodes_match_cpu() {
    let expected = run_nodes::<Cpu>();
    let tree = tree();
    assert_eq!(expected.4, tree.slot_rows);
    assert_eq!(expected.5, tree.row_slots);
    for_each_non_cpu_backend!(|B| {
        let actual = run_nodes::<B>();
        assert_close("scores", &actual.0, &expected.0, 1e-4);
        assert_close("nodes", &actual.1, &expected.1, 2e-2);
        assert_eq!(actual.2, expected.2);
        assert_close("child numbers", &actual.3, &expected.3, 1e-4);
        assert_eq!(actual.4, expected.4);
        assert_eq!(actual.5, expected.5);
    });
}

fn frontier_floats(
    frontier: &[u32],
    field: FrontierIdx,
) -> Vec<f32> {
    frontier[field as usize * CAPACITY..(field as usize + 1) * CAPACITY].iter().map(|bits| f32::from_bits(*bits)).collect()
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
    let pools = (0..CAPACITY).map(|entry| CHILD_INDEX[entry % CHILDREN] + (entry / CHILDREN) as u32).collect::<Vec<_>>();
    let u32s = |data: &[u32]| create_buffer_with_data::<B, u32>(&context, data);
    let f32s = |data: &[f32]| create_buffer_with_data::<B, f32>(&context, data);
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let inputs = (
        u32s(&metadata),
        u32s(&tree.slot_rows),
        bf16_buffer(&values(SLOTS * WIDTH as usize, 1.1, 1.0)),
        bf16_buffer(&values(ROWS * WIDTH as usize, 1.3, 0.5)),
        bf16_buffer(&values(ROWS * WIDTH as usize, 1.9, 0.5)),
        bf16_buffer(&values(VOCAB as usize * WIDTH as usize, 3.1, 1.0)),
        f32s(&values(CAPACITY * 4, 2.7, 1.0)),
        u32s(&pools),
        f32s(&values(SLOTS * 2 * CANDIDATES as usize, 0.5, 4.0)),
    );
    let weights = edge_weights().iter().map(|values| bf16_buffer(values)).collect::<Vec<_>>();
    let mut frontier = u32s(&tree.frontier);
    let mut packed_tree = u32s(&tree.packed_tree);
    let mut child_inputs = bf16_buffer(&vec![0.0; CAPACITY * (WIDTH as usize + 4)]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel::new(&context, WIDTH, EDGE_HIDDEN, refine).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        (!refine).then_some(&inputs.0),
        &inputs.1,
        &inputs.2,
        refine.then_some(&inputs.3),
        refine.then_some(&inputs.4),
        &inputs.5,
        &inputs.6,
        &inputs.7,
        &inputs.8,
        &mut frontier,
        &mut packed_tree,
        (!refine).then_some(&mut child_inputs),
        &weights[0],
        &weights[1],
        &weights[2],
        &weights[3],
        &weights[4],
        &weights[5],
        &weights[6],
        &weights[7],
        &weights[8],
        &weights[9],
        &weights[10],
        &weights[11],
        &weights[12],
        &weights[13],
        &weights[14],
        &weights[15],
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
        let expected = run_edges::<Cpu>(refine);
        let edges = frontier_floats(&expected.0, FrontierIdx::EdgeLogprobBits);
        assert!(edges[..8].iter().chain(&edges[12..]).all(|edge| *edge < 0.0));
        if refine {
            assert!(packed_edges(&expected.1)[1..].iter().all(|edge| *edge < 0.0));
        }
        for_each_non_cpu_backend!(|B| {
            let actual = run_edges::<B>(refine);
            for field in [FrontierIdx::EdgeLogprobBits, FrontierIdx::PathLogprobBits] {
                assert_close("frontier", &frontier_floats(&actual.0, field), &frontier_floats(&expected.0, field), 1e-3);
            }
            assert_close("tree edges", &packed_edges(&actual.1), &packed_edges(&expected.1), 1e-3);
            assert_close("child inputs", &actual.2, &expected.2, 2e-2);
        });
    }
}

fn run_tree_input<B: Backend>() -> Vec<f32> {
    let context = create_context::<B>();
    let tree = tree();
    let width = 32u32;
    let bf16_buffer = |data: &[f32]| create_buffer_with_data::<B, bf16>(&context, &bf16s(data));
    let inputs = (
        bf16_buffer(&values((STRIDE + 1) * width as usize, 0.2, 1.0)),
        bf16_buffer(&values(CAPACITY * width as usize, 0.4, 1.0)),
        bf16_buffer(&values(SLOTS * width as usize, 0.6, 1.0)),
        create_buffer_with_data::<B, u32>(&context, &tree.frontier),
        create_buffer_with_data::<B, u32>(&context, &tree.row_slots),
    );
    let mut output = bf16_buffer(&vec![0.0; ROWS * width as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeInputKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &inputs.0,
        &inputs.1,
        &inputs.2,
        &inputs.3,
        &inputs.4,
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
    let inputs = (
        create_buffer_with_data::<B, u32>(&context, &tree.frontier),
        create_buffer_with_data::<B, u32>(&context, &tree.slot_ancestors),
        create_buffer_with_data::<B, u32>(&context, &tree.slot_rows),
    );
    let mut relations = create_buffer_with_data::<B, u32>(&context, &[0; ROWS * ROWS]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeRelationsKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&inputs.0, &inputs.1, &inputs.2, &mut relations, ROWS as u32, STRIDE as u32, &mut command_buffer);
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
    let relations = (0..ROWS * ROWS)
        .map(|index| match (index / ROWS, index % ROWS) {
            (row, _) | (_, row) if (9..13).contains(&row) => 7,
            (query, key) if query == key => 1,
            _ => (index as u32 * 5 + (index / ROWS) as u32) % 7,
        })
        .collect::<Vec<_>>();
    let inputs = (
        create_buffer_with_data::<B, bf16>(&context, &bf16s(&values(ROWS * 3 * (heads * head_dim) as usize, 0.6, 2.0))),
        create_buffer_with_data::<B, u32>(&context, &relations),
        create_buffer_with_data::<B, bf16>(&context, &bf16s(&values(12, 2.2, 3.0))),
    );
    let mut output = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; ROWS * (heads * head_dim) as usize]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel::new(&context, heads, head_dim).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&inputs.0, &inputs.1, &inputs.2, &mut output, ROWS as u32, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    floats(&buffer_to_vec::<B, bf16>(&context, &output))
}

#[uzu_test]
fn tree_attention_matches_cpu() {
    let expected = run_tree_attention::<Cpu>();
    assert!(expected[9 * 64..13 * 64].iter().all(|value| *value == 0.0));
    for_each_non_cpu_backend!(|B| assert_close("tree_attention", &run_tree_attention::<B>(), &expected, 1e-2));
}
