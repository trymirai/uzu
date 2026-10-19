#![cfg(backend = "metal")]

use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::trie::TrieNode, kernel::TreeVerifyFusedKernel,
        },
        cpu::Cpu,
        metal::Metal,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
    },
};

const HEAD_DIM: usize = 128;
const K_HEADS: usize = 2;
const VALUE_HEADS: usize = 6;
const CONV_DIM: usize = 2 * K_HEADS * HEAD_DIM + VALUE_HEADS * HEAD_DIM;
const TOTAL_PROJ_DIM: usize = CONV_DIM + VALUE_HEADS * HEAD_DIM + 2 * VALUE_HEADS;

struct Inputs {
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    trie: Vec<TrieNode>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    h0: Vec<f32>,
    in_proj: Vec<bf16>,
    norm_weight: Vec<f32>,
}

// A branching tree in DFS order: node i hangs off a node on the current root-to-leaf path.
fn dfs_tree(tree_size: usize) -> Vec<TrieNode> {
    let mut nodes = vec![
        TrieNode {
            trie_start: 0,
            trie_end: 0,
            height: 0,
        };
        tree_size
    ];
    let mut path = vec![0usize];
    for index in 1..tree_size {
        let depth = 1 + (index * 7 + index / 3) % path.len();
        for closed in path.drain(depth..) {
            nodes[closed].trie_end = index as u32 - 1;
        }
        nodes[index] = TrieNode {
            trie_start: index as u32,
            trie_end: index as u32,
            height: depth as u32,
        };
        path.push(index);
    }
    for open in path {
        nodes[open].trie_end = tree_size as u32 - 1;
    }
    nodes
}

// Unit-norm rows of HEAD_DIM, scaled.
fn unit_rows(
    rows: usize,
    scale: f32,
    phase: f32,
) -> Vec<bf16> {
    (0..rows)
        .flat_map(|row| {
            let values =
                (0..HEAD_DIM).map(|column| ((row * HEAD_DIM + column) as f32 * 0.37 + phase).sin()).collect::<Vec<_>>();
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            values.into_iter().map(move |value| bf16::from_f32(value / norm * scale))
        })
        .collect()
}

fn make_inputs(tree_size: usize) -> Inputs {
    Inputs {
        q: unit_rows(tree_size * K_HEADS, (HEAD_DIM as f32).sqrt().recip(), 0.1),
        k: unit_rows(tree_size * K_HEADS, 1.0, 0.7),
        v: (0..tree_size * VALUE_HEADS * HEAD_DIM).map(|i| bf16::from_f32((i as f32 * 0.013).cos() * 0.5)).collect(),
        trie: dfs_tree(tree_size),
        log_decay: (0..tree_size * VALUE_HEADS).map(|i| -0.02 - (i % 5) as f32 * 0.03).collect(),
        beta: (0..tree_size * VALUE_HEADS).map(|i| 0.2 + (i % 7) as f32 * 0.1).collect(),
        h0: (0..VALUE_HEADS * HEAD_DIM * HEAD_DIM).map(|i| (i as f32 * 0.019).cos() * 0.05).collect(),
        in_proj: (0..tree_size * TOTAL_PROJ_DIM).map(|i| bf16::from_f32((i as f32 * 0.011).sin())).collect(),
        norm_weight: (0..HEAD_DIM).map(|i| 0.5 + (i % 9) as f32 * 0.1).collect(),
    }
}

fn run<B: Backend>(
    inputs: &Inputs,
    max_tree: u32,
) -> Vec<bf16> {
    let tree_size = inputs.trie.len();
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::TreeVerifyFusedKernel::new(
        &context,
        crate::data_type::DataType::BF16,
        max_tree,
    )
    .expect("TreeVerifyFusedKernel");
    // The CPU backend runs the dispatch at submit, so every buffer must outlive the wait.
    let q = create_buffer_with_data::<B, bf16>(&context, &inputs.q);
    let k = create_buffer_with_data::<B, bf16>(&context, &inputs.k);
    let v = create_buffer_with_data::<B, bf16>(&context, &inputs.v);
    let trie = create_buffer_with_data::<B, u32>(
        &context,
        &inputs.trie.iter().flat_map(|node| [node.trie_start, node.trie_end, node.height]).collect::<Vec<_>>(),
    );
    let log_decay = create_buffer_with_data::<B, f32>(&context, &inputs.log_decay);
    let beta = create_buffer_with_data::<B, f32>(&context, &inputs.beta);
    let h0 = create_buffer_with_data::<B, f32>(&context, &inputs.h0);
    let in_proj = create_buffer_with_data::<B, bf16>(&context, &inputs.in_proj);
    let norm_weight = create_buffer_with_data::<B, f32>(&context, &inputs.norm_weight);
    let mut out = create_buffer::<B, bf16>(&context, tree_size * VALUE_HEADS * HEAD_DIM);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &q,
        &k,
        &v,
        &trie,
        &log_decay,
        &beta,
        &h0,
        &in_proj,
        &norm_weight,
        &mut out,
        tree_size as u32,
        K_HEADS as u32,
        VALUE_HEADS as u32,
        CONV_DIM as u32,
        TOTAL_PROJ_DIM as u32,
        1e-6,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec(&out)
}

#[uzu_test]
fn test_tree_verify_fused_matches_cpu() {
    for (tree_size, max_tree) in [(1, 16), (9, 16), (16, 16), (17, 32), (32, 32)] {
        let inputs = make_inputs(tree_size);
        let expected = run::<Cpu>(&inputs, max_tree);
        let actual = run::<Metal>(&inputs, max_tree);
        assert_eq_float::<bf16>(&expected, &actual, 3e-2, &format!("tree {tree_size} max {max_tree}"));
    }
}
