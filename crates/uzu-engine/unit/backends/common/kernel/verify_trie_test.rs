use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
            GlobalBufferCpuGpu, Kernels, gpu_types::trie::TrieNode as GpuTrieNode, kernel::VerifyTrieKernel,
        },
        cpu::Cpu,
    },
    tests::helpers::{
        buffer_readback, buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend,
    },
};

struct Verification {
    accepted_indices: Vec<u32>,
    accepted_input_tokens: Vec<u32>,
    accepted_output_tokens: Vec<u32>,
    accepted_output_norm: Vec<u8>,
}

fn flat_trie(rows: &[(u32, u32, u32)]) -> (Vec<GpuTrieNode>, Vec<u32>, Vec<i32>) {
    let nodes: Vec<GpuTrieNode> = rows
        .iter()
        .enumerate()
        .map(|(index, &(_, height, end))| GpuTrieNode {
            trie_start: index as u32,
            trie_end: end,
            height,
        })
        .collect();
    let token_ids: Vec<u32> = rows.iter().map(|&(token, _, _)| token).collect();

    let mut parents: Vec<i32> = Vec::with_capacity(rows.len());
    // pre-order: the parent of a row is the nearest previous row one level up
    let mut path: Vec<usize> = Vec::new();
    for (index, &(_, height, _)) in rows.iter().enumerate() {
        path.truncate(height as usize);
        parents.push(path.last().map_or(-1, |&parent| parent as i32));
        path.push(index);
    }

    (nodes, token_ids, parents)
}

fn run_verify_trie<B: Backend>(
    nodes: &[GpuTrieNode],
    parents: &[i32],
    token_ids: &[u32],
    sampled_tokens: &[u32],
    norm_row_bytes: usize,
) -> Verification {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::VerifyTrieKernel::new(&context).expect("create verify trie kernel");

    let num_nodes = nodes.len();
    let max_accepted = nodes.iter().map(|node| node.height).max().unwrap_or_default() as usize + 1;
    let capacity = max_accepted + 1;

    // each norm row is recognizable: row `n` starts at byte `n * 0x10`
    let norm_rows: Vec<u8> =
        (0..num_nodes).flat_map(|node| (0..norm_row_bytes).map(move |byte| (node * 0x10 + byte) as u8)).collect();

    let mut nodes_staging =
        context.create_buffer_cpu_gpu(nodes.len() * std::mem::size_of::<GpuTrieNode>()).expect("stage trie nodes");
    nodes_staging.copyin(nodes);
    let nodes_buffer = nodes_staging.handoff();

    let parents_buffer = create_buffer_with_data::<B, i32>(&context, parents);
    let token_ids_buffer = create_buffer_with_data::<B, u32>(&context, token_ids);
    let sampled_tokens_buffer = create_buffer_with_data::<B, u32>(&context, sampled_tokens);
    let norm_buffer = create_buffer_with_data::<B, u8>(&context, &norm_rows);

    let mut accepted_indices = create_buffer::<B, u32>(&context, capacity);
    let mut accepted_input_tokens = create_buffer::<B, u32>(&context, capacity);
    let mut accepted_output_tokens = create_buffer::<B, u32>(&context, capacity);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut accepted_output_norm = command_buffer.allocate_scratch(norm_row_bytes).expect("norm scratch");
    kernel.encode(
        &nodes_buffer,
        &parents_buffer,
        &token_ids_buffer,
        &sampled_tokens_buffer,
        &norm_buffer,
        &mut accepted_indices,
        &mut accepted_input_tokens,
        &mut accepted_output_tokens,
        &mut accepted_output_norm,
        num_nodes as u32,
        capacity as u32,
        norm_row_bytes as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("submit");

    let accepted_output_norm = buffer_readback::<B>(&context, &accepted_output_norm);

    // the buffers are allocated to capacity; only the first `len + 1` words
    // are meaningful
    let truncate = |mut values: Vec<u32>| {
        values.truncate(values[0] as usize + 1);
        values
    };

    Verification {
        accepted_indices: truncate(buffer_to_vec::<B, u32>(&context, &accepted_indices)),
        accepted_input_tokens: truncate(buffer_to_vec::<B, u32>(&context, &accepted_input_tokens)),
        accepted_output_tokens: truncate(buffer_to_vec::<B, u32>(&context, &accepted_output_tokens)),
        accepted_output_norm: buffer_to_vec::<B, u8>(&context, &accepted_output_norm),
    }
}

fn expected_verification(
    rows: &[(u32, u32, u32)],
    sampled_tokens: &[u32],
    norm_row_bytes: usize,
) -> Verification {
    let (nodes, token_ids, _) = flat_trie(rows);

    // walk root-first, descending to the child whose token matches the sample
    let mut accepted = Vec::new();
    let mut current = 0;
    loop {
        accepted.push((current, token_ids[current] as u64, sampled_tokens[current] as u64));
        let mut child = current + 1;
        let mut descended = false;
        while child <= nodes[current].trie_end as usize {
            if token_ids[child] == sampled_tokens[current] {
                current = child;
                descended = true;
                break;
            }
            child = nodes[child].trie_end as usize + 1;
        }
        if !descended {
            break;
        }
    }

    let len = accepted.len() as u32;
    let reversed = |mut entries: Vec<u32>| {
        let mut buffer = vec![len];
        buffer.append(&mut entries);
        buffer[1..].reverse();
        buffer
    };

    let last = accepted.last().unwrap().0;

    Verification {
        accepted_indices: reversed(accepted.iter().map(|(index, _, _)| *index as u32).collect()),
        accepted_input_tokens: reversed(accepted.iter().map(|(_, input, _)| *input as u32).collect()),
        accepted_output_tokens: reversed(accepted.iter().map(|(_, _, output)| *output as u32).collect()),
        accepted_output_norm: (0..norm_row_bytes).map(|byte| (last * 0x10 + byte) as u8).collect(),
    }
}

fn assert_verification(
    rows: &[(u32, u32, u32)],
    sampled_tokens: &[u32],
    norm_row_bytes: usize,
) {
    let expected = expected_verification(rows, sampled_tokens, norm_row_bytes);
    let (nodes, token_ids, parents) = flat_trie(rows);

    let actual = run_verify_trie::<Cpu>(&nodes, &parents, &token_ids, sampled_tokens, norm_row_bytes);
    assert_eq!(actual.accepted_indices, expected.accepted_indices, "cpu accepted indices");
    assert_eq!(actual.accepted_input_tokens, expected.accepted_input_tokens, "cpu accepted input tokens");
    assert_eq!(actual.accepted_output_tokens, expected.accepted_output_tokens, "cpu accepted output tokens");
    assert_eq!(actual.accepted_output_norm, expected.accepted_output_norm, "cpu accepted output norm");

    for_each_non_cpu_backend!(|B| {
        let actual = run_verify_trie::<B>(&nodes, &parents, &token_ids, sampled_tokens, norm_row_bytes);
        assert_eq!(actual.accepted_indices, expected.accepted_indices, "accepted indices");
        assert_eq!(actual.accepted_input_tokens, expected.accepted_input_tokens, "accepted input tokens");
        assert_eq!(actual.accepted_output_tokens, expected.accepted_output_tokens, "accepted output tokens");
        assert_eq!(actual.accepted_output_norm, expected.accepted_output_norm, "accepted output norm");
    });
}

#[uzu_test]
fn verify_trie_accepts_full_chain() {
    // root(0) -> 1 -> 2 -> 3, sampled tokens walk the whole chain
    let rows = &[(0, 0, 4), (1, 1, 3), (2, 2, 3), (3, 3, 3), (4, 1, 4)];

    // the root samples 1, node 1 samples 2, node 2 samples 3
    assert_verification(rows, &[1, 2, 3, 5, 6], 7);
}

#[uzu_test]
fn verify_trie_stops_at_mismatch() {
    // root(0) with children 1 (-> 2) and 3
    let rows = &[(0, 0, 3), (1, 1, 2), (2, 2, 2), (3, 1, 3)];

    // the root samples 3 (the second child), node 3 has no child matching its
    // sampled token
    assert_verification(rows, &[3, 9, 9, 9], 3);
}

#[uzu_test]
fn verify_trie_accepts_root_only() {
    // root(0) with a single child 1
    let rows = &[(0, 0, 1), (1, 1, 1)];

    // the root samples a token that matches no child
    assert_verification(rows, &[7, 7], 5);
}

#[uzu_test]
fn verify_trie_skips_sibling_subtree() {
    // root with two children, each with own children; the walk must skip the
    // first child's subtree to reach the second child
    let rows = &[(0, 0, 5), (1, 1, 3), (2, 2, 2), (3, 2, 3), (4, 1, 5), (5, 2, 5)];

    // the root samples 4, node 4 samples 5
    assert_verification(rows, &[4, 9, 9, 9, 5, 9], 2);
}
