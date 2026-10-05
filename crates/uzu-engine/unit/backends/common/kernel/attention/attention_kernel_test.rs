use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::{mixer::attention::KVCacheView, sampling::PRng},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
    trie::TrieNode,
};

/// (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal)
type AttentionShape = (usize, usize, usize, usize, usize, bool);

fn fill(
    count: usize,
    phase: f32,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32(((i as f32) * 0.017 + phase).sin() * 0.5)).collect()
}

fn run_attention<B: Backend>(
    shape: AttentionShape,
    trie_nodes: Option<&[GpuTrieNode]>,
) -> Vec<bf16> {
    let (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, is_causal) = shape;
    let context = B::Context::new().expect("context");
    let config = AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: num_groups as u32,
        num_q_heads: num_q_heads as u32,
        has_sinks: false,
        is_kv_cache_ring: false,
        is_causal,
        sliding_window_size: None,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
        data_type: DataType::BF16,
    };
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), config).expect("attention kernel");
    let kv_count = (prefix_length + suffix_length) * num_groups * head_dim;
    let queries =
        create_buffer_with_data::<B, bf16>(context.as_ref(), &fill(num_q_heads * suffix_length * head_dim, 0.5));
    let keys = create_buffer_with_data::<B, bf16>(context.as_ref(), &fill(kv_count, 1.0));
    let values = create_buffer_with_data::<B, bf16>(context.as_ref(), &fill(kv_count, 2.0));
    let trie = trie_nodes.map(|nodes| {
        let words: Vec<u32> = nodes.iter().flat_map(|node| [node.trie_start, node.trie_end, node.height]).collect();
        create_buffer_with_data::<B, u32>(context.as_ref(), &words)
    });
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: trie.as_ref(),
        sinks: None,
        cache: KVCacheView::full(prefix_length as u32),
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let pooled = kernel.encode(arguments, &mut command_buffer).expect("encode");
    let mut output = create_buffer::<B, bf16>(context.as_ref(), suffix_length * num_q_heads * head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    let completed = command_buffer.end_encoding().submit().wait_until_completed().expect("submit");
    drop(pooled);
    drop(completed);
    buffer_to_vec::<B, bf16>(&output)
}

/// The backend's attention policy against the CPU reference, across the decode, verification and prefill
/// suffix lengths that pick its single-pass, two-pass and GEMM paths.
#[uzu_test]
fn attention_kernel_matches_cpu() {
    const TRIE_SUFFIX_LENGTH: usize = 31;
    let tokens: Vec<u64> = (0..TRIE_SUFFIX_LENGTH as u64).collect();
    let trie: Vec<GpuTrieNode> = TrieNode::flat(0, &tokens, &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    for &(head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal, use_trie) in &[
        (512, 8, 8, 9, 0, false, false),
        (128, 8, 2, 16, 1024, false, false),
        (256, 6, 1, 2, 1024, false, false),
        (256, 6, 1, 15, 2048, true, false),
        (256, 6, 1, 32, 1024, true, false),
        (256, 4, 1, 70, 37, true, false),
        (256, 6, 1, TRIE_SUFFIX_LENGTH, 1024, true, true),
        (512, 8, 8, 1, 0, false, false),
        (512, 8, 8, 1, 1024, false, false),
        (64, 8, 8, 9, 0, false, false),
    ] {
        let nodes = use_trie.then_some(trie.as_slice());
        let shape = (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal);
        let expected = run_attention::<Cpu>(shape, nodes);
        for_each_non_cpu_backend!(|B| {
            let actual = run_attention::<B>(shape, nodes);
            let label = format!("{} attention D{head_dim} S{suffix_length} L{prefix_length} causal={causal}", B::NAME);
            assert_eq_float::<bf16>(&expected, &actual, 1e-2, &label);
        });
    }
}
