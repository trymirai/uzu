use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig, SigmoidGateKernel},
        },
        cpu::Cpu,
        vulkan::Vulkan,
    },
    encodable_block::{mixer::attention::KVCacheView, sampling::PRng},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, submit_command_buffer},
    },
    trie::TrieNode,
};

// The Vulkan counterpart of Metal's attention kernel tests (unit/backends/metal/kernel/attention/kernel_test.rs): the
// routed AttentionKernel against the CPU's at Metal's tolerance, including the two-pass route of more than 1024 keys
// for at most 8 queries, its trie specialization, and the fused gate against the standalone SigmoidGate.

/// (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal)
type AttentionShape = (usize, usize, usize, usize, usize, bool);

fn config((head_dim, num_q_heads, num_groups, _, _, is_causal): AttentionShape) -> AttentionKernelConfig {
    AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: num_groups as u32,
        num_q_heads: num_q_heads as u32,
        has_sinks: false,
        gate_row_stride: None,
        is_kv_cache_ring: false,
        is_causal,
        sliding_window_size: None,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
        data_type: bf16::data_type(),
    }
}

fn fill(
    count: usize,
    phase: f32,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32(((i as f32) * 0.017 + phase).sin() * 0.5)).collect()
}

fn run<B: Backend>(
    context: &B::Context,
    shape: AttentionShape,
    trie_nodes: Option<&[GpuTrieNode]>,
) -> Vec<bf16> {
    let (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, _) = shape;
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context, config(shape)).expect("attention kernel");
    let rows = (prefix_length + suffix_length) * num_groups * head_dim;
    let queries = create_buffer_with_data::<B, bf16>(context, &fill(num_q_heads * suffix_length * head_dim, 0.5));
    let keys = create_buffer_with_data::<B, bf16>(context, &fill(rows, 1.0));
    let values = create_buffer_with_data::<B, bf16>(context, &fill(rows, 2.0));
    let trie = trie_nodes.map(|nodes| {
        let words: Vec<u32> = nodes.iter().flat_map(|node| [node.trie_start, node.trie_end, node.height]).collect();
        create_buffer_with_data::<B, u32>(context, &words)
    });
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: trie.as_ref(),
        sinks: None,
        cache: KVCacheView::full(prefix_length as u32),
        gate: None::<&B::GlobalBuffer>,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let pooled = kernel.encode(arguments, &mut command_buffer).expect("encode");
    let mut output = create_buffer::<B, bf16>(context, suffix_length * num_q_heads * head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, bf16>(&output)
}

/// Single-pass and two-pass routes, GQA, causal and not, a trie of 5 nodes after 1500 keys.
#[uzu_test]
fn attention_kernel_matches_cpu() {
    let trie: Vec<GpuTrieNode> =
        TrieNode::flat(0, &[0, 1, 2, 3, 4], &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let (cpu, vulkan) = (create_context::<Cpu>(), create_context::<Vulkan>());
    for &(shape, use_trie) in &[
        ((256, 8, 2, 1, 1000, true), false),
        ((256, 8, 2, 1, 1355, true), false),
        ((256, 8, 2, 8, 2040, true), false),
        ((256, 6, 1, 2, 1024, false), false),
        ((128, 8, 2, 3, 3000, true), false),
        ((64, 8, 8, 1, 1100, false), false),
        ((512, 8, 8, 1, 1024, false), false),
        ((256, 6, 1, 5, 1500, true), true),
        ((256, 8, 2, 9, 1100, true), false),
    ] {
        let nodes = use_trie.then_some(trie.as_slice());
        let expected = run::<Cpu>(&cpu, shape, nodes);
        let actual = run::<Vulkan>(&vulkan, shape, nodes);
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, &format!("attention kernel {shape:?} trie {use_trie}"));
    }
}

/// The routed attention with a gate equals the attention without one followed by the standalone SigmoidGate, bit
/// for bit, as Metal checks its fused gate.
fn check_gate<B: Backend>(
    context: &B::Context,
    suffix_length: usize,
    prefix_length: usize,
) {
    const HEAD_DIM: usize = 256;
    const Q_HEADS: usize = 8;
    const KV_HEADS: usize = 2;
    const GATE_DIM: usize = Q_HEADS * HEAD_DIM;
    const PROJECTION_DIM: usize = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM;
    let shape = (HEAD_DIM, Q_HEADS, KV_HEADS, suffix_length, prefix_length, true);
    let mut gated_config = config(shape);
    let plain = <B::Kernels as Kernels>::AttentionKernel::new(context, gated_config).expect("attention kernel");
    gated_config.gate_row_stride = Some(PROJECTION_DIM as u32);
    let gated = <B::Kernels as Kernels>::AttentionKernel::new(context, gated_config).expect("gated attention kernel");
    let sigmoid_gate =
        <B::Kernels as Kernels>::SigmoidGateKernel::new(context, bf16::data_type()).expect("sigmoid gate kernel");
    let rows = (prefix_length + suffix_length) * KV_HEADS * HEAD_DIM;
    let queries = create_buffer_with_data::<B, bf16>(context, &fill(Q_HEADS * suffix_length * HEAD_DIM, 0.5));
    let keys = create_buffer_with_data::<B, bf16>(context, &fill(rows, 1.0));
    let values = create_buffer_with_data::<B, bf16>(context, &fill(rows, 2.0));
    let qkvg = create_buffer_with_data::<B, bf16>(context, &fill(suffix_length * PROJECTION_DIM, 3.0));
    let gate_segment = qkvg.subrange((PROJECTION_DIM - GATE_DIM) * size_of::<bf16>()..);
    let arguments = |gate| AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: None::<&B::GlobalBuffer>,
        sinks: None,
        cache: KVCacheView::full(prefix_length as u32),
        gate,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut expected = plain.encode(arguments(None), &mut command_buffer).expect("plain attention");
    sigmoid_gate.encode(
        gate_segment,
        &mut expected,
        GATE_DIM as u32,
        suffix_length as u32,
        PROJECTION_DIM as u32,
        &mut command_buffer,
    );
    let fused = gated.encode(arguments(Some(gate_segment)), &mut command_buffer).expect("gated attention");
    let mut expected_out = create_buffer::<B, bf16>(context, suffix_length * GATE_DIM);
    let mut fused_out = create_buffer::<B, bf16>(context, suffix_length * GATE_DIM);
    command_buffer.encode_copy(&expected, &mut expected_out);
    command_buffer.encode_copy(&fused, &mut fused_out);
    submit_command_buffer(command_buffer);
    assert_eq!(
        buffer_to_vec::<B, bf16>(&expected_out),
        buffer_to_vec::<B, bf16>(&fused_out),
        "gated attention differs at S{suffix_length} L{prefix_length}"
    );
}

/// Single-pass and two-pass routes.
#[uzu_test]
fn attention_gate_matches_standalone_gate() {
    let context = create_context::<Vulkan>();
    for (suffix_length, prefix_length) in [(1, 100), (1, 2000), (3, 1500), (8, 1100), (9, 1100)] {
        check_gate::<Vulkan>(&context, suffix_length, prefix_length);
    }
}
