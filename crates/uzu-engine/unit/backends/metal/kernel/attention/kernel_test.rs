use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{gemm, single_pass, two_pass};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{AttentionArguments, AttentionKernel, SigmoidGateKernel},
        },
        cpu::Cpu,
        metal::Metal,
    },
    encodable_block::{mixer::attention::KVCacheView, sampling::PRng},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, submit_command_buffer},
    },
    trie::TrieNode,
};

type AttentionShape = (usize, usize, usize, usize, usize, bool);

fn attention_data(shape: AttentionShape) -> (Vec<bf16>, Vec<bf16>) {
    let (head_dim, _, num_groups, suffix_length, prefix_length, _) = shape;
    let row = num_groups * head_dim;
    let count = (prefix_length + suffix_length) * row;
    (fill_attention(count, 1.0), fill_attention(count, 2.0))
}

fn fill_attention(
    count: usize,
    phase: f32,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32(((i as f32) * 0.017 + phase).sin() * 0.5)).collect()
}

fn run_attention<B: Backend>(
    context: &B::Context,
    shape: AttentionShape,
    trie_nodes: Option<&[GpuTrieNode]>,
    sinks_data: Option<&[bf16]>,
) -> Vec<bf16> {
    let (keys, values) = attention_data(shape);
    let (_, _, _, _, prefix_length, _) = shape;
    let (head_dim, num_q_heads, num_groups, _, _, is_causal) = shape;
    let mut config = super::default_attention_config::<bf16>(head_dim, num_q_heads, num_groups, is_causal);
    config.has_sinks = sinks_data.is_some();
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context, config).expect("attention kernel");
    run_attention_with_kernel::<B>(
        context,
        &kernel,
        shape,
        &keys,
        &values,
        KVCacheView::full(prefix_length as u32),
        trie_nodes,
        sinks_data,
    )
}

fn run_attention_with_kernel<B: Backend>(
    context: &B::Context,
    kernel: &<B::Kernels as Kernels>::AttentionKernel,
    shape: AttentionShape,
    keys_data: &[bf16],
    values_data: &[bf16],
    cache: KVCacheView,
    trie_nodes: Option<&[GpuTrieNode]>,
    sinks_data: Option<&[bf16]>,
) -> Vec<bf16> {
    let (head_dim, num_q_heads, _, suffix_length, _, _) = shape;
    let queries =
        create_buffer_with_data::<B, bf16>(context, &fill_attention(num_q_heads * suffix_length * head_dim, 0.5));
    let keys = create_buffer_with_data::<B, bf16>(context, keys_data);
    let values = create_buffer_with_data::<B, bf16>(context, values_data);
    let sinks = sinks_data.map(|data| create_buffer_with_data::<B, bf16>(context, data));
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
        sinks: sinks.as_ref(),
        cache,
        gate: None::<&B::GlobalBuffer>,
        rope: None,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let pooled = kernel.encode(arguments, &mut command_buffer).expect("encode");
    let mut output = create_buffer::<B, bf16>(context, suffix_length * num_q_heads * head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    let completed = command_buffer.end_encoding().submit().wait_until_completed().expect("submit");
    drop(pooled);
    drop(completed);
    buffer_to_vec::<B, bf16>(&output)
}

enum DirectKernel {
    SinglePass,
    TwoPass,
    Gemm,
}

fn run_direct_metal(
    context: &<Metal as Backend>::Context,
    shape: AttentionShape,
    sinks_data: Option<&[bf16]>,
    route: DirectKernel,
) -> Vec<bf16> {
    let (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, is_causal) = shape;
    let mut config = super::default_attention_config::<bf16>(head_dim, num_q_heads, num_groups, is_causal);
    config.has_sinks = sinks_data.is_some();
    let (keys_data, values_data) = attention_data(shape);
    let queries =
        create_buffer_with_data::<Metal, bf16>(context, &fill_attention(num_q_heads * suffix_length * head_dim, 0.5));
    let keys = create_buffer_with_data::<Metal, bf16>(context, &keys_data);
    let values = create_buffer_with_data::<Metal, bf16>(context, &values_data);
    let sinks = sinks_data.map(|data| create_buffer_with_data::<Metal, bf16>(context, data));
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: None::<&<Metal as Backend>::GlobalBuffer>,
        sinks: sinks.as_ref(),
        cache: KVCacheView::full(prefix_length as u32),
        gate: None::<&<Metal as Backend>::GlobalBuffer>,
        rope: None,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let pooled = match route {
        DirectKernel::SinglePass => {
            single_pass::AttentionSinglePass::new(&config).encode(arguments, &mut command_buffer)
        },
        DirectKernel::TwoPass => two_pass::AttentionTwoPass::new(&config).encode(arguments, &mut command_buffer),
        DirectKernel::Gemm => gemm::AttentionGemm::new(&config).encode(arguments, &mut command_buffer),
    }
    .expect("direct attention");
    let mut output = create_buffer::<Metal, bf16>(context, suffix_length * num_q_heads * head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Metal, bf16>(&output)
}

#[uzu_test]
fn direct_bf16_attention_matches_cpu() {
    let cpu = <Cpu as Backend>::Context::new().expect("CPU context");
    let metal = <Metal as Backend>::Context::new().expect("Metal context");
    for (shape, route, has_sinks) in [
        ((64, 4, 4, 8, 0, false), DirectKernel::SinglePass, false),
        ((64, 4, 4, 64, 0, false), DirectKernel::SinglePass, true),
        ((64, 8, 2, 8, 0, false), DirectKernel::SinglePass, false),
        ((64, 8, 2, 256, 0, true), DirectKernel::SinglePass, false),
        ((64, 8, 2, 256, 0, true), DirectKernel::Gemm, false),
        ((64, 8, 8, 128, 0, false), DirectKernel::TwoPass, false),
        ((64, 8, 2, 64, 1024, false), DirectKernel::TwoPass, false),
    ] {
        let sinks = has_sinks.then(|| vec![bf16::from_f32(1.0); shape.1]);
        let expected = run_attention::<Cpu>(cpu.as_ref(), shape, None, sinks.as_deref());
        let actual = run_direct_metal(metal.as_ref(), shape, sinks.as_deref(), route);
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, "direct BF16 attention");
    }
}

#[uzu_test]
fn attention_kernel_matches_cpu() {
    const TRIE_SUFFIX_LENGTH: usize = 31;
    let tokens: Vec<u64> = (0..TRIE_SUFFIX_LENGTH as u64).collect();
    let trie: Vec<GpuTrieNode> = TrieNode::flat(0, &tokens, &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU attention context");
    let metal_context = <Metal as Backend>::Context::new().expect("Metal attention context");
    for &(head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal, use_trie) in &[
        (512, 8, 8, 9, 0, false, false),
        (128, 8, 2, 16, 1024, false, false),
        (256, 6, 1, 2, 1024, false, false),
        (256, 6, 1, 15, 2048, true, false),
        (256, 6, 1, 32, 1024, true, false),
        (256, 6, 1, TRIE_SUFFIX_LENGTH, 1024, true, true),
        (512, 8, 8, 1, 0, false, false),
        (512, 8, 8, 1, 1024, false, false),
        (64, 8, 8, 9, 0, false, false),
    ] {
        let nodes = use_trie.then_some(trie.as_slice());
        let shape = (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal);
        let expected = run_attention::<Cpu>(cpu_context.as_ref(), shape, nodes, None);
        let actual = run_attention::<Metal>(metal_context.as_ref(), shape, nodes, None);
        let label = format!("attention kernel D{head_dim} S{suffix_length} L{prefix_length} causal={causal}");
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, &label);
    }
    let sinks = vec![bf16::from_f32(1.0); 8];
    for prefix_length in [32, 2048] {
        let shape = (512, 8, 2, 1, prefix_length, false);
        let expected = run_attention::<Cpu>(cpu_context.as_ref(), shape, None, Some(&sinks));
        let actual = run_attention::<Metal>(metal_context.as_ref(), shape, None, Some(&sinks));
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, "BF16 GQA with sinks");
    }
}

#[uzu_test]
fn attention_kernel_reuses_instance_for_flat_and_trie() {
    let shape = (256, 6, 1, 5, 1024, true);
    let trie: Vec<GpuTrieNode> =
        TrieNode::flat(0, &[0, 1, 2, 3, 4], &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let context = <Metal as Backend>::Context::new().expect("Metal attention context");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(
        context.as_ref(),
        super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5),
    )
    .expect("attention kernel");
    let (keys, values) = attention_data(shape);
    let cache = KVCacheView::full(1024);
    let flat = run_attention_with_kernel::<Metal>(context.as_ref(), &kernel, shape, &keys, &values, cache, None, None);
    let trie_output =
        run_attention_with_kernel::<Metal>(context.as_ref(), &kernel, shape, &keys, &values, cache, Some(&trie), None);
    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU attention context");
    let cpu_kernel = <<Cpu as Backend>::Kernels as Kernels>::AttentionKernel::new(
        cpu_context.as_ref(),
        super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5),
    )
    .expect("CPU attention kernel");
    let expected_flat =
        run_attention_with_kernel::<Cpu>(cpu_context.as_ref(), &cpu_kernel, shape, &keys, &values, cache, None, None);
    let expected_trie = run_attention_with_kernel::<Cpu>(
        cpu_context.as_ref(),
        &cpu_kernel,
        shape,
        &keys,
        &values,
        cache,
        Some(&trie),
        None,
    );
    assert_eq_float::<bf16>(&expected_flat, &flat, 1e-2, "flat mask specialization");
    assert_eq_float::<bf16>(&expected_trie, &trie_output, 1e-2, "trie mask specialization");
}

#[uzu_test]
fn attention_kernel_ring_matches_full_on_wrap() {
    const HEAD_DIM: usize = 64;
    const NUM_HEADS: usize = 8;
    const CAPACITY: usize = 8;
    const CONTEXT: usize = 12;
    const SUFFIX: usize = 2;
    let shape = (HEAD_DIM, NUM_HEADS, NUM_HEADS, SUFFIX, CONTEXT, true);
    let row = NUM_HEADS * HEAD_DIM;
    let (full_keys, full_values) = attention_data(shape);
    let mut ring_keys = vec![bf16::ZERO; (CAPACITY + SUFFIX) * row];
    let mut ring_values = vec![bf16::ZERO; (CAPACITY + SUFFIX) * row];
    let ring_offset = (CONTEXT - CAPACITY) % CAPACITY;
    for index in 0..CAPACITY {
        let source = CONTEXT - CAPACITY + index;
        let destination = (ring_offset + index) % CAPACITY;
        ring_keys[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_keys[source * row..(source + 1) * row]);
        ring_values[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_values[source * row..(source + 1) * row]);
    }
    for index in 0..SUFFIX {
        let source = CONTEXT + index;
        let destination = CAPACITY + index;
        ring_keys[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_keys[source * row..(source + 1) * row]);
        ring_values[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_values[source * row..(source + 1) * row]);
    }

    let context = <Metal as Backend>::Context::new().expect("Metal attention context");
    let full_kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), {
        let mut config = super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5);
        config.sliding_window_size = Some(CAPACITY as u32);
        config
    })
    .expect("full attention kernel");
    let ring_kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), {
        let mut config = super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5);
        config.is_kv_cache_ring = true;
        config.sliding_window_size = Some(CAPACITY as u32);
        config
    })
    .expect("ring attention kernel");
    let full = run_attention_with_kernel::<Metal>(
        context.as_ref(),
        &full_kernel,
        shape,
        &full_keys,
        &full_values,
        KVCacheView::full(CONTEXT as u32),
        None,
        None,
    );
    let ring = run_attention_with_kernel::<Metal>(
        context.as_ref(),
        &ring_kernel,
        shape,
        &ring_keys,
        &ring_values,
        KVCacheView::ring(CAPACITY as u32, ring_offset as u32, CONTEXT as u32),
        None,
        None,
    );
    assert_eq_float::<bf16>(&full, &ring, 1e-2, "wrapped ring attention");
}

fn check_attention_gate<B: Backend>(
    context: &B::Context,
    suffix_length: usize,
    prefix_length: usize,
) {
    const HEAD_DIM: usize = 256;
    const Q_HEADS: usize = 24;
    const KV_HEADS: usize = 4;
    const GATE_DIM: usize = Q_HEADS * HEAD_DIM;
    const PROJECTION_DIM: usize = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM;
    let mut config = super::default_attention_config::<bf16>(HEAD_DIM, Q_HEADS, KV_HEADS, true);
    let plain = <B::Kernels as Kernels>::AttentionKernel::new(context, config).expect("attention kernel");
    config.gate_row_stride = Some(PROJECTION_DIM as u32);
    let gated = <B::Kernels as Kernels>::AttentionKernel::new(context, config).expect("gated attention kernel");
    let sigmoid_gate =
        <B::Kernels as Kernels>::SigmoidGateKernel::new(context, bf16::data_type()).expect("sigmoid gate kernel");
    let shape = (HEAD_DIM, Q_HEADS, KV_HEADS, suffix_length, prefix_length, true);
    let (keys, values) = attention_data(shape);
    let queries = create_buffer_with_data::<B, bf16>(context, &fill_attention(Q_HEADS * suffix_length * HEAD_DIM, 0.5));
    let keys = create_buffer_with_data::<B, bf16>(context, &keys);
    let values = create_buffer_with_data::<B, bf16>(context, &values);
    let qkvg = create_buffer_with_data::<B, bf16>(context, &fill_attention(suffix_length * PROJECTION_DIM, 3.0));
    let arguments = |gate| AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: None::<&B::GlobalBuffer>,
        sinks: None,
        cache: KVCacheView::full(prefix_length as u32),
        gate,
        rope: None,
    };
    let gate_segment = qkvg.subrange((PROJECTION_DIM - GATE_DIM) * size_of::<bf16>()..);
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
    let expected = buffer_to_vec::<B, bf16>(&expected_out);
    let fused = buffer_to_vec::<B, bf16>(&fused_out);
    assert_eq!(expected, fused, "gated attention differs at S{suffix_length} L{prefix_length}");
}

#[uzu_test]
fn attention_gate_matches_standalone_gate_on_cpu() {
    let context = <Cpu as Backend>::Context::new().expect("CPU attention context");
    check_attention_gate::<Cpu>(context.as_ref(), 3, 29);
}

#[uzu_test]
fn attention_gate_matches_standalone_gate_on_metal() {
    let context = <Metal as Backend>::Context::new().expect("Metal attention context");
    for (suffix_length, prefix_length) in
        [(1, 100), (1, 2000), (200, 800), (2, 1022), (16, 16368), (65, 959), (65, 16319), (1024, 0)]
    {
        check_attention_gate::<Metal>(context.as_ref(), suffix_length, prefix_length);
    }
}
