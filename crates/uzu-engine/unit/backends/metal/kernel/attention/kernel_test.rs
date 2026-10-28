use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::super::FullPrecisionInput;
use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareInt8Kernel, Int8Input, KVFormat,
                SigmoidGateKernel,
            },
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::{
        mixer::attention::{KVCacheView, kv_storage::KVStorage},
        sampling::PRng,
    },
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
    let (head_dim, num_q_heads, num_groups, _, prefix_length, is_causal) = shape;
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
    let trie = trie_nodes.map(|nodes| create_buffer_with_data::<B, u32>(context, bytemuck::cast_slice(nodes)));
    let input = AttentionInput::FullPrecision(FullPrecisionInput {
        queries: &queries,
        keys: &keys,
        values: &values,
    });
    let arguments = AttentionArguments {
        suffix_length: suffix_length as u32,
        trie: trie.as_ref(),
        sinks: sinks.as_ref(),
        cache,
        gate: None::<&B::GlobalGpuBuffer>,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let pooled = kernel.encode(input, arguments, &mut command_buffer).expect("encode");
    let mut output = create_buffer::<B, bf16>(context, suffix_length * num_q_heads * head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, bf16>(context, &output)
}

#[uzu_test]
fn attention_kernel_matches_cpu() {
    const TRIE_SUFFIX_LENGTH: usize = 31;
    let tokens: Vec<u64> = (0..TRIE_SUFFIX_LENGTH as u64).collect();
    let trie: Vec<GpuTrieNode> = TrieNode::flat(0, &tokens, &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let cpu_context = block_on(<Cpu as Backend>::Context::new()).expect("CPU attention context");
    let metal_context = block_on(<Metal as Backend>::Context::new()).expect("Metal attention context");
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
        (64, 4, 4, 8, 0, false, false),
        (64, 8, 2, 8, 0, false, false),
        (64, 8, 8, 8, 1024, false, false),
        (64, 8, 2, 8, 1024, false, false),
        (256, 24, 4, 1, 16_383, true, false),
        (256, 24, 4, 2, 16_383, true, false),
        (256, 24, 4, 4, 16_380, true, false),
        (128, 32, 8, 1, 16_383, true, false),
        (256, 6, 1, 512, 60_928, true, false),
        (256, 1, 1, 1024, 60_417, true, false),
        (128, 4, 1, 512, 30_208, true, false),
        (128, 4, 1, 1024, 29_696, true, false),
        (128, 8, 2, 512, 30_208, true, false),
    ] {
        let nodes = use_trie.then_some(trie.as_slice());
        let shape = (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal);
        let expected = run_attention::<Cpu>(cpu_context.as_ref(), shape, nodes, None);
        let actual = run_attention::<Metal>(metal_context.as_ref(), shape, nodes, None);
        let label = format!("attention kernel D{head_dim} S{suffix_length} L{prefix_length} causal={causal}");
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, &label);
    }
    for shape in [(512, 8, 2, 1, 32, false), (512, 8, 2, 1, 2048, false), (64, 4, 4, 64, 0, false)] {
        let sinks = vec![bf16::ONE; shape.1];
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
    let context = block_on(<Metal as Backend>::Context::new()).expect("Metal attention context");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(
        context.as_ref(),
        super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5),
    )
    .expect("attention kernel");
    let (keys, values) = attention_data(shape);
    let cache = KVCacheView::full(1024);
    let cpu = block_on(<Cpu as Backend>::Context::new()).expect("CPU attention context");
    for nodes in [None, Some(trie.as_slice())] {
        let expected = run_attention::<Cpu>(&cpu, shape, nodes, None);
        let actual = run_attention_with_kernel::<Metal>(&context, &kernel, shape, &keys, &values, cache, nodes, None);
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, "reused flat/trie attention kernel");
    }
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
    let rows = (0..CAPACITY)
        .map(|index| (CONTEXT - CAPACITY + index, (ring_offset + index) % CAPACITY))
        .chain((0..SUFFIX).map(|index| (CONTEXT + index, CAPACITY + index)));
    for (source, destination) in rows {
        for (full, ring) in [(&full_keys, &mut ring_keys), (&full_values, &mut ring_values)] {
            ring[destination * row..(destination + 1) * row].copy_from_slice(&full[source * row..(source + 1) * row]);
        }
    }

    let context = block_on(<Metal as Backend>::Context::new()).expect("Metal attention context");
    let mut config = super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5);
    config.sliding_window_size = Some(CAPACITY as u32);
    let [full, ring] = [
        (&full_keys, &full_values, KVCacheView::full(CONTEXT as u32)),
        (&ring_keys, &ring_values, KVCacheView::ring(CAPACITY as u32, ring_offset as u32)),
    ]
    .map(|(keys, values, cache)| {
        config.is_kv_cache_ring = cache.ring_params().is_some();
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&context, config).unwrap();
        run_attention_with_kernel::<Metal>(&context, &kernel, shape, keys, values, cache, None, None)
    });
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
    let input = AttentionInput::FullPrecision(FullPrecisionInput {
        queries: &queries,
        keys: &keys,
        values: &values,
    });
    let arguments = |gate| AttentionArguments {
        suffix_length: suffix_length as u32,
        trie: None::<&B::GlobalGpuBuffer>,
        sinks: None,
        cache: KVCacheView::full(prefix_length as u32),
        gate,
    };
    let gate_segment = qkvg.subrange((PROJECTION_DIM - GATE_DIM) * size_of::<bf16>()..);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut expected = plain.encode(input, arguments(None), &mut command_buffer).expect("plain attention");
    sigmoid_gate.encode(
        gate_segment,
        &mut expected,
        GATE_DIM as u32,
        suffix_length as u32,
        PROJECTION_DIM as u32,
        &mut command_buffer,
    );
    let fused = gated.encode(input, arguments(Some(gate_segment)), &mut command_buffer).expect("gated attention");
    let mut expected_out = create_buffer::<B, bf16>(context, suffix_length * GATE_DIM);
    let mut fused_out = create_buffer::<B, bf16>(context, suffix_length * GATE_DIM);
    command_buffer.encode_copy(&expected, &mut expected_out);
    command_buffer.encode_copy(&fused, &mut fused_out);
    submit_command_buffer(command_buffer);
    let expected = buffer_to_vec::<B, bf16>(context, &expected_out);
    let fused = buffer_to_vec::<B, bf16>(context, &fused_out);
    assert_eq!(expected, fused, "gated attention differs at S{suffix_length} L{prefix_length}");
}

#[uzu_test]
fn attention_gate_matches_standalone_gate_on_cpu() {
    let context = block_on(<Cpu as Backend>::Context::new()).expect("CPU attention context");
    check_attention_gate::<Cpu>(context.as_ref(), 3, 29);
}

#[uzu_test]
fn attention_gate_matches_standalone_gate_on_metal() {
    let context = block_on(<Metal as Backend>::Context::new()).expect("Metal attention context");
    for (suffix_length, prefix_length) in
        [(1, 100), (1, 2000), (200, 800), (2, 1022), (16, 16368), (65, 959), (65, 16319), (1024, 0), (512, 60_928)]
    {
        check_attention_gate::<Metal>(context.as_ref(), suffix_length, prefix_length);
    }
}

#[uzu_test]
fn int8_attention_matches_cpu() {
    let (metal, cpu) =
        (block_on(<Metal as Backend>::Context::new()).unwrap(), block_on(<Cpu as Backend>::Context::new()).unwrap());
    type Prepare = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareInt8Kernel;
    let trie: [GpuTrieNode; 3] = bytemuck::cast([0u32, 2, 0, 1, 1, 1, 2, 2, 1]); // (start, end, height) per token
    // (query heads, suffix, prefix, trie, shared KV, unwritten scale bits)
    const NAN: u16 = bf16::NAN.to_bits();
    const DENORMAL: u16 = 1; // 1 / scale overflows to inf
    for (q_heads, suffix, prefix, use_trie, reader, poison) in [
        (24, 1, 1024, false, false, NAN),
        (16, 16, 1024, false, false, DENORMAL),
        (24, 80, 16, false, false, NAN),
        (24, 3, 16, true, true, NAN),
        (24, 8, 32, false, false, DENORMAL),
        (16, 3, 37, true, false, DENORMAL),
        (32, 1, 1024, false, false, NAN),
        (32, 1024, 24, false, false, bf16::MAX.to_bits()),
        (32, 3, 1024, true, false, bf16::INFINITY.to_bits()),
    ] {
        let (head_dim, kv_heads) = if q_heads == 32 {
            (128, 2)
        } else {
            (256, 4)
        };
        let (shape, nodes) = ((head_dim, q_heads, kv_heads, suffix, prefix, true), use_trie.then_some(trie.as_slice()));
        let config = super::default_attention_config::<bf16>(head_dim, q_heads, kv_heads, true);
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&metal, config).unwrap();
        if !kernel.supports_int8() {
            return;
        }
        // Fused QKV rows over the data `run_attention` uses; its queries are head-major.
        let ((keys, values), stride) = (attention_data(shape), (q_heads + 2 * kv_heads) * head_dim);
        let mut rows = vec![bf16::ZERO; (prefix + suffix) * stride];
        for (row, (k, v)) in
            rows.chunks_mut(stride).zip(keys.chunks(kv_heads * head_dim).zip(values.chunks(kv_heads * head_dim)))
        {
            let (key_row, value_row) = row[q_heads * head_dim..].split_at_mut(kv_heads * head_dim);
            key_row.copy_from_slice(k);
            value_row.copy_from_slice(v);
        }
        for (index, query) in fill_attention(q_heads * suffix * head_dim, 0.5).chunks(head_dim).enumerate() {
            rows[(prefix + index % suffix) * stride + index / suffix * head_dim..][..head_dim].copy_from_slice(query);
        }
        let mut cache = KVStorage::<Metal>::create_empty(
            KVFormat::INT8,
            kv_heads as u32,
            head_dim as u32,
            2048,
            DataType::BF16,
            &metal,
        )
        .unwrap();
        cache.prepare(2048).unwrap();
        let KVStorage::Int8(kv, _) = &mut cache else {
            unreachable!()
        };
        let mut fill = metal.create_command_buffer(None, None).unwrap();
        let pattern =
            create_buffer_with_data::<Metal, u16>(&metal, &vec![poison; kv.k_scales.size() / size_of::<bf16>()]);
        for buffer in [&mut kv.k_codes, &mut kv.v_codes] {
            fill.encode_fill(buffer.as_mut(), poison as u8);
        }
        for buffer in [&mut kv.k_scales, &mut kv.v_scales] {
            fill.encode_copy(&pattern, buffer.as_mut());
        }
        submit_command_buffer(fill);
        let mut step = |rows: &[bf16], prefix: usize, trie: Option<&[GpuTrieNode]>, append_kv: bool| {
            let suffix = rows.len() / stride;
            let query_rows = q_heads / kv_heads * suffix;
            let prepare = Prepare::new(
                &metal,
                DataType::BF16,
                DataType::F32,
                head_dim as u32,
                false,
                true,
                kv_heads as u32,
                append_kv,
            )
            .unwrap();
            let mut query_codes = create_buffer::<Metal, i8>(&metal, kv_heads * query_rows * head_dim);
            let mut query_scales = create_buffer::<Metal, bf16>(&metal, kv_heads * query_rows);
            let mut command_buffer = metal.create_command_buffer(None, None).unwrap();
            let trie = trie.map(|nodes| command_buffer.allocate_constant_from_slice(nodes).unwrap());
            prepare.encode(
                &create_buffer_with_data::<Metal, bf16>(&metal, rows),
                None::<&mut <Metal as Backend>::GlobalGpuBuffer>,
                Some(&mut query_codes),
                Some(&mut query_scales),
                append_kv.then_some(kv.k_codes.as_mut()),
                append_kv.then_some(kv.k_scales.as_mut()),
                append_kv.then_some(kv.v_codes.as_mut()),
                append_kv.then_some(kv.v_scales.as_mut()),
                None::<&<Metal as Backend>::GlobalGpuBuffer>,
                None::<&<Metal as Backend>::GlobalGpuBuffer>,
                q_heads as u32,
                None,
                prefix as u32,
                stride as u32,
                suffix as u32,
                query_rows as u32,
                &mut command_buffer,
            );
            let input = AttentionInput::Int8(Int8Input {
                query_codes: &query_codes,
                query_scales: &query_scales,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            });
            let arguments = AttentionArguments {
                sinks: None,
                suffix_length: suffix as u32,
                trie: trie.as_ref(),
                cache: KVCacheView::full(prefix as u32),
                gate: None::<&<Metal as Backend>::GlobalGpuBuffer>,
            };
            let pooled = kernel.encode(input, arguments, &mut command_buffer).unwrap();
            let mut output = create_buffer::<Metal, bf16>(&metal, suffix * q_heads * head_dim);
            command_buffer.encode_copy(&pooled, &mut output);
            drop((pooled, trie));
            submit_command_buffer(command_buffer);
            buffer_to_vec::<Metal, bf16>(&metal, &output)
        };
        step(&rows[..prefix * stride], 0, None, true);
        let mut actual = step(&rows[prefix * stride..], prefix, nodes, true);
        if reader {
            // The reader's own K/V rows are junk: appending them would change the output.
            rows[prefix * stride..].chunks_mut(stride).for_each(|row| row[q_heads * head_dim..].fill(bf16::ONE));
            actual = step(&rows[prefix * stride..], prefix, nodes, false);
        }
        let label = format!("int8 attention q{q_heads} S{suffix} L{prefix} trie={use_trie} reader={reader}");
        assert_eq_float::<bf16>(&run_attention::<Cpu>(cpu.as_ref(), shape, nodes, None), &actual, 1e-2, &label);
    }
}
