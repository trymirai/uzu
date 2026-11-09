use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareKernel},
        },
        metal::{Metal, kernel::attention::AttentionMetalKernel},
    },
    data_type::DataType,
    encodable_block::{
        mixer::attention::{KVCache, KVCacheView},
        sampling::PRng,
    },
    tests::{
        assert::assert_eq_float,
        attention::default_attention_config,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, submit_command_buffer},
    },
    trie::TrieNode,
};

type AttentionShape = (usize, usize, usize, usize, usize, bool);

fn flat_trie(
    prefix_length: usize,
    tokens: &[u64],
    prng: &PRng,
) -> TrieNode {
    assert!(!tokens.is_empty(), "need seed node");

    let mut node = TrieNode::new(tokens[0], prng.derive(prefix_length as u64), 0.0);
    for (index, token) in tokens.iter().copied().enumerate().skip(1).rev() {
        let mut parent = TrieNode::new(token, prng.derive((prefix_length + index) as u64), 0.0);
        parent.add(node).unwrap();
        node = parent;
    }
    node
}

pub(super) fn fill_attention(
    count: usize,
    phase: f32,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32(((i as f32) * 0.017 + phase).sin() * 0.5)).collect()
}

#[uzu_test]
fn attention_matches_cpu() {
    let metal = block_on(<Metal as Backend>::Context::new()).unwrap();
    type Prepare = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareKernel;
    const NAN: u16 = bf16::NAN.to_bits();
    const DENORMAL: u16 = 1; // 1 / scale overflows to inf
    for use_mxu in [false, true].into_iter().filter(|&use_mxu| !use_mxu || metal.supports_mxu) {
        // (query heads, suffix, prefix, trie, shared KV, unwritten scale bits)
        for (q_heads, suffix, prefix, use_trie, reader, poison) in [
            (8, 80, 16, false, false, NAN),
            (8, 24, 37, false, false, NAN),
            (8, 64, 0, true, false, NAN),
            (64, 80, 16, false, false, NAN),
            (24, 1, 1024, false, false, NAN),
            (16, 16, 1024, false, false, DENORMAL),
            (24, 80, 16, false, false, NAN),
            (24, 3, 16, true, true, NAN),
            (24, 8, 32, false, false, DENORMAL),
            (16, 3, 37, true, false, DENORMAL),
            (32, 1, 1024, false, false, NAN),
            (32, 1024, 24, false, false, bf16::MAX.to_bits()),
            (32, 3, 1024, true, false, bf16::INFINITY.to_bits()),
            (24, 8, 0, true, false, NAN),
            (24, 8, 4096, true, false, NAN),
            (24, 8, 32768, true, false, DENORMAL),
            (32, 8, 4097, true, false, NAN),
            (24, 63, 29, true, false, NAN),
            (24, 64, 0, true, false, NAN),
        ] {
            let (head_dim, kv_heads) = match q_heads {
                8 => (64, 2),
                32 => (128, 2),
                64 => (512, 4),
                _ => (256, 4),
            };
            let tokens: Vec<u64> = (0..suffix as u64).collect();
            let trie: Vec<GpuTrieNode> = if suffix == 8 {
                // (start, end, height) per token
                bytemuck::cast_slice::<[u32; 3], GpuTrieNode>(&[
                    [0, 7, 0],
                    [1, 3, 1],
                    [2, 2, 2],
                    [3, 3, 2],
                    [4, 7, 1],
                    [5, 6, 2],
                    [6, 6, 3],
                    [7, 7, 2],
                ])
                .to_vec()
            } else {
                flat_trie(0, &tokens, &PRng::new(0)).linearize().token_subtrie_ranges().collect()
            };
            let (shape, nodes) =
                ((head_dim, q_heads, kv_heads, suffix, prefix, true), use_trie.then_some(trie.as_slice()));
            let mut config = default_attention_config::<bf16>(head_dim, q_heads, kv_heads, true);
            let gate_stride = q_heads * head_dim + 17;
            let gates = matches!(prefix, 37 | 4096).then(|| fill_attention(suffix * gate_stride, 3.0));
            config.gate_row_stride = gates.as_ref().map(|_| gate_stride as u32);
            let gate = gates.as_ref().map(|data| create_buffer_with_data::<Metal, bf16>(&metal, data));
            let kernel = AttentionMetalKernel::new_with_mxu(&metal, &config, use_mxu).unwrap();
            let (queries, keys, values) = attention_inputs(shape);
            // Fused QKV rows from `attention_inputs`; its queries are head-major.
            let stride = (q_heads + 2 * kv_heads) * head_dim;
            let mut rows = vec![bf16::ZERO; (prefix + suffix) * stride];
            for (row, (k, v)) in
                rows.chunks_mut(stride).zip(keys.chunks(kv_heads * head_dim).zip(values.chunks(kv_heads * head_dim)))
            {
                let (key_row, value_row) = row[q_heads * head_dim..].split_at_mut(kv_heads * head_dim);
                key_row.copy_from_slice(k);
                value_row.copy_from_slice(v);
            }
            for (index, query) in queries.chunks(head_dim).enumerate() {
                rows[(prefix + index % suffix) * stride + index / suffix * head_dim..][..head_dim]
                    .copy_from_slice(query);
            }
            let mut fill = metal.create_command_buffer(None, None).unwrap();
            let mut kv =
                KVCache::<Metal>::allocate_scratch(kernel.kv_layout(), (prefix + suffix).max(2048) as u32, &mut fill)
                    .unwrap();
            let pattern =
                create_buffer_with_data::<Metal, u16>(&metal, &vec![poison; kv.k_scales.size() / size_of::<bf16>()]);
            for buffer in [&mut kv.k_codes, &mut kv.v_codes] {
                fill.encode_fill(buffer.as_mut(), poison as u8);
            }
            for buffer in [&mut kv.k_scales, &mut kv.v_scales] {
                fill.encode_copy(&pattern, buffer.as_mut());
            }
            submit_command_buffer(fill);
            let mut step =
                |rows: &[bf16], prefix: usize, trie: Option<&[GpuTrieNode]>, append_kv: bool, attend: bool| {
                    let suffix = rows.len() / stride;
                    let query_rows = q_heads / kv_heads * suffix;
                    let prepare = Prepare::new(
                        &metal,
                        DataType::BF16,
                        DataType::F32,
                        head_dim as u32,
                        false,
                        kv_heads as u32,
                        kernel.kv_layout().backend_opaque_layout,
                        append_kv,
                        false,
                        false,
                        false,
                    )
                    .unwrap();
                    let mut query_codes = create_buffer::<Metal, i8>(&metal, kv_heads * query_rows * head_dim);
                    let mut query_scales = create_buffer::<Metal, bf16>(&metal, kv_heads * query_rows);
                    let mut command_buffer = metal.create_command_buffer(None, None).unwrap();
                    let trie = trie.map(|nodes| command_buffer.allocate_constant_from_slice(nodes).unwrap());
                    prepare.encode(
                        &create_buffer_with_data::<Metal, bf16>(&metal, rows),
                        &mut query_codes,
                        &mut query_scales,
                        append_kv.then_some(kv.k_codes.as_mut()),
                        append_kv.then_some(kv.k_scales.as_mut()),
                        append_kv.then_some(kv.v_codes.as_mut()),
                        append_kv.then_some(kv.v_scales.as_mut()),
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        Default::default(),
                        q_heads as u32,
                        None,
                        prefix as u32,
                        stride as u32,
                        suffix as u32,
                        query_rows as u32,
                        &mut command_buffer,
                    );
                    if !attend {
                        submit_command_buffer(command_buffer);
                        return Vec::new();
                    }
                    let input = AttentionInput {
                        query_codes: &query_codes,
                        query_scales: &query_scales,
                        k_codes: kv.k_codes.as_ref(),
                        k_scales: kv.k_scales.as_ref(),
                        v_codes: kv.v_codes.as_ref(),
                        v_scales: kv.v_scales.as_ref(),
                    };
                    let arguments = AttentionArguments {
                        sinks: None,
                        suffix_length: suffix as u32,
                        trie: trie.as_ref(),
                        cache: KVCacheView::full(prefix as u32),
                        gate: gate.as_ref(),
                    };
                    let pooled = kernel.encode(input, arguments, &mut command_buffer).unwrap();
                    let mut output = create_buffer::<Metal, bf16>(&metal, suffix * q_heads * head_dim);
                    command_buffer.encode_copy(&pooled, &mut output);
                    drop((pooled, trie));
                    submit_command_buffer(command_buffer);
                    buffer_to_vec::<Metal, bf16>(&metal, &output)
                };
            if prefix > 0 {
                step(&rows[..prefix * stride], 0, None, true, false);
            }
            let mut actual = step(&rows[prefix * stride..], prefix, nodes, true, true);
            if reader {
                // The reader's own K/V rows are junk: appending them would change the output.
                rows[prefix * stride..].chunks_mut(stride).for_each(|row| row[q_heads * head_dim..].fill(bf16::ONE));
                actual = step(&rows[prefix * stride..], prefix, nodes, false, true);
            }
            let label =
                format!("int8 attention mxu={use_mxu} q{q_heads} S{suffix} L{prefix} trie={use_trie} reader={reader}");
            let mut expected = quantized_attention_reference(shape, nodes);
            if let Some(gates) = &gates {
                for (token, row) in expected.chunks_exact_mut(q_heads * head_dim).enumerate() {
                    for (channel, value) in row.iter_mut().enumerate() {
                        *value = bf16::from_f32(
                            value.to_f32() / (1.0 + (-gates[token * gate_stride + channel].to_f32()).exp()),
                        );
                    }
                }
            }
            assert_eq_float::<bf16>(
                &expected,
                &actual,
                if use_mxu {
                    1e-2
                } else {
                    5e-3
                },
                &label,
            );
        }
    }
}

fn attention_inputs(shape: AttentionShape) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    let (dim, q_heads, kv_heads, suffix, prefix, _) = shape;
    let mut rng = SmallRng::seed_from_u64(42);
    let queries = (0..q_heads * suffix * dim).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();
    let keys = (0..(prefix + suffix) * kv_heads * dim).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let values = (0..(prefix + suffix) * kv_heads * dim)
        .map(|i| bf16::from_f32(rng.random_range(-0.5..0.5) + 0.25 * (i % dim) as f32 / dim as f32))
        .collect();
    (queries, keys, values)
}

fn quantized_attention_reference(
    shape: AttentionShape,
    nodes: Option<&[GpuTrieNode]>,
) -> Vec<bf16> {
    use crate::backends::common::gpu_types::kv_layout;
    fn hadamard_transform(row: &mut [f32]) {
        let mut width = 1;
        while width < row.len() {
            for block in row.chunks_exact_mut(2 * width) {
                let (left, right) = block.split_at_mut(width);
                for (a, b) in left.iter_mut().zip(right) {
                    (*a, *b) = (*a + *b, *a - *b);
                }
            }
            width *= 2;
        }
        let scale = (row.len() as f32).sqrt().recip();
        row.iter_mut().for_each(|value| *value *= scale);
    }
    let (head_dim, q_heads, kv_heads, suffix, prefix, causal) = shape;
    let signs: Vec<f32> = (0..head_dim)
        .map(|channel| {
            if kv_layout::ROTATION_SIGN_BITS[channel / 32 % 8] >> (channel % 32) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let quantize = |input: &[bf16]| {
        let mut result = Vec::with_capacity(input.len());
        for row in input.chunks_exact(head_dim) {
            let mut rotated: Vec<f32> = row.iter().zip(&signs).map(|(value, sign)| value.to_f32() * sign).collect();
            hadamard_transform(&mut rotated);
            let maximum = rotated.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = bf16::from_f32(if maximum > 0.0 {
                maximum / 127.0
            } else {
                1.0
            })
            .to_f32();
            result.extend(rotated.iter().map(|value| (value / scale).round_ties_even().clamp(-127.0, 127.0) * scale));
        }
        result
    };
    let (queries, keys, values) = attention_inputs(shape);
    let queries = quantize(&queries);
    let keys = quantize(&keys);
    let values = quantize(&values);
    let length = prefix + suffix;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut result = vec![0.0f32; suffix * q_heads * head_dim];
    for head in 0..q_heads {
        let kv_head = head / (q_heads / kv_heads);
        for token in 0..suffix {
            let query = &queries[(head * suffix + token) * head_dim..][..head_dim];
            let mut scores = vec![f32::NEG_INFINITY; length];
            for (key, score) in scores.iter_mut().enumerate() {
                if causal && key >= prefix {
                    let visible = nodes.map_or(key - prefix <= token, |nodes| {
                        let ancestor = &nodes[key - prefix];
                        (ancestor.trie_start..=ancestor.trie_end).contains(&(token as u32))
                    });
                    if !visible {
                        continue;
                    }
                }
                let k = &keys[(key * kv_heads + kv_head) * head_dim..][..head_dim];
                *score = query.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f32 = weights.iter().sum();
            let output = &mut result[(token * q_heads + head) * head_dim..][..head_dim];
            for (key, weight) in weights.iter().enumerate() {
                let v = &values[(key * kv_heads + kv_head) * head_dim..][..head_dim];
                for (sum, value) in output.iter_mut().zip(v) {
                    *sum += weight * value / denominator;
                }
            }
        }
    }
    for row in result.chunks_exact_mut(head_dim) {
        hadamard_transform(row);
        row.iter_mut().zip(&signs).for_each(|(value, sign)| *value *= sign);
    }
    result.into_iter().map(bf16::from_f32).collect()
}
