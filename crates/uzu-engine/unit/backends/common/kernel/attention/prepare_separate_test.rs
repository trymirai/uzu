use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::{attention_prepare::SeparatePrepareParams, trie::TrieNode},
            kernel::AttentionPrepareSeparateKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_parity,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const DIM: usize = 128;
const Q_HEADS: usize = 6;
const KV_HEADS: usize = 2;
const KERNEL_SIZE: usize = 4;
const RELATIVE_DIM: usize = 16;
const EXTENT: usize = 64;
const CACHE_TOKENS: usize = 64;
const PREFIX: u32 = 3;

#[derive(Clone, Copy, PartialEq)]
enum Batch {
    Decode,
    Flat,
    Trie,
}

struct Prepared {
    q_codes: Vec<i8>,
    q_scales: Vec<bf16>,
    k_codes: Vec<i8>,
    k_scales: Vec<bf16>,
    v_codes: Vec<u8>,
    v_scales: Vec<bf16>,
    relative_logits: Vec<bf16>,
    key_state: Vec<f32>,
    value_state: Vec<f32>,
    key_nodes: Vec<f32>,
}

fn value(index: usize, salt: f32) -> f32 {
    (index as f32 * 0.379 + salt).sin() * (1.5 + (salt * 1.3).cos())
}

fn run<B: Backend>(batch: Batch) -> Prepared {
    let tokens = if batch == Batch::Decode {
        1
    } else {
        5
    };
    let q_dim = Q_HEADS * DIM;
    let kv_dim = KV_HEADS * DIM;
    let query_stride = q_dim + Q_HEADS * RELATIVE_DIM + 9;
    let key_stride = kv_dim + KV_HEADS * KERNEL_SIZE;
    let history = KERNEL_SIZE - 1;
    let context = block_on(B::Context::new()).expect("create context");
    // Row-major KV layout, so that the codes compare across backends.
    let kernel = <B::Kernels as Kernels>::AttentionPrepareSeparateKernel::new(
        &context,
        DataType::BF16,
        DIM as u32,
        Q_HEADS as u32,
        KV_HEADS as u32,
        0,
        KERNEL_SIZE as u32,
        true,
        true,
        true,
        true,
        batch == Batch::Decode,
        true,
        true,
        true,
        true,
        true,
        true,
        RELATIVE_DIM as u32,
        EXTENT as u32,
        batch == Batch::Trie,
    )
    .expect("create kernel");
    let rows = |count: usize, stride: usize, salt: f32| {
        (0..count * stride).map(|i| bf16::from_f32(value(i, salt))).collect::<Vec<_>>()
    };
    let queries = create_buffer_with_data::<B, bf16>(&context, &rows(tokens, query_stride, 0.2));
    let mut key_rows = rows(tokens, key_stride, 1.1);
    let mut value_rows = rows(tokens, key_stride, 2.7);
    // Small dynamic taps, like trained ones.
    for row in [&mut key_rows, &mut value_rows] {
        for token in 0..tokens {
            for tap in 0..KV_HEADS * KERNEL_SIZE {
                row[token * key_stride + kv_dim + tap] = bf16::from_f32(0.1 * value(token * 31 + tap, 0.5));
            }
        }
    }
    let keys = create_buffer_with_data::<B, bf16>(&context, &key_rows);
    let values = create_buffer_with_data::<B, bf16>(&context, &value_rows);
    let weights = (0..kv_dim * KERNEL_SIZE).map(|i| 0.4 * value(i, 3.3)).collect::<Vec<f32>>();
    let key_weights = create_buffer_with_data::<B, f32>(&context, &weights);
    let value_weights = create_buffer_with_data::<B, f32>(&context, &weights.iter().map(|w| -w).collect::<Vec<_>>());
    let state = (0..kv_dim * history).map(|i| value(i, 4.4)).collect::<Vec<f32>>();
    let mut key_state = create_buffer_with_data::<B, f32>(&context, &state);
    let mut value_state = create_buffer_with_data::<B, f32>(&context, &state);
    let mut key_nodes = create_buffer::<B, f32>(&context, tokens * kv_dim * history);
    let mut value_nodes = create_buffer::<B, f32>(&context, tokens * kv_dim * history);
    let clip_scales = (0..Q_HEADS * DIM).map(|i| 0.8 + 0.4 * value(i, 0.9).abs()).collect::<Vec<f32>>();
    let clip_scales = create_buffer_with_data::<B, f32>(&context, &clip_scales);
    let relative = (0..Q_HEADS * RELATIVE_DIM + RELATIVE_DIM * EXTENT).map(|i| value(i, 5.1)).collect::<Vec<f32>>();
    let relative = create_buffer_with_data::<B, f32>(&context, &relative);
    let mut relative_logits = create_buffer::<B, bf16>(&context, Q_HEADS * tokens * EXTENT);
    let mut q_codes = create_buffer::<B, i8>(&context, Q_HEADS * tokens * DIM);
    let mut q_scales = create_buffer::<B, bf16>(&context, Q_HEADS * tokens);
    let mut k_codes = create_buffer::<B, i8>(&context, CACHE_TOKENS * kv_dim);
    let mut k_scales = create_buffer::<B, bf16>(&context, CACHE_TOKENS * KV_HEADS);
    let mut v_codes = create_buffer::<B, u8>(&context, CACHE_TOKENS * kv_dim);
    let mut v_scales = create_buffer::<B, bf16>(&context, CACHE_TOKENS * KV_HEADS);
    // (trie_start, trie_end, height, parent)
    let nodes: &[TrieNode] = bytemuck::cast_slice::<[u32; 4], TrieNode>(&[
        [0, 4, 0, u32::MAX],
        [1, 2, 1, 0],
        [2, 2, 2, 1],
        [3, 4, 1, 0],
        [4, 4, 2, 3],
    ]);
    let trie = create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(nodes));
    let params = SeparatePrepareParams {
        query_row_stride: query_stride as u32,
        key_row_stride: key_stride as u32,
        value_row_stride: key_stride as u32,
        relative_offset: q_dim as u32,
        kv_token_offset: PREFIX,
        query_rows: (Q_HEADS / KV_HEADS * tokens) as u32,
        position_offset: 9000,
        query_clip_threshold: 6.0,
        key_clip_threshold: 5.0,
        relative_clip_threshold: 2.0,
        log_scaling_n_floor: 8192.0,
        log_scaling_alpha: 0.1,
        key_conv_scale: 1.0 / 3.0,
        value_conv_scale: 0.5,
        value_norm_epsilon: 1e-6,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &queries,
        &keys,
        &values,
        Some(&key_weights),
        Some(&value_weights),
        Some(&mut key_state),
        Some(&mut value_state),
        (batch == Batch::Trie).then_some(&mut key_nodes),
        (batch == Batch::Trie).then_some(&mut value_nodes),
        Some(&clip_scales),
        Some(&relative),
        Some(&mut relative_logits),
        &mut q_codes,
        &mut q_scales,
        &mut k_codes,
        &mut k_scales,
        &mut v_codes,
        &mut v_scales,
        (batch == Batch::Trie).then_some(&trie),
        params,
        tokens as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run kernel");
    Prepared {
        q_codes: buffer_to_vec::<B, i8>(&context, &q_codes),
        q_scales: buffer_to_vec::<B, bf16>(&context, &q_scales),
        k_codes: buffer_to_vec::<B, i8>(&context, &k_codes),
        k_scales: buffer_to_vec::<B, bf16>(&context, &k_scales),
        v_codes: buffer_to_vec::<B, u8>(&context, &v_codes),
        v_scales: buffer_to_vec::<B, bf16>(&context, &v_scales),
        relative_logits: buffer_to_vec::<B, bf16>(&context, &relative_logits),
        key_state: buffer_to_vec::<B, f32>(&context, &key_state),
        value_state: buffer_to_vec::<B, f32>(&context, &value_state),
        key_nodes: buffer_to_vec::<B, f32>(&context, &key_nodes),
    }
}

fn assert_codes_close<T: Copy + Into<i32>>(
    label: &str,
    expected: &[T],
    actual: &[T],
) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    let mismatches = expected.iter().zip(actual).filter(|(e, a)| ((**e).into() - (**a).into()).abs() > 1).count();
    assert_eq!(mismatches, 0, "{label}: {mismatches} codes differ by more than one step");
}

/// The KV cache row of a key and head: `[key][head]` on the CPU, `[key block][head][key in block]` on the GPU.
fn kv_row(
    cpu: bool,
    key: usize,
    head: usize,
) -> usize {
    const TILE_KEYS: usize = 32;
    if cpu {
        key * KV_HEADS + head
    } else {
        (key / TILE_KEYS * KV_HEADS + head) * TILE_KEYS + key % TILE_KEYS
    }
}

/// The rows of the written keys, in key and head order.
fn written_rows<T: Copy>(
    cpu: bool,
    values: &[T],
    tokens: usize,
    width: usize,
) -> Vec<T> {
    let mut rows = Vec::new();
    for key in PREFIX as usize..PREFIX as usize + tokens {
        for head in 0..KV_HEADS {
            let row = kv_row(cpu, key, head);
            rows.extend_from_slice(&values[row * width..(row + 1) * width]);
        }
    }
    rows
}

#[uzu_test]
fn test_attention_prepare_separate_matches_cpu() {
    for batch in [Batch::Decode, Batch::Flat, Batch::Trie] {
        let expected = run::<Cpu>(batch);
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(batch);
            let label = match batch {
                Batch::Decode => "decode",
                Batch::Flat => "flat",
                Batch::Trie => "trie",
            };
            assert_codes_close(&format!("{label} q codes"), &expected.q_codes, &actual.q_codes);
            assert_parity(&format!("{label} q scales"), &expected.q_scales, &actual.q_scales, 1e-2, 1e-4);
            let tokens = expected.q_scales.len() / Q_HEADS;
            let widen = |codes: &[i8]| codes.iter().map(|&code| i32::from(code)).collect::<Vec<i32>>();
            let widen_values = |codes: &[u8]| codes.iter().map(|&code| i32::from(code)).collect::<Vec<i32>>();
            for (name, expected_codes, actual_codes, expected_scales, actual_scales) in [
                ("k", widen(&expected.k_codes), widen(&actual.k_codes), &expected.k_scales, &actual.k_scales),
                (
                    "v",
                    widen_values(&expected.v_codes),
                    widen_values(&actual.v_codes),
                    &expected.v_scales,
                    &actual.v_scales,
                ),
            ] {
                assert_codes_close(
                    &format!("{label} {name} codes"),
                    &written_rows(true, &expected_codes, tokens, DIM),
                    &written_rows(false, &actual_codes, tokens, DIM),
                );
                assert_parity(
                    &format!("{label} {name} scales"),
                    &written_rows(true, expected_scales, tokens, 1),
                    &written_rows(false, actual_scales, tokens, 1),
                    1e-2,
                    1e-4,
                );
            }
            assert_parity(&format!("{label} relative"), &expected.relative_logits, &actual.relative_logits, 1e-2, 1e-2);
            assert_parity(&format!("{label} key state"), &expected.key_state, &actual.key_state, 1e-6, 1e-6);
            assert_parity(&format!("{label} value state"), &expected.value_state, &actual.value_state, 1e-6, 1e-6);
            if batch == Batch::Trie {
                assert_parity(&format!("{label} key nodes"), &expected.key_nodes, &actual.key_nodes, 1e-6, 1e-6);
            }
        });
    }
}
