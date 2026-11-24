use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::{qkv_norm::QKVNormParams, trie::TrieNode},
        kernel::AttentionPrepareKernel,
    },
    data_type::DataType,
    tests::{
        attention::kv_layout,
        helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
    },
};

#[derive(Debug, PartialEq)]
struct PrepareOutput {
    codes: [Vec<u8>; 3],
    scales: [Vec<bf16>; 3],
    raw_keys: Option<Vec<bf16>>,
}

#[derive(Clone, Copy)]
struct NormCase {
    full_layer: bool,
    enabled: [bool; 3],
    affine: [bool; 3],
}

#[derive(Clone, Copy)]
enum PrepareMode {
    Normal,
    PoolKeys,
    KeysOnly,
    KeyReference,
}

fn run_prepare<B: Backend, const DIM: usize>(
    num_q_heads: u32,
    normalization: NormCase,
    fused: bool,
    rope: bool,
    mode: PrepareMode,
) -> PrepareOutput {
    const CACHE_TOKENS: usize = 64;
    const SUFFIX: usize = 2;
    const ROW_PADDING: usize = 5;
    const PREFIX: u32 = 3;
    let pool_keys = matches!(mode, PrepareMode::PoolKeys | PrepareMode::KeysOnly);
    let keys_only = matches!(mode, PrepareMode::KeysOnly);
    let key_reference = matches!(mode, PrepareMode::KeyReference);
    let rope_offset = if key_reference {
        0
    } else {
        5
    };
    let code_bytes = CACHE_TOKENS * DIM;
    let norm_params = QKVNormParams {
        enabled: normalization.enabled.map(u32::from),
        full_layer: [u32::from(normalization.full_layer); 3],
        epsilon: [1e-6, 1e-5, 1e-4],
        scale_offset: [0.0, 1.0, 0.25],
    };
    let weights: [Vec<f32>; 3] =
        std::array::from_fn(|head| (0..DIM).map(|i| 0.5 + (i as f32 * 0.13 + head as f32).sin() * 0.3).collect());
    let mut rows = Vec::new();
    for token in 0..SUFFIX {
        for head in 0..num_q_heads as usize + 2 {
            let norm = if head < num_q_heads as usize {
                0
            } else {
                head - num_q_heads as usize + 1
            };
            let mut values = (0..DIM)
                .map(|i| bf16::from_f32(((token * 3 * DIM + head * DIM + i) as f32 * 0.37).sin()))
                .collect::<Vec<_>>();
            if !fused && normalization.enabled[norm] {
                let sum = values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
                let inverse_rms = (sum / DIM as f32 + norm_params.epsilon[norm]).sqrt().recip();
                for (i, value) in values.iter_mut().enumerate() {
                    let normalized = value.to_f32() * inverse_rms;
                    let scale = weights[norm][i] + norm_params.scale_offset[norm];
                    *value = if !normalization.affine[norm] {
                        bf16::from_f32(normalized)
                    } else if normalization.full_layer {
                        bf16::from_f32(normalized * scale)
                    } else {
                        bf16::from_f32(normalized) * bf16::from_f32(scale)
                    };
                }
            }
            rows.extend(values);
        }
        rows.extend([bf16::ZERO; ROW_PADDING]);
    }
    let key_offset = num_q_heads as usize * DIM;
    let row_stride = (num_q_heads as usize + 2) * DIM + ROW_PADDING;
    let keys: [Vec<bf16>; SUFFIX] =
        std::array::from_fn(|token| rows[token * row_stride + key_offset..][..DIM].to_vec());
    let cached_key: Vec<_> = keys[0].iter().map(|value| bf16::from_f32(value.to_f32() * 0.5 + 0.25)).collect();
    if key_reference {
        for (token, previous) in [&cached_key, &keys[0]].into_iter().enumerate() {
            for (channel, (&previous, &current)) in previous.iter().zip(&keys[token]).enumerate() {
                rows[token * row_stride + key_offset + channel] =
                    bf16::from_f32((previous.to_f32() + current.to_f32()) * 0.5);
            }
        }
    }
    let context = block_on(B::Context::new()).unwrap();
    let layout = kv_layout::<B>(&context, DIM as u32, 1);
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &rows);
    let norm_scales = weights.map(|weights| create_buffer_with_data::<B, f32>(&context, &weights));
    let mut q_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; num_q_heads.max(1) as usize * SUFFIX * DIM]);
    let mut q_scales =
        create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; num_q_heads.max(1) as usize * SUFFIX]);
    let mut k_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; code_bytes]);
    let mut v_codes = create_buffer_with_data::<B, u8>(&context, &vec![0; code_bytes]);
    let mut k_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let mut v_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let rope_dim = DIM as u32 / 2;
    let rope_inverse_frequencies = create_buffer_with_data::<B, f32>(
        &context,
        &(0..rope_dim / 2).map(|pair| 10_000f32.powf(-2.0 * pair as f32 / rope_dim as f32)).collect::<Vec<_>>(),
    );
    let affine = normalization.affine.map(|affine| affine && fused);
    let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DIM as u32,
        rope,
        1,
        layout.backend_opaque_layout,
        true,
        affine[0],
        affine[1],
        affine[2],
        pool_keys,
        4 * u32::from(pool_keys),
        true,
    )
    .unwrap();
    let nodes = create_buffer_with_data::<B, u32>(
        &context,
        bytemuck::cast_slice(&[0, 4].map(|height| TrieNode {
            trie_start: 0,
            trie_end: 0,
            height,
            parent: -1,
        })),
    );
    let path_token_indices = create_buffer_with_data::<B, i32>(&context, &[0, 0, 3, 3, 3, 3, 4, 4]);
    let mut raw_keys = create_buffer_with_data::<B, bf16>(&context, &[cached_key, vec![bf16::ZERO; 5 * DIM]].concat());
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let query_heads = if keys_only {
        0
    } else {
        num_q_heads
    };
    let input_offset = if keys_only {
        key_offset * DataType::BF16.size_in_bytes()
    } else {
        0
    };
    kernel.encode(
        (&qkvg).subrange(input_offset..),
        &mut q_codes,
        &mut q_scales,
        Some(&mut k_codes),
        Some(&mut k_scales),
        (!pool_keys).then_some(&mut v_codes),
        (!pool_keys).then_some(&mut v_scales),
        rope.then_some(&rope_inverse_frequencies),
        affine[0].then_some(&norm_scales[0]),
        affine[1].then_some(&norm_scales[1]),
        affine[2].then_some(&norm_scales[2]),
        pool_keys.then_some(&nodes),
        pool_keys.then_some(&path_token_indices),
        pool_keys.then_some(&mut raw_keys),
        if fused {
            norm_params
        } else {
            QKVNormParams::default()
        },
        query_heads,
        rope.then_some(rope_dim),
        rope.then_some(1.0),
        rope.then_some(rope_offset),
        33,
        (num_q_heads + 2) * DIM as u32 + ROW_PADDING as u32,
        SUFFIX as u32,
        query_heads * SUFFIX as u32,
        pool_keys.then_some(PREFIX),
        Some(&nodes),
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    PrepareOutput {
        codes: [&q_codes, &k_codes, &v_codes].map(|buffer| buffer_to_vec::<B, u8>(&context, buffer)),
        scales: [&q_scales, &k_scales, &v_scales].map(|buffer| buffer_to_vec::<B, bf16>(&context, buffer)),
        raw_keys: pool_keys.then(|| buffer_to_vec::<B, bf16>(&context, &raw_keys)),
    }
}

#[uzu_test]
fn test_attention_prepare_fused_norm() {
    fn check<const DIM: usize>() {
        for queries in [0, 1, 4] {
            for full_layer in [false, true] {
                for (enabled, affine) in [
                    ([true, true, false], [true, true, false]),
                    ([true, true, true], [false, false, false]),
                    ([false, true, true], [false, false, true]),
                ] {
                    let case = NormCase {
                        full_layer,
                        enabled,
                        affine,
                    };
                    for rope in [false, true] {
                        for_each_backend!(|B| {
                            let separate = run_prepare::<B, DIM>(queries, case, false, rope, PrepareMode::Normal);
                            let fused = run_prepare::<B, DIM>(queries, case, true, rope, PrepareMode::Normal);
                            assert_eq!(
                                separate, fused,
                                "fused norm mismatch D{DIM} queries={queries} full_layer={full_layer} rope={rope}"
                            );
                            let pooled = run_prepare::<B, DIM>(queries, case, true, rope, PrepareMode::PoolKeys);
                            let key_reference =
                                run_prepare::<B, DIM>(queries, case, true, rope, PrepareMode::KeyReference);
                            let keys_only = run_prepare::<B, DIM>(queries, case, true, rope, PrepareMode::KeysOnly);
                            assert_eq!(pooled.codes[1], keys_only.codes[1]);
                            assert_eq!(pooled.scales[1], keys_only.scales[1]);
                            assert_eq!(pooled.raw_keys, keys_only.raw_keys);
                            assert!(keys_only.codes[0].iter().all(|&code| code == 0));
                            assert!(keys_only.scales[0].iter().all(|&scale| scale == bf16::ZERO));
                            for (head, reference) in [fused, key_reference].iter().enumerate() {
                                assert_eq!(pooled.codes[head], reference.codes[head]);
                                assert_eq!(pooled.scales[head], reference.scales[head]);
                            }
                        });
                    }
                }
            }
        }
    }
    check::<64>();
    check::<128>();
    check::<256>();
    check::<512>();
}
