use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::{
            qkv_norm::{QKVL2Op, QKVNormParams},
            rope::RoPEGridParams,
            trie::TrieNode,
        },
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
}

#[derive(Clone, Copy)]
struct NormCase {
    full_layer: bool,
    ops: [QKVL2Op; 3],
    affine: [bool; 3],
}

const CLIP_THRESHOLDS: [f32; 3] = [2.0, 22.627417, 0.0];
const CLIP_NORM_RATIOS: [f32; 2] = [0.5, 3.0];

fn run_prepare<B: Backend, const DIM: usize>(
    num_q_heads: u32,
    normalization: NormCase,
    fused: bool,
    rope: bool,
    block_mean: bool,
) -> PrepareOutput {
    const CACHE_TOKENS: usize = 64;
    const SUFFIX: usize = 2;
    const ROW_PADDING: usize = 5;
    const PREFIX: u32 = 3;
    let code_bytes = CACHE_TOKENS * DIM;
    let norm_params = QKVNormParams {
        op: normalization.ops.map(|op| op as u32),
        full_layer: [u32::from(normalization.full_layer); 3],
        epsilon: [1e-6, 1e-5, 1e-4],
        clip_threshold: CLIP_THRESHOLDS,
        scale_offset: [0.0, 1.0, 0.25],
    };
    let weights: [Vec<f32>; 3] =
        std::array::from_fn(|head| (0..DIM).map(|i| 0.5 + (i as f32 * 0.13 + head as f32).sin() * 0.3).collect());
    let mut rows = Vec::new();
    for token in 0..SUFFIX {
        for head in (1 - num_q_heads) as usize..3 {
            let mut values = (0..DIM)
                .map(|i| bf16::from_f32(((token * 3 * DIM + head * DIM + i) as f32 * 0.37).sin()))
                .collect::<Vec<_>>();
            let op = normalization.ops[head];
            if op == QKVL2Op::Clip {
                let target = CLIP_THRESHOLDS[head] * CLIP_NORM_RATIOS[token];
                let norm = values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>().sqrt();
                values.iter_mut().for_each(|value| *value = bf16::from_f32(value.to_f32() * target / norm));
            }
            if !fused && op != QKVL2Op::None {
                let sum = values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
                let row_scale = if op == QKVL2Op::Clip {
                    (norm_params.clip_threshold[head] / (sum.sqrt() + norm_params.epsilon[head])).min(1.0)
                } else {
                    (sum / DIM as f32 + norm_params.epsilon[head]).sqrt().recip()
                };
                for (i, value) in values.iter_mut().enumerate() {
                    let normalized = value.to_f32() * row_scale;
                    let scale = weights[head][i] + norm_params.scale_offset[head];
                    *value = if !normalization.affine[head] {
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
    let context = block_on(B::Context::new()).unwrap();
    let layout = kv_layout::<B>(&context, DIM as u32, 1);
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &rows);
    let norm_scales = weights.map(|weights| create_buffer_with_data::<B, f32>(&context, &weights));
    let mut q_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; SUFFIX * DIM]);
    let mut q_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; SUFFIX]);
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
        false,
        1,
        layout.backend_opaque_layout,
        true,
        affine[0],
        affine[1],
        affine[2],
        block_mean,
        4 * u32::from(block_mean),
        false,
    )
    .unwrap();
    let nodes = create_buffer_with_data::<B, u32>(
        &context,
        bytemuck::cast_slice(&[
            TrieNode {
                trie_start: 0,
                trie_end: 0,
                height: 0,
                parent: -1,
            },
            TrieNode {
                trie_start: 0,
                trie_end: 0,
                height: 4,
                parent: -1,
            },
        ]),
    );
    let windows = create_buffer_with_data::<B, i32>(&context, &[0, 0, 3, 3, 4, 4, 4, 4]);
    let mut raw_keys =
        create_buffer_with_data::<B, bf16>(&context, &[rows[..DIM].to_vec(), vec![bf16::ZERO; 5 * DIM]].concat());
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &qkvg,
        &mut q_codes,
        &mut q_scales,
        Some(&mut k_codes),
        Some(&mut k_scales),
        Some(&mut v_codes),
        Some(&mut v_scales),
        rope.then_some(&rope_inverse_frequencies),
        affine[0].then_some(&norm_scales[0]),
        affine[1].then_some(&norm_scales[1]),
        affine[2].then_some(&norm_scales[2]),
        block_mean.then_some(&nodes),
        block_mean.then_some(&windows),
        block_mean.then_some(&mut raw_keys),
        if fused {
            norm_params
        } else {
            QKVNormParams::default()
        },
        num_q_heads,
        rope.then_some(rope_dim),
        rope.then_some(1.0),
        rope.then_some(5),
        None,
        33,
        (num_q_heads + 2) * DIM as u32 + ROW_PADDING as u32,
        SUFFIX as u32,
        num_q_heads * SUFFIX as u32,
        block_mean.then_some(PREFIX),
        None::<&B::GlobalGpuBuffer>,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    PrepareOutput {
        codes: [&q_codes, &k_codes, &v_codes].map(|buffer| buffer_to_vec::<B, u8>(&context, buffer)),
        scales: [&q_scales, &k_scales, &v_scales].map(|buffer| buffer_to_vec::<B, bf16>(&context, buffer)),
    }
}

#[uzu_test]
fn test_attention_prepare_fused_norm() {
    use QKVL2Op::{Clip, RmsNorm};

    fn check<const DIM: usize>() {
        for queries in [0, 1] {
            for full_layer in [false, true] {
                for (ops, affine) in [
                    ([RmsNorm, RmsNorm, QKVL2Op::None], [true, true, false]),
                    ([RmsNorm, RmsNorm, RmsNorm], [false, false, false]),
                    ([QKVL2Op::None, RmsNorm, RmsNorm], [false, false, true]),
                    ([Clip, Clip, RmsNorm], [false, false, true]),
                    ([Clip, Clip, QKVL2Op::None], [false, false, false]),
                ] {
                    let case = NormCase {
                        full_layer,
                        ops,
                        affine,
                    };
                    for rope in [false, true] {
                        for_each_backend!(|B| {
                            let separate = run_prepare::<B, DIM>(queries, case, false, rope, false);
                            let fused = run_prepare::<B, DIM>(queries, case, true, rope, false);
                            assert_eq!(
                                separate, fused,
                                "fused norm mismatch D{DIM} queries={queries} full_layer={full_layer} rope={rope}"
                            );
                            if queries == 1 {
                                let block_mean = run_prepare::<B, DIM>(queries, case, true, rope, true);
                                assert_eq!(
                                    (&block_mean.codes[0], &block_mean.scales[0]),
                                    (&fused.codes[0], &fused.scales[0])
                                );
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

/// Codes and scales of the query head of each of `rows`, rotated at `position_offset + row` or over `grid`
fn run_rope_queries<B: Backend>(
    context: &B::Context,
    dim: usize,
    rows: &[bf16],
    inverse_frequencies: &[f32],
    position_offset: u32,
    grid: Option<RoPEGridParams>,
) -> (Vec<i8>, Vec<bf16>) {
    let suffix = rows.len() / dim;
    let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        context,
        DataType::BF16,
        dim as u32,
        true,
        grid.is_some(),
        1,
        kv_layout::<B>(context, dim as u32, 1).backend_opaque_layout,
        false,
        false,
        false,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    let qkvg = create_buffer_with_data::<B, bf16>(context, rows);
    let inverse_frequencies_buffer = create_buffer_with_data::<B, f32>(context, inverse_frequencies);
    let mut q_codes = create_buffer_with_data::<B, i8>(context, &vec![0; suffix * dim]);
    let mut q_scales = create_buffer_with_data::<B, bf16>(context, &vec![bf16::ZERO; suffix]);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &qkvg,
        &mut q_codes,
        &mut q_scales,
        None::<&mut B::GlobalGpuBuffer>,
        None::<&mut B::GlobalGpuBuffer>,
        None::<&mut B::GlobalGpuBuffer>,
        None::<&mut B::GlobalGpuBuffer>,
        Some(&inverse_frequencies_buffer),
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        None::<&mut B::GlobalGpuBuffer>,
        QKVNormParams::default(),
        1,
        Some(2 * inverse_frequencies.len() as u32),
        Some(1.0),
        Some(position_offset),
        grid,
        0,
        dim as u32,
        suffix as u32,
        suffix as u32,
        None,
        None::<&B::GlobalGpuBuffer>,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (buffer_to_vec::<B, i8>(context, &q_codes), buffer_to_vec::<B, bf16>(context, &q_scales))
}

/// Checks M-RoPE over `grid` against plain RoPE. `axes` is the axis of each pair: 0 position, 1 row, 2 column;
/// `positions` are the expected positions of each row of the batch along the three axes.
fn check_mrope<B: Backend>(
    dim: usize,
    axes: &[usize],
    grid: RoPEGridParams,
    position_offset: u32,
    positions: &[[u32; 3]],
) {
    let context = block_on(B::Context::new()).unwrap();
    let pairs = axes.len();
    let frequencies: Vec<f32> = (0..pairs).map(|pair| 10_000f32.powf(-(pair as f32) / pairs as f32)).collect();
    let rows: Vec<bf16> = (0..positions.len() * dim).map(|i| bf16::from_f32((i as f32 * 0.37).sin())).collect();
    let (codes, scales) = run_rope_queries::<B>(&context, dim, &rows, &frequencies, position_offset, Some(grid));
    for (row, positions) in positions.iter().enumerate() {
        // Plain RoPE at position 1 turns a pair by its frequency, so scale each by the position along its axis
        let scaled: Vec<f32> =
            frequencies.iter().zip(axes).map(|(frequency, &axis)| positions[axis] as f32 * frequency).collect();
        let (expected_codes, expected_scales) =
            run_rope_queries::<B>(&context, dim, &rows[row * dim..][..dim], &scaled, 1, None);
        assert_eq!(
            (&codes[row * dim..][..dim], scales[row]),
            (&expected_codes[..], expected_scales[0]),
            "grid {grid:?} row {row}"
        );
    }
}

#[uzu_test]
fn test_attention_prepare_mrope() {
    for_each_backend!(|B| {
        // Decoder: pairs 1, 4, 7, ... of 11 rotate by the row, pairs 2, 5, 8, ... of 10 by the column
        let mut axes = vec![0; 32];
        (1..3 * 11).step_by(3).for_each(|pair| axes[pair] = 1);
        (2..3 * 10).step_by(3).for_each(|pair| axes[pair] = 2);
        let decoder_grid = |grid_width, grid_height, first_cell| RoPEGridParams {
            grid_width,
            grid_height,
            first_cell,
            height_pairs: 11,
            width_pairs: 10,
            merge_size: 1,
            interleaved_axes: 1,
        };
        // An image sits at one position, its cells go in raster order
        let image: Vec<[u32; 3]> = (0..15).map(|cell| [7, 7 + cell / 5, 7 + cell % 5]).collect();
        check_mrope::<B>(64, &axes, decoder_grid(5, 3, 0), 7, &image);
        check_mrope::<B>(64, &axes, decoder_grid(5, 3, 6), 7, &image[6..10]);
        // Text tokens follow each other along all the axes
        let text: Vec<[u32; 3]> = (7..10).map(|position| [position; 3]).collect();
        check_mrope::<B>(64, &axes, decoder_grid(1, 1, 0), 7, &text);

        // Vision encoder: the first half of the pairs rotates by the row, the second by the column
        let axes = [[1; 18], [2; 18]].concat();
        let vision_grid = RoPEGridParams {
            grid_width: 6,
            grid_height: 4,
            first_cell: 0,
            height_pairs: 18,
            width_pairs: 18,
            merge_size: 2,
            interleaved_axes: 0,
        };
        // Patches go block by block, each 2x2 block in raster order
        let mut patches = Vec::new();
        for block_row in 0..2 {
            for block_column in 0..3 {
                for in_row in 0..2 {
                    for in_column in 0..2 {
                        patches.push([0, block_row * 2 + in_row, block_column * 2 + in_column]);
                    }
                }
            }
        }
        check_mrope::<B>(128, &axes, vision_grid, 0, &patches);
    });
}
