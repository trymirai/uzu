use futures::executor::block_on;
use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
            GlobalBufferCpuGpu,
        },
        metal::{
            Metal,
            kernel::{
                AnePackMetalKernel, AneStageDownRowsMetalKernel, AneStageUpRowsMetalKernel,
                AneTransposeCodesMetalKernel, AneUnpackMetalKernel,
            },
        },
    },
    tests::helpers::create_context,
};

type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;

fn upload<T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
    context: &<Metal as Backend>::Context,
    values: &[T],
) -> GpuBuffer {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(values).max(16)).unwrap();
    buffer.as_slice_mut::<u8>().fill(0);
    buffer.as_slice_mut::<u8>()[..size_of_val(values)].copy_from_slice(bytemuck::cast_slice(values));
    buffer.handoff()
}

/// +-1 signs as the bit mask `AneStageRows` reads (bit `i % 16` of u16 `i / 16` set for negative signs).
fn sign_bits(signs: &[f32]) -> Vec<u16> {
    signs
        .chunks(16)
        .map(|group| {
            group.iter().enumerate().fold(0u16, |bits, (index, sign)| bits | (u16::from(*sign < 0.0) << index))
        })
        .collect()
}

fn zeros(
    context: &<Metal as Backend>::Context,
    bytes: usize,
) -> GpuBuffer {
    upload::<u8>(context, &vec![0u8; bytes])
}

fn read<T: bytemuck::AnyBitPattern>(buffer: &GpuBuffer) -> &[T] {
    buffer.as_slice::<T>()
}

fn run(
    context: &std::sync::Arc<<Metal as Backend>::Context>,
    encode: impl FnOnce(&mut <<Metal as Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding),
) {
    let mut encoding = context.create_command_buffer(None, None).unwrap();
    encode(&mut encoding);
    block_on(encoding.end_encoding().submit().wait_until_completed()).unwrap();
}

/// Orthonormal Sylvester Walsh-Hadamard transform over consecutive blocks of `block`.
fn hadamard(
    values: &mut [f64],
    block: usize,
) {
    for chunk in values.chunks_mut(block) {
        let mut half_span = 1;
        while half_span < block {
            for start in (0..block).step_by(2 * half_span) {
                for index in start..start + half_span {
                    let (left, right) = (chunk[index], chunk[index + half_span]);
                    chunk[index] = left + right;
                    chunk[index + half_span] = left - right;
                }
            }
            half_span *= 2;
        }
        let scale = 1.0 / (block as f64).sqrt();
        chunk.iter_mut().for_each(|value| *value *= scale);
    }
}

fn signs(
    rng: &mut StdRng,
    count: usize,
) -> Vec<f32> {
    (0..count)
        .map(|_| {
            if rng.random::<bool>() {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn quantize_row(values: &[f64]) -> (Vec<i8>, f64) {
    let peak = values.iter().fold(0.0f64, |peak, value| peak.max(value.abs())).max(1e-20);
    let codes = values.iter().map(|value| (value * 127.0 / peak).round().clamp(-127.0, 127.0) as i8).collect();
    (codes, peak / 127.0)
}

fn assert_codes_close(
    actual: &[i8],
    expected: &[i8],
    what: &str,
) {
    let off_by_one = actual.iter().zip(expected).filter(|(a, e)| a != e).count();
    let worst = actual.iter().zip(expected).map(|(a, e)| (*a as i32 - *e as i32).abs()).max().unwrap_or(0);
    assert!(worst <= 1, "{what}: code differs by {worst}");
    assert!(off_by_one * 100 <= actual.len(), "{what}: {off_by_one} of {} codes differ", actual.len());
}

#[uzu_test]
fn ane_pack_rotates_and_quantizes_per_token() {
    let context = create_context::<Metal>();
    let mut rng = StdRng::seed_from_u64(1);
    let (hidden, tokens) = (4096usize, 37usize);
    let input: Vec<bf16> = (0..hidden * tokens)
        .map(|index| {
            let outlier = if index % 997 == 0 {
                40.0
            } else {
                1.0
            };
            bf16::from_f32((rng.random::<f32>() - 0.5) * outlier)
        })
        .collect();
    let sign = signs(&mut rng, hidden);
    let input_buffer = upload(&context, &input);
    let sign_buffer = upload(&context, &sign);
    let mut codes = zeros(&context, hidden * tokens);
    let mut scales = zeros(&context, tokens * 2);
    let kernel = AnePackMetalKernel::new(&context).unwrap();
    run(&context, |encoding| {
        kernel.encode(&input_buffer, &sign_buffer, &mut codes, &mut scales, hidden as u32, tokens as u32, encoding)
    });

    for token in 0..tokens {
        let mut values: Vec<f64> =
            (0..hidden).map(|index| input[token * hidden + index].to_f32() as f64 * sign[index] as f64).collect();
        hadamard(&mut values, hidden);
        let (expected, scale) = quantize_row(&values);
        let actual = &read::<i8>(&codes)[token * hidden..(token + 1) * hidden];
        assert_codes_close(actual, &expected, "pack");
        let actual_scale = read::<f16>(&scales)[token].to_f32() as f64 / 128.0;
        assert!((actual_scale - scale).abs() <= scale * 1e-3, "token scale {actual_scale} vs {scale}");
    }
}

#[uzu_test]
fn ane_transpose_codes_to_channel_major_surface() {
    let context = create_context::<Metal>();
    let mut rng = StdRng::seed_from_u64(2);
    let (channels, tokens, row_stride) = (256usize, 45usize, 128usize);
    let codes: Vec<i8> = (0..channels * tokens).map(|_| rng.random::<i8>()).collect();
    let codes_buffer = upload(&context, &codes);
    let mut surface = zeros(&context, channels * row_stride);
    let kernel = AneTransposeCodesMetalKernel::new(&context).unwrap();
    run(&context, |encoding| {
        kernel.encode(&codes_buffer, &mut surface, channels as u32, tokens as u32, row_stride as u32, encoding)
    });
    let actual = read::<i8>(&surface);
    for channel in 0..channels {
        for token in 0..tokens {
            assert_eq!(actual[channel * row_stride + token], codes[token * channels + channel]);
        }
    }
}

fn check_unpack(scale_row: bool) {
    let context = create_context::<Metal>();
    let mut rng = StdRng::seed_from_u64(3);
    let (hidden, tokens, row_stride) = (512usize, 45usize, 64usize);
    let surface: Vec<f16> =
        (0..(hidden + 1) * row_stride).map(|_| f16::from_f32(rng.random::<f32>() * 2.0 - 1.0)).collect();
    let token_scales: Vec<f16> = (0..row_stride).map(|_| f16::from_f32(rng.random::<f32>() + 0.5)).collect();
    let sign = signs(&mut rng, hidden);
    let surface_buffer = upload(&context, &surface);
    let token_buffer = upload(&context, &token_scales);
    let sign_buffer = upload(&context, &sign);
    let mut output = zeros(&context, hidden * tokens * 2);
    let kernel = AneUnpackMetalKernel::new(&context, scale_row).unwrap();
    run(&context, |encoding| {
        kernel.encode(
            &surface_buffer,
            &token_buffer,
            &sign_buffer,
            &mut output,
            hidden as u32,
            tokens as u32,
            row_stride as u32,
            encoding,
        )
    });
    let actual = read::<bf16>(&output);
    for token in 0..tokens {
        let row_scale = if scale_row {
            surface[hidden * row_stride + token].to_f64()
        } else {
            1.0
        };
        let scale = row_scale * token_scales[token].to_f64();
        let mut values: Vec<f64> =
            (0..hidden).map(|channel| surface[channel * row_stride + token].to_f64() * scale).collect();
        hadamard(&mut values, 32);
        for channel in 0..hidden {
            let expected = values[channel] * sign[channel] as f64;
            let got = actual[token * hidden + channel].to_f64();
            assert!(
                (got - expected).abs() <= 1e-2 * expected.abs() + 1e-3,
                "unpack[{token},{channel}] (scale row {scale_row}) {got} vs {expected}"
            );
        }
    }
}

#[uzu_test]
fn ane_unpack_scales_rotates_and_transposes() {
    check_unpack(true);
}

#[uzu_test]
fn ane_unpack_without_scale_row_scales_rotates_and_transposes() {
    check_unpack(false);
}

struct W4 {
    codes: Vec<u8>,
    scales: Vec<bf16>,
    zero_points: Vec<u8>,
    scale_stride: usize,
}

fn random_w4(
    rng: &mut StdRng,
    rows: usize,
    columns: usize,
) -> W4 {
    let groups = columns / 32;
    let scale_stride = rows.next_multiple_of(4);
    W4 {
        codes: (0..rows * columns / 2).map(|_| rng.random::<u8>()).collect(),
        scales: (0..groups * scale_stride).map(|_| bf16::from_f32(rng.random::<f32>() * 0.02 + 0.001)).collect(),
        zero_points: (0..groups * scale_stride / 2).map(|_| rng.random::<u8>()).collect(),
        scale_stride,
    }
}

fn dequantized_row(
    weights: &W4,
    row: usize,
    columns: usize,
) -> Vec<f64> {
    (0..columns)
        .map(|column| {
            let code = (weights.codes[row * columns / 2 + column / 2] >> ((column & 1) * 4)) & 0xF;
            let group = column / 32;
            let zero = (weights.zero_points[group * weights.scale_stride / 2 + row / 2] >> ((row & 1) * 4)) & 0xF;
            weights.scales[group * weights.scale_stride + row].to_f64() * (code as f64 - zero as f64)
        })
        .collect()
}

fn check_stage_rows(
    columns: usize,
    block: usize,
    with_post_signs: bool,
) {
    let context = create_context::<Metal>();
    let mut rng = StdRng::seed_from_u64(columns as u64 + block as u64);
    let (total_rows, row_offset, rows) = (96usize, 32usize, 40usize);
    let weights = random_w4(&mut rng, total_rows, columns);
    let pre = signs(&mut rng, columns);
    let post = signs(&mut rng, columns);
    let staged_row_stride = columns;
    let row_scale_stride = 32usize; // fp16 [rows, 1] surface: 64-byte rows
    let codes_buffer = upload(&context, &weights.codes);
    let scales_buffer = upload(&context, &weights.scales);
    let zero_buffer = upload(&context, &weights.zero_points);
    let mut staged = zeros(&context, rows * staged_row_stride);
    let mut row_scales = zeros(&context, rows * row_scale_stride * 2);
    if with_post_signs {
        let kernel = AneStageDownRowsMetalKernel::new(&context, (columns / 4096) as u32).unwrap();
        let (pre_buffer, post_buffer) = (upload(&context, &sign_bits(&pre)), upload(&context, &sign_bits(&post)));
        run(&context, |encoding| {
            kernel.encode(
                &codes_buffer,
                &scales_buffer,
                &zero_buffer,
                &pre_buffer,
                &post_buffer,
                &mut staged,
                &mut row_scales,
                row_offset as u32,
                rows as u32,
                weights.scale_stride as u32,
                staged_row_stride as u32,
                row_scale_stride as u32,
                encoding,
            )
        });
    } else {
        let kernel = AneStageUpRowsMetalKernel::new(&context).unwrap();
        let pre_buffer = upload(&context, &pre);
        run(&context, |encoding| {
            kernel.encode(
                &codes_buffer,
                &scales_buffer,
                &zero_buffer,
                &pre_buffer,
                &mut staged,
                &mut row_scales,
                row_offset as u32,
                rows as u32,
                weights.scale_stride as u32,
                staged_row_stride as u32,
                row_scale_stride as u32,
                encoding,
            )
        });
    }
    for row in 0..rows {
        let mut values = dequantized_row(&weights, row_offset + row, columns);
        hadamard(&mut values, 32);
        values.iter_mut().zip(&pre).for_each(|(value, sign)| *value *= *sign as f64);
        hadamard(&mut values, block);
        if with_post_signs {
            values.iter_mut().zip(&post).for_each(|(value, sign)| *value *= *sign as f64);
        }
        let (expected, scale) = quantize_row(&values);
        let actual = &read::<i8>(&staged)[row * staged_row_stride..row * staged_row_stride + columns];
        assert_codes_close(actual, &expected, "stage");
        let actual_scale = read::<f16>(&row_scales)[row * row_scale_stride].to_f64() / 128.0;
        assert!((actual_scale - scale).abs() <= scale * 1e-3, "row scale {actual_scale} vs {scale}");
    }
}

#[uzu_test]
fn ane_stage_rows_up_projection_layout() {
    check_stage_rows(4096, 4096, false);
}

#[uzu_test]
fn ane_stage_rows_down_projection_layout() {
    check_stage_rows(12288, 512, true);
}

/// Stages the 9B layer's up (2 x 12288 rows x 4096 columns, block 4096) and down (4096 rows x 12288
/// columns, block 512) weights `LAYERS` times in one command buffer; reports ms and GB/s per layer.
#[uzu_test]
#[ignore]
fn ane_stage_rows_perf() {
    use crate::backends::common::CommandBufferCompleted;

    const LAYERS: usize = 8;
    let context = create_context::<Metal>();
    let mut rng = StdRng::seed_from_u64(7);
    let (intermediate, hidden, up_columns, down_columns) = (12288usize, 4096usize, 4096usize, 12288usize);
    let up = random_w4(&mut rng, 2 * intermediate, up_columns);
    let down = random_w4(&mut rng, hidden, down_columns);
    let up_codes = upload(&context, &up.codes);
    let up_scales = upload(&context, &up.scales);
    let up_zeros = upload(&context, &up.zero_points);
    let down_codes = upload(&context, &down.codes);
    let down_scales = upload(&context, &down.scales);
    let down_zeros = upload(&context, &down.zero_points);
    let up_signs = upload(&context, &signs(&mut rng, up_columns));
    let down_pre = upload(&context, &sign_bits(&signs(&mut rng, down_columns)));
    let down_post = upload(&context, &sign_bits(&signs(&mut rng, down_columns)));
    let mut wu = zeros(&context, intermediate * up_columns);
    let mut wg = zeros(&context, intermediate * up_columns);
    let mut wd = zeros(&context, hidden * down_columns);
    let mut su = zeros(&context, intermediate * 64);
    let mut sg = zeros(&context, intermediate * 64);
    let mut sd = zeros(&context, hidden * 64);
    let stage_up = AneStageUpRowsMetalKernel::new(&context).unwrap();
    let stage_down = AneStageDownRowsMetalKernel::new(&context, 3).unwrap();
    let up_bytes = up.codes.len() + up.scales.len() * 2 + up.zero_points.len() + 2 * intermediate * up_columns;
    let down_bytes = down.codes.len() + down.scales.len() * 2 + down.zero_points.len() + hidden * down_columns;
    let mut totals = [f64::MAX; 3];
    for iteration in 0..15 {
        let (run_up, run_down) = [(true, false), (false, true), (true, true)][iteration % 3];
        let mut encoding = context.create_command_buffer(None, None).unwrap();
        for _ in 0..LAYERS {
            for (staged, scales, offset) in [(&mut wu, &mut su, 0u32), (&mut wg, &mut sg, intermediate as u32)] {
                if !run_up {
                    break;
                }
                stage_up.encode(
                    &up_codes,
                    &up_scales,
                    &up_zeros,
                    &up_signs,
                    staged,
                    scales,
                    offset,
                    intermediate as u32,
                    up.scale_stride as u32,
                    up_columns as u32,
                    32,
                    &mut encoding,
                );
            }
            if !run_down {
                continue;
            }
            stage_down.encode(
                &down_codes,
                &down_scales,
                &down_zeros,
                &down_pre,
                &down_post,
                &mut wd,
                &mut sd,
                0,
                hidden as u32,
                down.scale_stride as u32,
                down_columns as u32,
                32,
                &mut encoding,
            );
        }
        let completed = block_on(encoding.end_encoding().submit().wait_until_completed()).unwrap();
        let ms = completed.gpu_execution_time().unwrap().as_secs_f64() * 1e3 / LAYERS as f64;
        totals[iteration % 3] = totals[iteration % 3].min(ms);
    }
    for (name, ms, bytes) in
        [("up", totals[0], up_bytes), ("down", totals[1], down_bytes), ("both", totals[2], up_bytes + down_bytes)]
    {
        let gigabytes = bytes as f64 / 1e9;
        println!(
            "ane_stage_rows_perf {name}: {ms:.3} ms/layer, {gigabytes:.3} GB/layer, {:.1} GB/s",
            gigabytes / (ms / 1e3)
        );
    }
}
