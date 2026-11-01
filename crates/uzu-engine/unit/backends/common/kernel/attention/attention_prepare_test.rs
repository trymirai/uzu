use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::qkv_norm::QKVNormParams, kernel::AttentionPrepareKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
};

#[derive(Debug, PartialEq)]
struct Int8Output {
    codes: [Vec<u8>; 3],
    scales: [Vec<bf16>; 3],
}

#[derive(Clone, Copy)]
struct NormCase {
    full_layer: bool,
    enabled: [bool; 3],
    affine: [bool; 3],
}

fn run_int8<B: Backend, const DIM: usize>(
    num_q_heads: u32,
    normalization: Option<NormCase>,
    fused: bool,
    rope: bool,
    use_mxu: bool,
) -> Int8Output {
    const CACHE_TOKENS: usize = 64;
    const SUFFIX: usize = 2;
    const ROW_PADDING: usize = 5;
    let code_bytes = CACHE_TOKENS * DIM;
    let norm_params = QKVNormParams {
        enabled: normalization.map_or([0; 3], |case| case.enabled.map(u32::from)),
        full_layer: [u32::from(normalization.is_some_and(|case| case.full_layer)); 3],
        epsilon: [1e-6, 1e-5, 1e-4],
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
            if let Some(case) = normalization.filter(|case| !fused && case.enabled[head]) {
                // Independent reference with the original materialized BF16
                // norm, including its intermediate rounding before RoPE.
                let sum = values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
                let inverse_rms = (sum / DIM as f32 + norm_params.epsilon[head]).sqrt().recip();
                for (i, value) in values.iter_mut().enumerate() {
                    let normalized = value.to_f32() * inverse_rms;
                    let scale = weights[head][i] + norm_params.scale_offset[head];
                    *value = if !case.affine[head] {
                        bf16::from_f32(normalized)
                    } else if case.full_layer {
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
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &rows);
    let norm_scales = weights.map(|weights| create_buffer_with_data::<B, f32>(&context, &weights));
    let mut q_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; SUFFIX * DIM]);
    let mut q_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; SUFFIX]);
    let mut k_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; code_bytes]);
    let mut v_codes = create_buffer_with_data::<B, u8>(&context, &vec![0; code_bytes]);
    let mut k_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let mut v_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let rope_dim = DIM as u32 / 2;
    let cosines = create_buffer_with_data::<B, f32>(
        &context,
        &(0..SUFFIX as u32 * rope_dim).map(|i| (i as f32 * 0.07).cos()).collect::<Vec<_>>(),
    );
    let sines = create_buffer_with_data::<B, f32>(
        &context,
        &(0..SUFFIX as u32 * rope_dim).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>(),
    );
    let fused_norm = normalization.is_some() && fused;
    let affine = normalization.map_or([false; 3], |case| case.affine.map(|affine| affine && fused));
    let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        DIM as u32,
        rope,
        1,
        use_mxu,
        true,
        affine[0],
        affine[1],
        affine[2],
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &qkvg,
        &mut q_codes,
        &mut q_scales,
        Some(&mut k_codes),
        Some(&mut k_scales),
        Some(&mut v_codes),
        Some(&mut v_scales),
        rope.then_some(&cosines),
        rope.then_some(&sines),
        affine[0].then_some(&norm_scales[0]),
        affine[1].then_some(&norm_scales[1]),
        affine[2].then_some(&norm_scales[2]),
        if fused_norm {
            norm_params
        } else {
            QKVNormParams::default()
        },
        num_q_heads,
        rope.then_some(rope_dim),
        33,
        (num_q_heads + 2) * DIM as u32 + ROW_PADDING as u32,
        SUFFIX as u32,
        num_q_heads * SUFFIX as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    Int8Output {
        codes: [&q_codes, &k_codes, &v_codes].map(|buffer| buffer_to_vec::<B, u8>(&context, buffer)),
        scales: [&q_scales, &k_scales, &v_scales].map(|buffer| buffer_to_vec::<B, bf16>(&context, buffer)),
    }
}

#[uzu_test]
fn test_attention_prepare() {
    fn check<const DIM: usize>() {
        for (num_q_heads, use_mxu) in [(0, false), (0, true), (1, false), (1, true)] {
            let cpu = run_int8::<Cpu, DIM>(num_q_heads, None, false, false, use_mxu);
            for_each_non_cpu_backend!(|B| {
                assert_eq!(
                    cpu,
                    run_int8::<B, DIM>(num_q_heads, None, false, false, use_mxu),
                    "INT8 prepare mismatch on {} D{DIM}",
                    std::any::type_name::<B>()
                );
            });
        }
    }
    check::<64>();
    check::<128>();
    check::<256>();
    check::<512>();
}

#[uzu_test]
fn test_attention_prepare_fused_norm() {
    fn check<const DIM: usize>() {
        for (queries, use_mxu) in [(0, false), (0, true), (1, false), (1, true)] {
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
                        for_each_non_cpu_backend!(|B| {
                            let separate = run_int8::<B, DIM>(queries, Some(case), false, rope, use_mxu);
                            let fused = run_int8::<B, DIM>(queries, Some(case), true, rope, use_mxu);
                            assert_eq!(
                                separate, fused,
                                "fused norm mismatch D{DIM} queries={queries} full_layer={full_layer} rope={rope}"
                            );
                        });
                        let separate = run_int8::<Cpu, DIM>(queries, Some(case), false, rope, use_mxu);
                        let fused = run_int8::<Cpu, DIM>(queries, Some(case), true, rope, use_mxu);
                        assert_eq!(separate, fused, "CPU fused norm mismatch D{DIM}");
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

#[uzu_test]
fn test_attention_prepare_rounding_boundaries() {
    const DIM: usize = 256;
    // Two Hadamard basis inputs produce rows whose maximum is 127 * scale
    // and whose other magnitude is a quantization half-integer. Perturbing
    // one input also exercises both sides of each rounding boundary.
    let mut input = Vec::new();
    for exponent in [-80, -10, 0, 20, 80] {
        for mantissa in [128, 129, 131, 159, 191, 255] {
            let scale = mantissa as f32 / 128.0 * 2.0_f32.powi(exponent);
            for code in [-126, -31, -1, 0, 1, 30, 125] {
                let half = code as f32 + 0.5;
                let a = (127.0 + half) * 8.0 * scale;
                let b = (127.0 - half) * 8.0 * scale;
                for a in [a.next_down(), a, a.next_up()] {
                    let mut row = [0.0; DIM];
                    // Undo the input rotation signs for channels zero and one.
                    row[0] = -a;
                    row[1] = b;
                    input.extend(row);
                }
            }
        }
    }
    fn run<B: Backend>(input: &[f32]) -> (Vec<i8>, Vec<bf16>) {
        let context = block_on(B::Context::new()).unwrap();
        let rows = input.len() / DIM;
        let source = create_buffer_with_data::<B, f32>(&context, input);
        let mut codes = create_buffer_with_data::<B, i8>(&context, &vec![0; input.len()]);
        let mut scales = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; rows]);
        let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            &context,
            DataType::F32,
            DataType::F32,
            DIM as u32,
            false,
            1,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        let mut command = context.create_command_buffer(None, None).unwrap();
        kernel.encode(
            &source,
            &mut codes,
            &mut scales,
            None::<&mut B::GlobalGpuBuffer>,
            None::<&mut B::GlobalGpuBuffer>,
            None::<&mut B::GlobalGpuBuffer>,
            None::<&mut B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            Default::default(),
            1,
            None,
            0,
            DIM as u32,
            rows as u32,
            rows as u32,
            &mut command,
        );
        block_on(command.end_encoding().submit().wait_until_completed()).unwrap();
        (buffer_to_vec::<B, i8>(&context, &codes), buffer_to_vec::<B, bf16>(&context, &scales))
    }
    let expected = run::<Cpu>(&input);
    for_each_non_cpu_backend!(|B| {
        assert_eq!(expected, run::<B>(&input), "INT8 rounding boundaries on {}", std::any::type_name::<B>());
    });
}
