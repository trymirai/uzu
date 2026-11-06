use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::qkv_norm::QKVNormParams, kernel::AttentionPrepareKernel,
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
    enabled: [bool; 3],
    affine: [bool; 3],
}

fn run_prepare<B: Backend, const DIM: usize>(
    num_q_heads: u32,
    normalization: NormCase,
    fused: bool,
    rope: bool,
) -> PrepareOutput {
    const CACHE_TOKENS: usize = 64;
    const SUFFIX: usize = 2;
    const ROW_PADDING: usize = 5;
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
        for head in (1 - num_q_heads) as usize..3 {
            let mut values = (0..DIM)
                .map(|i| bf16::from_f32(((token * 3 * DIM + head * DIM + i) as f32 * 0.37).sin()))
                .collect::<Vec<_>>();
            if !fused && normalization.enabled[head] {
                let sum = values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>();
                let inverse_rms = (sum / DIM as f32 + norm_params.epsilon[head]).sqrt().recip();
                for (i, value) in values.iter_mut().enumerate() {
                    let normalized = value.to_f32() * inverse_rms;
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
    let cosines = create_buffer_with_data::<B, f32>(
        &context,
        &(0..SUFFIX as u32 * rope_dim).map(|i| (i as f32 * 0.07).cos()).collect::<Vec<_>>(),
    );
    let sines = create_buffer_with_data::<B, f32>(
        &context,
        &(0..SUFFIX as u32 * rope_dim).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>(),
    );
    let affine = normalization.affine.map(|affine| affine && fused);
    let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        DIM as u32,
        rope,
        1,
        layout.backend_opaque_layout,
        true,
        affine[0],
        affine[1],
        affine[2],
        false,
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
        if fused {
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
    fn check<const DIM: usize>() {
        for queries in [0, 1] {
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
                            let separate = run_prepare::<B, DIM>(queries, case, false, rope);
                            let fused = run_prepare::<B, DIM>(queries, case, true, rope);
                            assert_eq!(
                                separate, fused,
                                "fused norm mismatch D{DIM} queries={queries} full_layer={full_layer} rope={rope}"
                            );
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
