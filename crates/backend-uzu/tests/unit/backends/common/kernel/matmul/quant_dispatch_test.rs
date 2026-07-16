#![cfg(metal_backend)]

use std::{
    error::Error as StdError,
    fmt::{Debug, Display},
};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rstest::rstest;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{ActivationScaleStat, QuantizationMethod, gemm::GemmDTransform},
            kernel::{
                Kernels,
                matmul::{MatmulDOps, MatmulError, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
    tests::{
        helpers::allocation_to_vec,
        matmul::{QuantBuffers, QuantInput, quant_arguments, run_quant_cpu, run_quant_metal},
    },
};

fn check_tolerance(
    expected: f32,
    actual: f32,
    rel_tol: f64,
    abs_tol: f64,
) -> bool {
    let diff = (expected - actual).abs() as f64;
    let tol = abs_tol.max(expected.abs() as f64 * rel_tol);
    diff <= tol
}

fn assert_parity<T: ArrayElement + Float + Debug + Display>(
    label: &str,
    reference: &[T],
    actual: &[T],
    rel_tol: f64,
    abs_tol: f64,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    let mut errors = 0usize;
    for (i, (&exp, &got)) in reference.iter().zip(actual.iter()).enumerate() {
        let exp_f32 = exp.to_f32().unwrap();
        let got_f32 = got.to_f32().unwrap();
        if !check_tolerance(exp_f32, got_f32, rel_tol, abs_tol) {
            if errors < 5 {
                eprintln!("  {label} idx={i} expected={exp_f32} got={got_f32} diff={}", (exp_f32 - got_f32).abs());
            }
            errors += 1;
        }
    }
    assert_eq!(errors, 0, "{label}: {errors} mismatches");
}

fn run_parity<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    rel_tol: f64,
    abs_tol: f64,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 0);
    let unified = run_quant_metal::<T>(&context, &input, Some(GemmDispatchPath::Simdgroup));
    let reference = run_quant_cpu::<T>(&input);

    assert_parity::<T>(
        &format!(
            "m={m} k={k} n={n} gs={group_size} bits={bits} method={quant_method:?} dtype={}",
            std::any::type_name::<T>()
        ),
        &reference,
        &unified,
        rel_tol,
        abs_tol,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs16_4bit_zp_prefill(64, 256, 64, 16, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_4bit_mlx_prefill(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx_prefill(64, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs128_4bit_mlx_prefill(64, 256, 64, 128, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_4bit_symmetric_prefill(64, 256, 64, 32, 4, QuantizationMethod::ScaleSymmetric)]
#[case::gs32_4bit_mlx_decode(8, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs16_4bit_zp_decode(8, 256, 64, 16, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_4bit_zp_decode(8, 256, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_unaligned_n(64, 256, 96, 32, 4, QuantizationMethod::ScaleBias)]
fn parity_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, 0.05, 0.4);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_8bit_mlx_prefill(64, 256, 64, 32, 8, QuantizationMethod::ScaleBias)]
#[case::gs64_8bit_zp_prefill(64, 256, 64, 64, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs128_8bit_zp_prefill(64, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_8bit_symmetric_prefill(64, 256, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_bf16_8bit_splitk(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, 0.05, 1.0);
}

fn run_parity_gemv<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    rel_tol: f64,
    abs_tol: f64,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 0);
    let gemv = run_quant_metal::<T>(&context, &input, None);
    let reference = run_quant_cpu::<T>(&input);
    assert_parity::<T>(
        &format!(
            "gemv m={m} k={k} n={n} gs={group_size} bits={bits} method={quant_method:?} dtype={}",
            std::any::type_name::<T>()
        ),
        &reference,
        &gemv,
        rel_tol,
        abs_tol,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1_gs32_4bit_mlx(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::m1_gs64_4bit_mlx(1, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::m1_gs128_4bit_mlx(1, 256, 64, 128, 4, QuantizationMethod::ScaleBias)]
#[case::m1_gs32_4bit_zp(1, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_gs64_8bit_zp(1, 256, 64, 64, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_gs128_8bit_mlx(1, 256, 64, 128, 8, QuantizationMethod::ScaleBias)]
#[case::m1_gs32_4bit_sym(1, 256, 64, 32, 4, QuantizationMethod::ScaleSymmetric)]
#[case::m1_gs64_8bit_sym(1, 256, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
#[case::m2_gs32_4bit_mlx(2, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::m4_gs32_4bit_zp(4, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
fn parity_gemv_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity_gemv::<bf16>(m, k, n, gs, bits, method, 0.05, 0.4);
}

#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_with_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.5 + 0.1 * (j % 5) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let bias_pp_buf = crate::tests::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: Some(&bias_pp_buf),
        rht_factors: None,
    };
    matmul.gemm.encode_dispatch_path(args, GemmDispatchPath::Simdgroup, &mut encoder).expect("encode quant with bias");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);

    let mut reference = run_quant_cpu::<bf16>(&input);
    for i in 0..(input.m as usize) {
        for j in 0..(input.n as usize) {
            let idx = i * input.n as usize + j;
            reference[idx] = bf16::from_f32(reference[idx].to_f32() + bias_f32[j]);
        }
    }
    assert_parity::<bf16>("with_bias", &reference, &actual, 0.05, 0.4);
}

#[uzu_test]
fn parity_bf16_gemv_qmv_fused_scale_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    let scale = 2.0_f32;
    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.25 + 0.1 * (j % 7) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let raw = run_quant_cpu::<bf16>(&input);
    let reference: Vec<bf16> = (0..input.m as usize)
        .flat_map(|i| {
            (0..input.n as usize).map(move |j| {
                let idx = i * input.n as usize + j;
                (idx, j)
            })
        })
        .map(|(idx, j)| bf16::from_f32(scale * raw[idx].to_f32() + bias_f32[j]))
        .collect();

    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let bias_buf = crate::tests::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = MatmulDOps {
        ab_scale: scale,
        accumulate: false,
        bias: Some(&bias_buf),
        rht_factors: None,
    };
    matmul.encode(args, &mut encoder).expect("encode quant gemv with scale+bias");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);

    assert_parity::<bf16>("gemv_qmv_scale_bias", &reference, &actual, 0.05, 0.4);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs64_4bit(1, 96, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_zp(2, 96, 64, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs128_8bit(1, 192, 64, 128, 8, QuantizationMethod::ScaleBias)]
fn parity_gemv_partial_group_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // k is not a multiple of group_size, so the per-row scale/zp stride must use
    // ceil(k / group_size) groups.
    run_parity_gemv::<bf16>(m, k, n, gs, bits, method, 0.05, 0.6);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::n12_gs32_4bit(1, 256, 12, 32, 4, QuantizationMethod::ScaleBias)]
#[case::n20_gs64_4bit_zp(2, 256, 20, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::n36_gs32_8bit(1, 256, 36, 32, 8, QuantizationMethod::ScaleBias)]
fn parity_gemv_unaligned_width_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // n is not a multiple of 8; the output-tail clamp must cover the partial block.
    run_parity_gemv::<bf16>(m, k, n, gs, bits, method, 0.05, 0.6);
}

#[uzu_test]
fn parity_bf16_gemv_quant_rht_with_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);
    let rht: Vec<i32> = (0..input.n as usize)
        .map(|i| {
            if i % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.3 + 0.05 * (j % 4) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // CPU reference applies bias before RHT (matmul loop adds bias, then the
    // Hadamard pass transforms the result) — the same order GEMV uses.
    let cpu_context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut cpu_buffers = QuantBuffers::<Cpu, bf16>::allocate(&cpu_context, &input);
    let cpu_rht = crate::tests::helpers::alloc_allocation_with_data::<Cpu, i32>(&cpu_context, &rht);
    let cpu_bias = crate::tests::helpers::alloc_allocation_with_data::<Cpu, bf16>(&cpu_context, &bias_t);
    let mut cpu_matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &cpu_context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut cpu_encoder = Encoder::<Cpu>::new(&cpu_context).expect("cpu encoder");
    let mut cpu_args = quant_arguments(&mut cpu_buffers, &input);
    cpu_args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: Some(&cpu_bias),
        rht_factors: Some(&cpu_rht),
    };
    cpu_matmul.encode(cpu_args, &mut cpu_encoder).expect("cpu encode quant+rht+bias");
    cpu_encoder.end_encoding().submit().wait_until_completed().unwrap();
    let reference = allocation_to_vec::<Cpu, bf16>(&cpu_buffers.y);

    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let metal_rht = crate::tests::helpers::alloc_allocation_with_data::<Metal, i32>(&context, &rht);
    let metal_bias = crate::tests::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: Some(&metal_bias),
        rht_factors: Some(&metal_rht),
    };
    matmul.encode(args, &mut encoder).expect("encode quant gemv with rht+bias");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);

    assert_parity::<bf16>("gemv_quant_rht_bias", &reference, &actual, 0.05, 0.6);
}

#[uzu_test]
fn parity_bf16_gemv_quant_rht() {
    let context = MetalContext::new().expect("Metal context");
    // n % 32 == 0 so the output RHT covers whole 32-element blocks; m = 1 routes to GEMV.
    let input = QuantInput::<bf16>::new(1, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);
    let rht: Vec<i32> = (0..input.n as usize)
        .map(|i| {
            if i % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();

    // CPU reference applies the RHT through the CPU matmul kernel.
    let cpu_context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut cpu_buffers = QuantBuffers::<Cpu, bf16>::allocate(&cpu_context, &input);
    let cpu_rht = crate::tests::helpers::alloc_allocation_with_data::<Cpu, i32>(&cpu_context, &rht);
    let mut cpu_matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &cpu_context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut cpu_encoder = Encoder::<Cpu>::new(&cpu_context).expect("cpu encoder");
    let mut cpu_args = quant_arguments(&mut cpu_buffers, &input);
    cpu_args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: None,
        rht_factors: Some(&cpu_rht),
    };
    cpu_matmul.encode(cpu_args, &mut cpu_encoder).expect("cpu encode quant+rht");
    cpu_encoder.end_encoding().submit().wait_until_completed().unwrap();
    let reference = allocation_to_vec::<Cpu, bf16>(&cpu_buffers.y);

    // Metal GEMV: m = 1 quant routes to GEMV, RHT selects the 8-simdgroup (32-row) layout.
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let metal_rht = crate::tests::helpers::alloc_allocation_with_data::<Metal, i32>(&context, &rht);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: None,
        rht_factors: Some(&metal_rht),
    };
    matmul.encode(args, &mut encoder).expect("encode quant gemv with rht");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);

    assert_parity::<bf16>("gemv_quant_rht", &reference, &actual, 0.05, 0.6);
}

#[uzu_test]
fn quant_gemm_accumulate_returns_unsupported_dop() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: true,
        bias: None,
        rht_factors: None,
    };
    let result = matmul.encode(args, &mut encoder);

    let err = result.expect_err("expected error");
    let matmul: &MatmulError<Metal> = (&err as &dyn StdError)
        .source()
        .and_then(|s| s.downcast_ref::<MatmulError<Metal>>())
        .expect("expected MatmulError source");
    assert!(
        matches!(
            matmul,
            MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                ..
            }
        ),
        "got {matmul:?}"
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_4bit_mlx(128, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx(128, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs128_4bit_mlx(128, 256, 64, 128, 4, QuantizationMethod::ScaleBias)]
#[case::small_m_4bit_mlx(24, 256, 512, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_4bit_zp(128, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_8bit_zp(128, 256, 64, 64, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs128_8bit_zp(128, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint)]
fn mxu_quant_parity_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let input = QuantInput::<bf16>::new(m, k, n, gs, bits, method, 0);
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmDispatchPath::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(
        &format!("MXU m={m} k={k} n={n} gs={gs} bits={bits} method={method:?}"),
        &reference,
        &actual,
        0.05,
        0.5,
    );
}

/// A8W8 (int8 activation x int8 weight) GEMM parity against the CPU A8W8
/// reference. The int8 `matmul2d` path only runs on M5+/A19, so this self-skips
/// elsewhere. Both sides quantize A identically (~1 LSB apart from Metal
/// fast-math division) and read the symmetric weights excess-128, so a modest
/// tolerance covers quantization + int32-vs-float accumulation noise while still
/// catching a wrong MXU int8 fragment layout (which corrupts results grossly).
#[rstest]
#[test_attr(uzu_test)]
#[case::gs32(128, 256, 128, 32)]
#[case::gs64(64, 512, 64, 64)]
#[case::gs128(128, 256, 256, 128)]
#[case::unaligned(96, 256, 100, 32)]
fn a8w8_mxu_parity_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let input = QuantInput::<bf16>::new(m, k, n, gs, 8, QuantizationMethod::ScaleSymmetric, 0)
        .with_prepared_a(ActivationScaleStat::AbsMax);
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmDispatchPath::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(&format!("A8W8 m={m} k={k} n={n} gs={gs}"), &reference, &actual, 0.06, 0.6);
}
