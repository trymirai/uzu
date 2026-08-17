#![cfg(backend = "metal")]

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
            gpu_types::{QuantizationMethod, gemm::GemmDTransform},
            kernel::{
                Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{MatmulDOps, MatmulError, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{GemmEngine, Metal, MetalContext},
    },
    tests::{
        helpers::allocation_to_vec,
        matmul::{
            QuantBuffers, QuantInput,
            harness::TestDispatch,
            quant::{run_quant_cpu, run_quant_metal},
            quant_arguments, quant_b_variant,
        },
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
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    dispatch: TestDispatch,
    label: &str,
    rel_tol: f64,
    abs_tol: f64,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 0);
    let unified = run_quant_metal::<T>(&context, &input, dispatch);
    let reference = run_quant_cpu::<T>(&input);

    assert_parity::<T>(
        &format!(
            "{label} m={m} k={k} n={n} gs={group_size} bits={bits} method={quant_method:?} dtype={}",
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
#[case::gs32_narrow_n_tail_zp(64, 256, 72, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_narrow_n_tail_mlx(64, 256, 72, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_narrow_n_tail_sym(64, 256, 40, 32, 4, QuantizationMethod::ScaleSymmetric)]
#[case::gs16_k_tail_mlx(64, 272, 64, 16, 4, QuantizationMethod::ScaleBias)]
fn parity_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, Some(GemmEngine::Simdgroup), "gemm", 0.05, 0.4);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gs32_8bit_mlx_prefill(64, 256, 64, 32, 8, QuantizationMethod::ScaleBias)]
#[case::gs64_8bit_zp_prefill(64, 256, 64, 64, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs128_8bit_zp_prefill(64, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs64_8bit_symmetric_prefill(64, 256, 64, 64, 8, QuantizationMethod::ScaleSymmetric)]
fn parity_bf16_8bit_splitk(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, Some(GemmEngine::Simdgroup), "gemm", 0.05, 1.0);
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
#[case::m8_gs32_4bit_zp(8, 256, 64, 32, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::m8_gs32_4bit_mlx(8, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
fn parity_gemv_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.4);
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
        bias: Some(&bias_pp_buf),
        ..MatmulDOps::none()
    };
    matmul.gemm.encode_with_engine(args, GemmEngine::Simdgroup, &mut encoder).expect("encode quant with bias");
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
        bias: Some(&bias_buf),
        ..MatmulDOps::none()
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
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // k is not a multiple of group_size, so the per-row scale/zp stride must use
    // ceil(k / group_size) groups.
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.6);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::n12_gs32_4bit(1, 256, 12, 32, 4, QuantizationMethod::ScaleBias)]
#[case::n20_gs64_4bit_zp(2, 256, 20, 64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::n36_gs32_8bit(1, 256, 36, 32, 8, QuantizationMethod::ScaleBias)]
fn parity_gemv_unaligned_width_bf16(
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    // n is not a multiple of 8; the output-tail clamp must cover the partial block.
    run_parity::<bf16>(m, k, n, gs, bits, method, None, "gemv", 0.05, 0.6);
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
        bias: Some(&cpu_bias),
        rht_factors: Some(&cpu_rht),
        ..MatmulDOps::none()
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
        bias: Some(&metal_bias),
        rht_factors: Some(&metal_rht),
        ..MatmulDOps::none()
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
        rht_factors: Some(&cpu_rht),
        ..MatmulDOps::none()
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
        rht_factors: Some(&metal_rht),
        ..MatmulDOps::none()
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
        accumulate: true,
        ..MatmulDOps::none()
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
    #[case] m: u32,
    #[case] k: u32,
    #[case] n: u32,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let input = QuantInput::<bf16>::new(m, k, n, gs, bits, method, 0);
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(
        &format!("MXU m={m} k={k} n={n} gs={gs} bits={bits} method={method:?}"),
        &reference,
        &actual,
        0.05,
        0.5,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::sym_w4_gs32(16u32, 32u32, 4u32, QuantizationMethod::ScaleSymmetric)]
#[case::sym_w8_gs64(16u32, 64u32, 8u32, QuantizationMethod::ScaleSymmetric)]
#[case::sym_w4_gs128(32u32, 128u32, 4u32, QuantizationMethod::ScaleSymmetric)]
#[case::bias_w4_gs32(16u32, 32u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::bias_w8_gs64(16u32, 64u32, 8u32, QuantizationMethod::ScaleBias)]
#[case::bias_w4_gs128(32u32, 128u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::zp_w4_gs32(16u32, 32u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::zp_w8_gs64(16u32, 64u32, 8u32, QuantizationMethod::ScaleZeroPoint)]
#[case::zp_w4_gs128(32u32, 128u32, 4u32, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_bias_w4_gs64(1u32, 64u32, 4u32, QuantizationMethod::ScaleBias)]
#[case::m1_zp_w8_gs32(1u32, 32u32, 8u32, QuantizationMethod::ScaleZeroPoint)]
#[case::m1_sym_w4_gs128(1u32, 128u32, 4u32, QuantizationMethod::ScaleSymmetric)]
fn a8w_mxu_parity_bf16(
    #[case] m: u32,
    #[case] weight_gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let (k, n) = (256u32, 128u32);
    let input = QuantInput::<bf16>::new(m, k, n, weight_gs, bits, method, 0).with_prepared_a(
        ACTIVATION_SCALE_GROUP_SIZE,
        (method != QuantizationMethod::ScaleSymmetric).then_some(weight_gs),
    );
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(
        &format!("A8W{bits} MXU m={m} weight_gs={weight_gs} method={method:?}"),
        &reference,
        &actual,
        0.08,
        0.8,
    );
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1u32)]
#[case::m16(16u32)]
#[case::m33(33u32)]
fn a8w_independent_activation_group_parity_bf16(#[case] m: u32) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    for (activation_group_size, weight_group_size) in
        [(32u32, 32u32), (32, 64), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)]
    {
        for (bits, method) in [
            (4, QuantizationMethod::ScaleSymmetric),
            (8, QuantizationMethod::ScaleSymmetric),
            (8, QuantizationMethod::ScaleBias),
            (4, QuantizationMethod::ScaleZeroPoint),
        ] {
            let input = QuantInput::<bf16>::new(m, 256, 72, weight_group_size, bits, method, 0).with_prepared_a(
                activation_group_size,
                (method != QuantizationMethod::ScaleSymmetric).then_some(activation_group_size.min(weight_group_size)),
            );
            let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmEngine::Mxu));
            let reference = run_quant_cpu::<bf16>(&input);
            assert_parity::<bf16>(
                &format!("A8W{bits} act{activation_group_size} weight{weight_group_size} m={m} method={method:?}"),
                &reference,
                &actual,
                0.08,
                0.8,
            );
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1)]
#[case::m16(16)]
#[case::m33(33)]
fn a8w4_zero_point_tail_parity(#[case] m: u32) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let input = QuantInput::<bf16>::new(m, 256, 72, 32, 4, QuantizationMethod::ScaleZeroPoint, 0)
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, Some(32));
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>(&format!("A8W4 ZP N-tail m={m}"), &reference, &actual, 0.08, 0.8);
}

#[uzu_test]
fn a8w4_zero_point_tail_signed_codes_parity() {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }
    let input = QuantInput::<bf16>::new(33, 256, 72, 32, 4, QuantizationMethod::ScaleZeroPoint, 0)
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, Some(32))
        .with_signed_weight_codes();
    let actual = run_quant_metal::<bf16>(&context, &input, Some(GemmEngine::Mxu));
    let reference = run_quant_cpu::<bf16>(&input);
    assert_parity::<bf16>("A8W4 ZP signed-code N-tail", &reference, &actual, 0.08, 0.8);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w4_sym(4u32, QuantizationMethod::ScaleSymmetric, 256u32)]
#[case::w4_bias(4u32, QuantizationMethod::ScaleBias, 256u32)]
#[case::w4_zp(4u32, QuantizationMethod::ScaleZeroPoint, 256u32)]
#[case::w8_sym(8u32, QuantizationMethod::ScaleSymmetric, 256u32)]
#[case::w8_bias(8u32, QuantizationMethod::ScaleBias, 256u32)]
#[case::w8_zp(8u32, QuantizationMethod::ScaleZeroPoint, 256u32)]
#[case::w8_zp_tail(8u32, QuantizationMethod::ScaleZeroPoint, 224u32)]
fn signed_weights_full_precision_activations_parity_bf16(
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[case] k: u32,
) {
    let context = MetalContext::new().expect("Metal context");
    let (m, n, group_size) = (2u32, 128u32, 32u32);
    let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0).with_signed_weight_codes();
    let reference_input = QuantInput::<bf16>::new(m, k, n, group_size, bits, method, 0);
    let reference = run_quant_cpu::<bf16>(&reference_input);

    for (label, dispatch) in [("gemv", None), ("simdgroup", Some(GemmEngine::Simdgroup))] {
        let actual = run_quant_metal::<bf16>(&context, &input, dispatch);
        assert_parity::<bf16>(
            &format!("signed weights FP-A {label} bits={bits} method={method:?}"),
            &reference,
            &actual,
            0.05,
            0.5,
        );
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::w4_bias(4u32, false)]
#[case::w8_bias(8u32, false)]
#[case::w4_rht_bias(4u32, true)]
#[case::w8_rht_bias(8u32, true)]
fn a8w_mxu_output_bias_parity_bf16(
    #[case] bits: u32,
    #[case] with_output_hadamard: bool,
) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    let (m, k, n) = (
        16u32,
        256u32,
        if with_output_hadamard {
            64u32
        } else {
            70u32
        },
    );
    let input = QuantInput::<bf16>::new(m, k, n, 32, bits, QuantizationMethod::ScaleSymmetric, 0)
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);
    let output_bias: Vec<bf16> = (0..n).map(|column| bf16::from_f32(0.25 + 0.05 * (column % 7) as f32)).collect();
    let output_hadamard_factors: Option<Vec<i32>> = with_output_hadamard.then(|| {
        (0..n)
            .map(|column| {
                if column.is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect()
    });

    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU context");
    let mut cpu_buffers = QuantBuffers::<Cpu, bf16>::allocate(&cpu_context, &input);
    let cpu_output_hadamard_factors = output_hadamard_factors
        .as_ref()
        .map(|factors| crate::tests::helpers::alloc_allocation_with_data::<Cpu, i32>(&cpu_context, factors));
    let mut cpu_matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &cpu_context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("CPU matmul kernel");
    let mut cpu_encoder = Encoder::<Cpu>::new(&cpu_context).expect("CPU encoder");
    let mut cpu_arguments = quant_arguments(&mut cpu_buffers, &input);
    cpu_arguments.d_transform = MatmulDOps {
        rht_factors: cpu_output_hadamard_factors.as_ref(),
        ..MatmulDOps::none()
    };
    cpu_matmul.encode(cpu_arguments, &mut cpu_encoder).expect("CPU A8W matmul");
    cpu_encoder.end_encoding().submit().wait_until_completed().unwrap();
    let mut reference = allocation_to_vec::<Cpu, bf16>(&cpu_buffers.y);
    for row in reference.chunks_exact_mut(n as usize) {
        for (value, bias) in row.iter_mut().zip(&output_bias) {
            *value = bf16::from_f32(value.to_f32() + bias.to_f32());
        }
    }

    let mut metal_buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let metal_output_bias = crate::tests::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &output_bias);
    let metal_output_hadamard_factors = output_hadamard_factors
        .as_ref()
        .map(|factors| crate::tests::helpers::alloc_allocation_with_data::<Metal, i32>(&context, factors));
    let mut metal_matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("Metal matmul kernel");
    let mut metal_encoder = Encoder::<Metal>::new(&context).expect("Metal encoder");
    let mut metal_arguments = quant_arguments(&mut metal_buffers, &input);
    metal_arguments.d_transform = MatmulDOps {
        bias: Some(&metal_output_bias),
        rht_factors: metal_output_hadamard_factors.as_ref(),
        ..MatmulDOps::none()
    };
    metal_matmul
        .gemm
        .encode_with_engine(metal_arguments, GemmEngine::Mxu, &mut metal_encoder)
        .expect("Metal A8W MXU matmul with output bias");
    metal_encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&metal_buffers.y);

    assert_parity::<bf16>(
        &format!("A8W{bits} MXU output bias, output Hadamard={with_output_hadamard}"),
        &reference,
        &actual,
        0.08,
        0.8,
    );
}

fn run_widened_f32<B: Backend>(
    context: &B::Context,
    input: &QuantInput<bf16>,
) -> Vec<f32> {
    use crate::{
        backends::common::kernel::matmul::{MatmulA, MatmulRouting},
        data_type::DataType,
        tests::helpers::alloc_allocation,
    };
    let buffers = QuantBuffers::<B, bf16>::allocate(context, input);
    let mut y = alloc_allocation::<B, f32>(context, (input.m as usize) * (input.n as usize));
    let mut matmul =
        <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(context, DataType::BF16, DataType::BF16, DataType::F32)
            .expect("MatmulKernel widened");
    let b = quant_b_variant(&buffers.w, &buffers.scales, buffers.zp.as_ref(), buffers.bias.as_ref(), input);
    let mut encoder = Encoder::<B>::new(context).expect("encoder");
    matmul
        .encode(
            crate::backends::common::kernel::matmul::MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &buffers.x,
                    offset: 0,
                },
                b,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut y,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Dense,
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut encoder,
        )
        .expect("widened encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, f32>(&y)
}

#[rstest]
#[test_attr(uzu_test)]
#[case::gemv_m1(1)]
#[case::gemv_m4(4)]
#[case::gemm_m31(31)]
fn parity_widened_f32_output(#[case] m: u32) {
    // Mirrors the dflash draft-chain readout: quantized LM head (4-bit ZP,
    // group 32), BF16 input, widened F32 output.
    let (k, n, gs) = (5120u32, 32768u32, 32u32);
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(m, k, n, gs, 4, QuantizationMethod::ScaleZeroPoint, 0);
    let cpu_context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let reference = run_widened_f32::<Cpu>(&cpu_context, &input);
    let actual = run_widened_f32::<Metal>(&context, &input);
    assert_parity::<f32>(&format!("widened f32 m={m}"), &reference, &actual, 0.05, 0.4);
}
