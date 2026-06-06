#![cfg(metal_backend)]

use std::{
    error::Error as StdError,
    fmt::{Debug, Display},
};

use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulError, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
};
use half::{bf16, f16};
use num_traits::Float;
use proc_macros::__internal_uzu_test as uzu_test;
use rstest::rstest;

use crate::common::{
    helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    matmul::{QuantBuffers, QuantInput, quant_arguments, run_quant_cpu, run_quant_metal},
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

fn matmul_error_source(error: &dyn StdError) -> &MatmulError<Metal> {
    error.source().and_then(|source| source.downcast_ref::<MatmulError<Metal>>()).expect("expected MatmulError source")
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

const LLOYD_MAX_CODEBOOK_SIZE: usize = 16;
const BIAS_CODEBOOK_SIZE: usize = 16;

struct LloydMaxQuantInput<T: ArrayElement + Float> {
    w_packed: Vec<u32>,
    scales: Vec<T>,
    codebook: [f16; LLOYD_MAX_CODEBOOK_SIZE],
    bias_indices: Vec<u8>,
    bias_codebook: [f16; BIAS_CODEBOOK_SIZE],
    x: Vec<T>,
    k: u32,
    n: u32,
    m: u32,
    group_size: u32,
    mode: QuantizationMode,
}

impl<T: ArrayElement + Float> LloydMaxQuantInput<T> {
    fn new(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
    ) -> Self {
        let num_groups_k = k / group_size as usize;
        Self {
            w_packed: deterministic_packed_u4_weights(n, k),
            scales: lloyd_max_scales(n, num_groups_k),
            codebook: lloyd_max_codebook(),
            bias_indices: lloyd_max_bias_indices(n, num_groups_k),
            bias_codebook: lloyd_max_bias_codebook(),
            x: lloyd_max_input_values(m, k),
            k: k as u32,
            n: n as u32,
            m: m as u32,
            group_size,
            mode: QuantizationMode::U4,
        }
    }
}

struct LloydMaxQuantBuffers<B: Backend, T: ArrayElement + Float> {
    w: Allocation<B>,
    scales: Allocation<B>,
    codebook: Allocation<B>,
    bias_indices: Allocation<B>,
    bias_codebook: Allocation<B>,
    x: Allocation<B>,
    y: Allocation<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> LloydMaxQuantBuffers<B, T> {
    fn allocate(
        context: &B::Context,
        input: &LloydMaxQuantInput<T>,
    ) -> Self {
        Self {
            w: alloc_allocation_with_data::<B, u32>(context, &input.w_packed),
            scales: alloc_allocation_with_data::<B, T>(context, &input.scales),
            codebook: alloc_allocation_with_data::<B, f16>(context, &input.codebook),
            bias_indices: alloc_allocation_with_data::<B, u8>(context, &input.bias_indices),
            bias_codebook: alloc_allocation_with_data::<B, f16>(context, &input.bias_codebook),
            x: alloc_allocation_with_data::<B, T>(context, &input.x),
            y: alloc_allocation::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        }
    }
}

fn lloyd_max_quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut LloydMaxQuantBuffers<B, T>,
    input: &LloydMaxQuantInput<T>,
) -> MatmulArguments<'a, B> {
    MatmulArguments {
        a: &buffers.x,
        a_offset: 0,
        b: MatmulB::LloydMaxDequant {
            b: &buffers.w,
            scales: &buffers.scales,
            codebook: &buffers.codebook,
            bias_indices: &buffers.bias_indices,
            bias_codebook: &buffers.bias_codebook,
            mode: input.mode,
            group_size: input.group_size,
        },
        b_offset: 0,
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.y,
        d_transform: MatmulDOps::none(),
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

fn lloyd_max_codebook() -> [f16; LLOYD_MAX_CODEBOOK_SIZE] {
    [
        -1.0f32,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_04,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ]
    .map(f16::from_f32)
}

fn lloyd_max_bias_codebook() -> [f16; BIAS_CODEBOOK_SIZE] {
    [
        -0.045f32, -0.039, -0.033, -0.026, -0.020, -0.013, -0.007, 0.0, 0.007, 0.013, 0.020, 0.026, 0.033, 0.039,
        0.045, 0.052,
    ]
    .map(f16::from_f32)
}

fn deterministic_packed_u4_weights(
    output_size: usize,
    input_size: usize,
) -> Vec<u32> {
    (0..(output_size * input_size / 8)).map(|word_index| word_index.wrapping_mul(2_654_435_761) as u32).collect()
}

fn lloyd_max_scale_value(
    output_index: usize,
    group_index: usize,
) -> f32 {
    0.07 + 0.013 * ((output_index + 5 * group_index) % 13) as f32
}

fn lloyd_max_scales<T: ArrayElement + Float>(
    output_size: usize,
    group_count: usize,
) -> Vec<T> {
    (0..output_size)
        .flat_map(|output_index| {
            (0..group_count).map(move |group_index| T::from(lloyd_max_scale_value(output_index, group_index)).unwrap())
        })
        .collect()
}

fn lloyd_max_bias_indices(
    output_size: usize,
    group_count: usize,
) -> Vec<u8> {
    let bias_stride = group_count.div_ceil(2);
    (0..(output_size * bias_stride)).map(|byte_index| byte_index.wrapping_mul(2_246_822_519) as u8).collect()
}

fn lloyd_max_input_values<T: ArrayElement + Float>(
    batch_size: usize,
    input_size: usize,
) -> Vec<T> {
    (0..batch_size)
        .flat_map(|batch_index| {
            (0..input_size).map(move |input_index| {
                T::from((((batch_index * 11 + input_index * 7) % 23) as f32 - 11.0) * 0.022).unwrap()
            })
        })
        .collect()
}

fn run_lloyd_max_cpu<T: ArrayElement + Float>(input: &LloydMaxQuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut buffers = LloydMaxQuantBuffers::<Cpu, T>::allocate(&context, input);
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    matmul.encode(lloyd_max_quant_arguments(&mut buffers, input), &mut encoder).expect("encode CPU Lloyd-Max quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&buffers.y)
}

fn run_lloyd_max_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &LloydMaxQuantInput<T>,
) -> Vec<T> {
    let mut buffers = LloydMaxQuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    matmul.encode(lloyd_max_quant_arguments(&mut buffers, input), &mut encoder).expect("matmul encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&buffers.y)
}

#[rstest]
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

#[rstest]
#[case::m1_gs64(1, 512, 64, 64)]
#[case::m2_gs64(2, 512, 64, 64)]
#[case::m3_gs64(3, 512, 64, 64)]
#[case::m4_gs64(4, 512, 64, 64)]
fn parity_gemv_lloyd_max_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] group_size: u32,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = LloydMaxQuantInput::<bf16>::new(m, k, n, group_size);
    let reference = run_lloyd_max_cpu::<bf16>(&input);
    let actual = run_lloyd_max_metal::<bf16>(&context, &input);
    assert_parity::<bf16>("gemv_lloyd_max", &reference, &actual, 0.05, 0.4);
}

#[rstest]
#[case::u8_mode(1, 512, 64, 64, Some(QuantizationMode::U8), false)]
#[case::group_size96(1, 512, 64, 96, None, false)]
#[case::m5_batch(5, 512, 64, 64, None, false)]
#[case::n12_width(1, 512, 12, 64, None, false)]
#[case::k768_width(1, 768, 64, 64, None, false)]
#[case::rht(1, 512, 64, 64, None, true)]
fn lloyd_max_qmv_unsupported_config_returns_matmul_error(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] group_size: u32,
    #[case] mode_override: Option<QuantizationMode>,
    #[case] use_rht: bool,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = LloydMaxQuantInput::<bf16>::new(m, k, n, group_size);
    let mut buffers = LloydMaxQuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulMetalKernel");

    let rht = use_rht.then(|| {
        (0..n)
            .map(|index| {
                if index % 2 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect::<Vec<i32>>()
    });
    let rht_buffer =
        rht.as_ref().map(|factors| crate::common::helpers::alloc_allocation_with_data::<Metal, i32>(&context, factors));

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = lloyd_max_quant_arguments(&mut buffers, &input);
    if let Some(mode) = mode_override {
        if let backend_uzu::backends::common::kernel::matmul::MatmulB::LloydMaxDequant {
            mode: matmul_mode,
            ..
        } = &mut args.b
        {
            *matmul_mode = mode;
        }
    }
    args.d_transform = MatmulDOps {
        ab_scale: 1.0,
        accumulate: false,
        bias: None,
        rht_factors: rht_buffer.as_ref(),
    };

    let error = matmul.encode(args, &mut encoder).expect_err("expected unsupported Lloyd-Max QMV configuration");
    let matmul = matmul_error_source(&error);
    assert!(
        matches!(matmul, MatmulError::UnsupportedGroupSize(96))
            || matches!(
                matmul,
                MatmulError::UnsupportedFeature {
                    feature: "Lloyd-Max QMV",
                    ..
                }
            ),
        "got {matmul:?}"
    );
}

#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_with_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.5 + 0.1 * (j % 5) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let bias_pp_buf = crate::common::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
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
    let bias_buf = crate::common::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
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
    let cpu_rht = crate::common::helpers::alloc_allocation_with_data::<Cpu, i32>(&cpu_context, &rht);
    let cpu_bias = crate::common::helpers::alloc_allocation_with_data::<Cpu, bf16>(&cpu_context, &bias_t);
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
    let metal_rht = crate::common::helpers::alloc_allocation_with_data::<Metal, i32>(&context, &rht);
    let metal_bias = crate::common::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
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
    let cpu_rht = crate::common::helpers::alloc_allocation_with_data::<Cpu, i32>(&cpu_context, &rht);
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
    let metal_rht = crate::common::helpers::alloc_allocation_with_data::<Metal, i32>(&context, &rht);
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
fn quant_gemm_nonzero_b_offset_returns_unsupported_layout() {
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
    args.b_offset = 16;
    let result = matmul.encode(args, &mut encoder);

    let err = result.expect_err("expected error");
    let matmul: &MatmulError<Metal> = (&err as &dyn StdError)
        .source()
        .and_then(|s| s.downcast_ref::<MatmulError<Metal>>())
        .expect("expected MatmulError source");
    assert!(
        matches!(
            matmul,
            MatmulError::UnsupportedLayout {
                path: "QuantGemm"
            }
        ),
        "got {matmul:?}"
    );
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
#[case::gs32_4bit_mlx(128, 256, 64, 32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx(128, 256, 64, 64, 4, QuantizationMethod::ScaleBias)]
#[case::gs128_4bit_mlx(128, 256, 64, 128, 4, QuantizationMethod::ScaleBias)]
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
