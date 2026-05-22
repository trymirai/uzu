#![cfg(metal_backend)]

use std::{
    collections::HashSet,
    error::Error as StdError,
    fmt::{Debug, Display},
};

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOp, MatmulError, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{MatmulDispatchPath, Metal, MetalContext},
    },
};
use half::bf16;
use num_traits::Float;
use proc_macros::__internal_uzu_test as uzu_test;
use rstest::rstest;

use crate::common::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec};

struct QuantInput<T: ArrayElement + Float> {
    w_packed: Vec<u32>,
    scales: Vec<T>,
    zero_points: Option<Vec<u8>>,
    biases: Option<Vec<T>>,
    x: Vec<T>,
    k: u32,
    n: u32,
    m: u32,
    group_size: u32,
    quant_method: QuantizationMethod,
    mode: QuantizationMode,
}

fn pack_weights_u32(
    values: &[u8],
    bits: u32,
) -> Vec<u32> {
    if bits == 4 {
        values
            .chunks(8)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &v) in chunk.iter().enumerate() {
                    word |= ((v & 0xF) as u32) << (i * 4);
                }
                word
            })
            .collect()
    } else {
        values
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &v) in chunk.iter().enumerate() {
                    word |= (v as u32) << (i * 8);
                }
                word
            })
            .collect()
    }
}

fn pack_zero_points(
    values: &[u8],
    bits: u32,
) -> Vec<u8> {
    if bits == 4 {
        values
            .chunks(2)
            .map(|chunk| {
                let lo = chunk[0] & 0x0F;
                let hi = if chunk.len() > 1 {
                    chunk[1] & 0x0F
                } else {
                    0
                };
                lo | (hi << 4)
            })
            .collect()
    } else {
        values.to_vec()
    }
}

fn make_input<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) -> QuantInput<T> {
    let num_groups_k = k.div_ceil(group_size as usize);
    let max_val: u8 = if bits == 4 { 15 } else { 255 };

    let mut weights_raw: Vec<u8> = Vec::with_capacity(n * k);
    for j in 0..n {
        for l in 0..k {
            weights_raw.push(((j * 3 + l * 7 + 1) % (max_val as usize + 1)) as u8);
        }
    }
    let w_packed = pack_weights_u32(&weights_raw, bits);

    let scales: Vec<T> = (0..n * num_groups_k)
        .map(|i| {
            let (j, g) = (i / num_groups_k, i % num_groups_k);
            T::from(0.5 + 0.1 * ((j + g) % 5) as f32).unwrap()
        })
        .collect();

    let zp_stride = if bits == 4 { num_groups_k.div_ceil(2) } else { num_groups_k };

    let (zero_points, biases) = match quant_method {
        QuantizationMethod::ScaleZeroPoint => {
            let mut zp_raw: Vec<u8> = Vec::with_capacity(n * num_groups_k);
            for j in 0..n {
                for g in 0..num_groups_k {
                    let zp_val = ((j * 2 + g * 3) % (max_val as usize + 1)) as u8;
                    zp_raw.push(zp_val);
                }
            }
            let mut zp_packed: Vec<u8> = Vec::with_capacity(n * zp_stride);
            for j in 0..n {
                let row = &zp_raw[j * num_groups_k..(j + 1) * num_groups_k];
                let packed_row = pack_zero_points(row, bits);
                let mut padded = packed_row;
                padded.resize(zp_stride, 0);
                zp_packed.extend_from_slice(&padded);
            }
            (Some(zp_packed), None)
        },
        QuantizationMethod::ScaleBias => {
            let biases: Vec<T> = (0..n * num_groups_k)
                .map(|i| {
                    let (j, g) = (i / num_groups_k, i % num_groups_k);
                    T::from(0.01 * ((j + g * 2) % 7) as f32).unwrap()
                })
                .collect();
            (None, Some(biases))
        },
    };

    let x: Vec<T> = (0..m * k)
        .map(|i| T::from(0.1 * f32::sin(i as f32 * 0.05) + 0.5).unwrap())
        .collect();

    let mode = match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::I8,
        _ => unreachable!(),
    };

    QuantInput {
        w_packed,
        scales,
        zero_points,
        biases,
        x,
        k: k as u32,
        n: n as u32,
        m: m as u32,
        group_size,
        quant_method,
        mode,
    }
}

fn run_with_cpu_backend<T: ArrayElement + Float>(input: &QuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let w_buf = alloc_allocation_with_data::<Cpu, u32>(&context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Cpu, T>(&context, &input.scales);
    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<Cpu, u8>(&context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_allocation_with_data::<Cpu, T>(&context, b));
    let x_buf = alloc_allocation_with_data::<Cpu, T>(&context, &input.x);
    let mut y_buf = alloc_allocation::<Cpu, T>(&context, (input.m as usize) * (input.n as usize));

    let b_variant: MatmulB<'_, Cpu> = match input.quant_method {
        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
            b: &w_buf,
            scales: &s_buf,
            zero_points: zp_buf.as_ref().expect("zp"),
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
            b: &w_buf,
            scales: &s_buf,
            biases: bias_buf.as_ref().expect("bias"),
            mode: input.mode,
            group_size: input.group_size,
        },
    };

    let mut matmul = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulCpuKernel");
    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    matmul
        .encode(
            MatmulArguments {
                a: &x_buf,
                a_offset: 0,
                b: b_variant,
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut y_buf,
                d_transform: HashSet::new(),
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut encoder,
        )
        .expect("encode cpu quant");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&y_buf)
}

fn run_with_path<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    path: MatmulDispatchPath,
) -> Vec<T> {
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type())
        .expect("MatmulMetalKernel");

    let w_buf = alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Metal, T>(context, &input.scales);
    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<Metal, u8>(context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_allocation_with_data::<Metal, T>(context, b));
    let x_buf = alloc_allocation_with_data::<Metal, T>(context, &input.x);
    let mut y_buf = alloc_allocation::<Metal, T>(context, (input.m as usize) * (input.n as usize));

    let b_variant: MatmulB<'_, Metal> = match input.quant_method {
        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
            b: &w_buf,
            scales: &s_buf,
            zero_points: zp_buf.as_ref().expect("zp"),
            mode: input.mode,
            group_size: input.group_size,
        },
        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
            b: &w_buf,
            scales: &s_buf,
            biases: bias_buf.as_ref().expect("bias"),
            mode: input.mode,
            group_size: input.group_size,
        },
    };

    let mut encoder = Encoder::new(context).expect("encoder");
    matmul
        .encode_with_path(
            MatmulArguments {
                a: &x_buf,
                a_offset: 0,
                b: b_variant,
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut y_buf,
                d_transform: HashSet::new(),
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut encoder,
            path,
        )
        .expect("encode quantized matmul");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&y_buf)
}

fn check_tolerance(expected: f32, actual: f32, rel_tol: f64, abs_tol: f64) -> bool {
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
    let input = make_input::<T>(m, k, n, group_size, bits, quant_method);
    let unified = run_with_path::<T>(&context, &input, MatmulDispatchPath::QuantGemm);
    let reference = run_with_cpu_backend::<T>(&input);

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

// --- bf16: parity vs CPU oracle across (group_size, bits, method, shape) ---
// bf16 ZP cases use wider tolerance: the unified gemm kernel does the
// (scale * w + bias) multiply-add in T precision per element, so the per-group
// accumulator drifts further from the f32 oracle than the legacy qmm kernel did.
// Deltas observed up to ~2.7 on inputs of magnitude ~30.
#[rstest]
//                       (   m,    k,   n,   gs, bits, method,                                rel,  abs)
#[case::gs32_4bit_mlx_prefill ( 64, 256,  64,  32, 4, QuantizationMethod::ScaleBias,        0.05, 0.5)]
#[case::gs64_4bit_mlx_prefill ( 64, 256,  64,  64, 4, QuantizationMethod::ScaleBias,        0.05, 0.5)]
#[case::gs128_4bit_mlx_prefill( 64, 256,  64, 128, 4, QuantizationMethod::ScaleBias,        0.05, 0.5)]
#[case::gs32_8bit_mlx_prefill ( 64, 256,  64,  32, 8, QuantizationMethod::ScaleBias,        0.05, 0.5)]
#[case::gs64_8bit_zp_prefill  ( 64, 256,  64,  64, 8, QuantizationMethod::ScaleZeroPoint,   0.1,  5.0)]
#[case::gs128_8bit_zp_prefill ( 64, 256,  64, 128, 8, QuantizationMethod::ScaleZeroPoint,   0.1,  5.0)]
#[case::gs32_4bit_mlx_decode  (  8, 256,  64,  32, 4, QuantizationMethod::ScaleBias,        0.05, 0.5)]
#[case::gs64_4bit_zp_decode   (  8, 256,  64,  64, 4, QuantizationMethod::ScaleZeroPoint,   0.1,  5.0)]
#[case::gs32_unaligned_n      ( 64, 256,  96,  32, 4, QuantizationMethod::ScaleBias,        0.05, 0.5)]
fn parity_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[case] rel: f64,
    #[case] abs: f64,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, rel, abs);
}

// --- f16: exercises (32,32,32) tile only ---
#[rstest]
//                  (  m,    k,   n,   gs, bits, method,                              rel,  abs)
#[case::gs64_4bit_mlx ( 32, 256,  64,  64, 4, QuantizationMethod::ScaleBias,        0.02, 0.5)]
#[case::gs128_8bit_zp ( 32, 256,  64, 128, 8, QuantizationMethod::ScaleZeroPoint,   0.02, 0.5)]
fn parity_f16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
    #[case] rel: f64,
    #[case] abs: f64,
) {
    run_parity::<half::f16>(m, k, n, gs, bits, method, rel, abs);
}

// Bias post-pass for quant_gemm: a MatmulDOp::Bias is applied as a separate
// bias-add kernel after the matmul core. Oracle adds the same broadcast bias.
#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_with_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = make_input::<bf16>(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias);

    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.5 + 0.1 * (j % 5) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");
    let w_buf = alloc_allocation_with_data::<Metal, u32>(&context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.scales);
    let bias_buf = alloc_allocation_with_data::<Metal, bf16>(&context, input.biases.as_ref().expect("bias"));
    let bias_pp_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
    let x_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.x);
    let mut y_buf = alloc_allocation::<Metal, bf16>(&context, (input.m as usize) * (input.n as usize));

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    matmul
        .encode_with_path::<backend_uzu::backends::common::Allocation<Metal>>(
            MatmulArguments {
                a: &x_buf,
                a_offset: 0,
                b: MatmulB::ScaleBiasDequant {
                    b: &w_buf,
                    scales: &s_buf,
                    biases: &bias_buf,
                    mode: input.mode,
                    group_size: input.group_size,
                },
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut y_buf,
                d_transform: HashSet::from([MatmulDOp::Bias {
                    bias: &bias_pp_buf,
                }]),
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut encoder,
            MatmulDispatchPath::QuantGemm,
        )
        .expect("encode quant with bias");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&y_buf);

    let mut reference = run_with_cpu_backend::<bf16>(&input);
    for i in 0..(input.m as usize) {
        for j in 0..(input.n as usize) {
            let idx = i * input.n as usize + j;
            reference[idx] = bf16::from_f32(reference[idx].to_f32() + bias_f32[j]);
        }
    }
    assert_parity::<bf16>("with_bias", &reference, &actual, 0.1, 5.0);
}

// Regression: quant dispatch rejects nonzero b_offset rather than silently dropping it.
#[uzu_test]
fn quant_gemm_nonzero_b_offset_returns_unsupported_layout() {
    let context = MetalContext::new().expect("Metal context");
    let input = make_input::<bf16>(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias);

    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");

    let w_buf = alloc_allocation_with_data::<Metal, u32>(&context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.scales);
    let bias_buf = alloc_allocation_with_data::<Metal, bf16>(&context, input.biases.as_ref().expect("bias"));
    let x_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.x);
    let mut y_buf = alloc_allocation::<Metal, bf16>(&context, (input.m as usize) * (input.n as usize));

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let result = matmul.encode_with_path::<backend_uzu::backends::common::Allocation<Metal>>(
        MatmulArguments {
            a: &x_buf,
            a_offset: 0,
            b: MatmulB::ScaleBiasDequant {
                b: &w_buf,
                scales: &s_buf,
                biases: &bias_buf,
                mode: input.mode,
                group_size: input.group_size,
            },
            b_offset: 16,
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut y_buf,
            d_transform: HashSet::new(),
            m: input.m,
            n: input.n,
            k: input.k,
        },
        &mut encoder,
        MatmulDispatchPath::QuantGemm,
    );

    let err = result.expect_err("expected error");
    let matmul: &MatmulError<Metal> = (&err as &dyn StdError)
        .source()
        .and_then(|s| s.downcast_ref::<MatmulError<Metal>>())
        .expect("expected MatmulError source");
    assert!(
        matches!(matmul, MatmulError::UnsupportedLayout { path: "QuantGemm" }),
        "got {matmul:?}"
    );
}

// Regression: encode returns Err on rejected D-transform instead of panicking.
#[uzu_test]
fn quant_gemm_accumulate_returns_unsupported_dop() {
    let context = MetalContext::new().expect("Metal context");
    let input = make_input::<bf16>(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias);

    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");

    let w_buf = alloc_allocation_with_data::<Metal, u32>(&context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.scales);
    let bias_buf = alloc_allocation_with_data::<Metal, bf16>(&context, input.biases.as_ref().expect("bias"));
    let x_buf = alloc_allocation_with_data::<Metal, bf16>(&context, &input.x);
    let mut y_buf = alloc_allocation::<Metal, bf16>(&context, (input.m as usize) * (input.n as usize));

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let result = matmul.encode_with_path::<backend_uzu::backends::common::Allocation<Metal>>(
        MatmulArguments {
            a: &x_buf,
            a_offset: 0,
            b: MatmulB::ScaleBiasDequant {
                b: &w_buf,
                scales: &s_buf,
                biases: &bias_buf,
                mode: input.mode,
                group_size: input.group_size,
            },
            b_offset: 0,
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut y_buf,
            d_transform: HashSet::from([MatmulDOp::Accumulate]),
            m: input.m,
            n: input.n,
            k: input.k,
        },
        &mut encoder,
        MatmulDispatchPath::QuantGemm,
    );

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

