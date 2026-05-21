#![cfg(metal_backend)]

use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ManualKernels,
                matmul::MatmulKernel,
                quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration},
            },
        },
        metal::{Metal, MetalContext, QuantizedMatmulDispatchPath},
    },
};
use half::bf16;
use num_traits::Float;
use proc_macros::__internal_uzu_test as uzu_test;

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
    #[allow(dead_code)]
    bits: u32,
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

    let mut scales_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
    for j in 0..n {
        for g in 0..num_groups_k {
            scales_f32.push(0.5 + 0.1 * ((j + g) % 5) as f32);
        }
    }
    let scales: Vec<T> = scales_f32.iter().map(|&v| T::from(v).unwrap()).collect();

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
            let mut biases_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
            for j in 0..n {
                for g in 0..num_groups_k {
                    biases_f32.push(0.01 * ((j + g * 2) % 7) as f32);
                }
            }
            let biases: Vec<T> = biases_f32.iter().map(|&v| T::from(v).unwrap()).collect();
            (None, Some(biases))
        },
    };

    let mut x_f32: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for l in 0..k {
            x_f32.push(0.1 * f32::sin((i * k + l) as f32 * 0.05) + 0.5);
        }
    }
    let x: Vec<T> = x_f32.iter().map(|&v| T::from(v).unwrap()).collect();

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
        bits,
        quant_method,
        mode,
    }
}

fn run_with_path<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    path: QuantizedMatmulDispatchPath,
) -> Vec<T> {
    let configuration = QuantizedMatmulConfiguration {
        data_type: T::data_type(),
        group_size: input.group_size as usize,
        input_dim: input.k as usize,
        output_dim: input.n as usize,
        mode: input.mode,
        quantization_method: input.quant_method,
        use_hadamard: false,
    };

    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type())
        .expect("MatmulMetalKernel");

    let w_buf = alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed);
    let s_buf = alloc_allocation_with_data::<Metal, T>(context, &input.scales);
    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_allocation_with_data::<Metal, u8>(context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_allocation_with_data::<Metal, T>(context, b));
    let x_buf = alloc_allocation_with_data::<Metal, T>(context, &input.x);
    let mut y_buf = alloc_allocation::<Metal, T>(context, (input.m as usize) * (input.n as usize));

    let zp_or_bias = match input.quant_method {
        QuantizationMethod::ScaleZeroPoint => zp_buf.as_ref().expect("zp"),
        QuantizationMethod::ScaleBias => bias_buf.as_ref().expect("bias"),
    };

    let mut encoder = Encoder::new(context).expect("encoder");
    matmul
        .encode_quantized_with_path(
            QuantizedMatmulArguments {
                a: &x_buf,
                a_offset: 0,
                b: &w_buf,
                scales: &s_buf,
                zero_points_or_biases: zp_or_bias,
                output: &mut y_buf,
                hadamard_factors: None,
                batch_dim: input.m as usize,
            },
            &configuration,
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
    let reference = run_with_path::<T>(&context, &input, QuantizedMatmulDispatchPath::Auto);
    let unified = run_with_path::<T>(&context, &input, QuantizedMatmulDispatchPath::Gemm);
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

// --- bf16 wide and small shapes — exercises (BM,BK,BN) ∈ {(32,32,32),(64,32,64),(64,64,32)} ---

#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_prefill() {
    run_parity::<bf16>(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs64_4bit_mlx_prefill() {
    run_parity::<bf16>(64, 256, 64, 64, 4, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs128_4bit_mlx_prefill() {
    run_parity::<bf16>(64, 256, 64, 128, 4, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs32_8bit_mlx_prefill() {
    run_parity::<bf16>(64, 256, 64, 32, 8, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs64_8bit_zp_prefill() {
    run_parity::<bf16>(64, 256, 64, 64, 8, QuantizationMethod::ScaleZeroPoint, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs128_8bit_zp_prefill() {
    run_parity::<bf16>(64, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_decode() {
    // batch < 32 → 32x32 tile.
    run_parity::<bf16>(8, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs64_4bit_zp_decode() {
    run_parity::<bf16>(8, 256, 64, 64, 4, QuantizationMethod::ScaleZeroPoint, 0.05, 0.5);
}

#[uzu_test]
fn parity_bf16_gs32_unaligned_n() {
    // n=96 → n % 64 != 0 — falls back to (32, 32, 32) tile.
    run_parity::<bf16>(64, 256, 96, 32, 4, QuantizationMethod::ScaleBias, 0.05, 0.5);
}

// --- f16 — exercises (32,32,32) tile only ---

#[uzu_test]
fn parity_f16_gs64_4bit_mlx() {
    run_parity::<half::f16>(32, 256, 64, 64, 4, QuantizationMethod::ScaleBias, 0.02, 0.5);
}

#[uzu_test]
fn parity_f16_gs128_8bit_zp() {
    run_parity::<half::f16>(32, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint, 0.02, 0.5);
}

