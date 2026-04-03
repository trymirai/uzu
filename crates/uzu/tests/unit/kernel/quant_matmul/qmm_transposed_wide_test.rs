use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Backend, Buffer, Context, Encoder, Kernels, kernel::QuantizedMatmulQmmTransposedWideKernel},
        cpu::Cpu,
    },
};

use super::{Input, check_tolerance, pack_weights_u32, pack_zero_points};
use crate::common::helpers::alloc_buffer_with_data;

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedWideKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        input.bits,
        input.use_zero_points,
        input.use_mlx_quant,
    )
    .expect("Failed to create QuantizedMatmulQmmTransposedWideKernel");

    let w_buf = alloc_buffer_with_data::<B, u32>(&context, &input.w_packed);
    let scales_buf = alloc_buffer_with_data::<B, T>(&context, &input.scales);

    let zp_buf = input.zero_points.as_ref().map(|zp| alloc_buffer_with_data::<B, u8>(&context, zp));
    let bias_buf = input.biases.as_ref().map(|b| alloc_buffer_with_data::<B, T>(&context, b));

    let x_buf = alloc_buffer_with_data::<B, T>(&context, &input.x);
    let output_size = (input.m as usize) * (input.n as usize) * T::data_type().size_in_bytes();
    let mut y_buf = context.create_buffer(output_size).expect("Failed to create buffer");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &w_buf,
        &scales_buf,
        zp_buf.as_ref(),
        bias_buf.as_ref(),
        &x_buf,
        &mut y_buf,
        input.k,
        input.n,
        input.m,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const T;
    let y_len = (input.m as usize) * (input.n as usize);
    unsafe { std::slice::from_raw_parts(y_ptr, y_len) }.to_vec()
}

/// Create test data for transposed wide qmm: x[m×k] * w_transposed[n×k] = y[m×n]
/// Weights are in transposed [n × k] layout.
fn get_test_data_basic<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) -> (Input<T>, Vec<T>) {
    let num_groups_k = (k + group_size as usize - 1) / group_size as usize;

    // Generate quantized weights [n × k]
    let max_val = if bits == 4 {
        15u8
    } else {
        255u8
    };
    let mut weights_raw: Vec<u8> = Vec::with_capacity(n * k);
    for j in 0..n {
        for l in 0..k {
            weights_raw.push(((j * 3 + l * 7 + 1) % (max_val as usize + 1)) as u8);
        }
    }
    let w_packed = pack_weights_u32(&weights_raw, bits);

    // Scales: [n × num_groups_k]
    let mut scales_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
    for j in 0..n {
        for g in 0..num_groups_k {
            scales_f32.push(0.5 + 0.1 * ((j + g) % 5) as f32);
        }
    }
    let scales: Vec<T> = scales_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    // Zero points or biases
    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    let (zero_points, biases) = if use_zero_points {
        let mut zp_raw: Vec<u8> = Vec::with_capacity(n * num_groups_k);
        for j in 0..n {
            for g in 0..num_groups_k {
                let zp_val = ((j * 2 + g * 3) % (max_val as usize + 1)) as u8;
                zp_raw.push(zp_val);
            }
        }
        // Pack zero points per row
        let mut zp_packed: Vec<u8> = Vec::with_capacity(n * zp_stride);
        for j in 0..n {
            let row = &zp_raw[j * num_groups_k..(j + 1) * num_groups_k];
            let packed_row = pack_zero_points(row, bits);
            let mut padded = packed_row;
            padded.resize(zp_stride, 0);
            zp_packed.extend_from_slice(&padded);
        }
        (Some(zp_packed), None)
    } else if use_mlx_quant {
        let mut biases_f32: Vec<f32> = Vec::with_capacity(n * num_groups_k);
        for j in 0..n {
            for g in 0..num_groups_k {
                biases_f32.push(0.01 * ((j + g * 2) % 7) as f32);
            }
        }
        let biases: Vec<T> = biases_f32.iter().map(|&v| T::from(v).unwrap()).collect();
        (None, Some(biases))
    } else {
        unreachable!("Must use either zero_points or mlx_quant");
    };

    // X: [m × k]
    let mut x_f32: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for l in 0..k {
            x_f32.push(0.1 * f32::sin((i * k + l) as f32 * 0.05) + 0.5);
        }
    }
    let x: Vec<T> = x_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    let input = Input {
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
        use_zero_points,
        use_mlx_quant,
    };

    let expected = get_output::<Cpu, T>(&input);
    (input, expected)
}

/// Small edge-case test data with known values
fn get_test_data_edge<T: ArrayElement + Float>(
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) -> (Input<T>, Vec<T>) {
    let m = 2usize;
    let k = group_size as usize;
    let n = 64usize;
    let num_groups_k = 1;

    // Simple weights [n × k]: all 1s
    let weights_raw: Vec<u8> = vec![1u8; n * k];
    let w_packed = pack_weights_u32(&weights_raw, bits);

    // Unit scales
    let scales: Vec<T> = vec![T::one(); n * num_groups_k];

    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    let (zero_points, biases) = if use_zero_points {
        // Zero points all 0 -> bias = 0
        let zp_packed = vec![0u8; n * zp_stride];
        (Some(zp_packed), None)
    } else if use_mlx_quant {
        let biases: Vec<T> = vec![T::zero(); n * num_groups_k];
        (None, Some(biases))
    } else {
        unreachable!("Must use either zero_points or mlx_quant");
    };

    // X: simple increasing values
    let x_f32: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32 * 0.1).collect();
    let x: Vec<T> = x_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    let input = Input {
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
        use_zero_points,
        use_mlx_quant,
    };

    let expected = get_output::<Cpu, T>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let (rel_tol, abs_tol): (f64, f64) = match T::data_type() {
        DataType::BF16 => (0.05, 0.5),
        DataType::F16 => (0.02, 0.5),
        _ => (0.01, 0.1),
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        assert_eq!(expected.len(), output.len(), "Output length mismatch");

        let mut errors = 0;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let exp_f32 = exp.to_f32().unwrap();
            let got_f32 = got.to_f32().unwrap();
            if !check_tolerance(exp_f32, got_f32, rel_tol, abs_tol) {
                if errors < 5 {
                    eprintln!("  idx={} expected={} got={} diff={}", i, exp_f32, got_f32, (exp_f32 - got_f32).abs());
                }
                errors += 1;
            }
        }
        assert_eq!(
            errors,
            0,
            "QMM transposed wide kernel: backend={}, m={}, k={}, n={}, gs={}, bits={}, zp={}, mlx={}: {} mismatches",
            std::any::type_name::<B>(),
            input.m,
            input.k,
            input.n,
            input.group_size,
            input.bits,
            input.use_zero_points,
            input.use_mlx_quant,
            errors,
        );
    });
}

fn get_test_dims(
    group_size: u32,
    bits: u32,
) -> Vec<(usize, usize, usize)> {
    let gs = group_size as usize;
    // Wide kernel: BM=64, BK=32, BN=64.
    // K must be >= group_size (transposed grouping along K).
    // N must be a multiple of 64 (BN tile size).
    // bf16 8-bit: keep dims small to avoid catastrophic cancellation.
    if bits == 8 {
        vec![(2, gs, 64), (4, gs, 64)]
    } else {
        vec![(2, gs * 2, 64), (4, gs * 2, 128), (8, gs, 128)]
    }
}

fn test_basic(
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let dims = get_test_dims(group_size, bits);

    for (m, k, n) in dims {
        let (input, expected) = get_test_data_basic::<bf16>(m, k, n, group_size, bits, use_zero_points, use_mlx_quant);
        test_internal::<bf16>(&input, &expected);
    }
}

fn test_edge(
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let (input, expected) = get_test_data_edge::<bf16>(group_size, bits, use_zero_points, use_mlx_quant);
    test_internal::<bf16>(&input, &expected);
}

// -- 4-bit, zero points -------------------------------------------------------

#[test]
fn test_bf16_gs32_4bit_zp() {
    test_basic(32, 4, true, false);
}

#[test]
fn test_bf16_gs64_4bit_zp() {
    test_basic(64, 4, true, false);
}

#[test]
fn test_bf16_gs128_4bit_zp() {
    test_basic(128, 4, true, false);
}

// -- 8-bit, zero points -------------------------------------------------------

#[test]
fn test_bf16_gs32_8bit_zp() {
    test_basic(32, 8, true, false);
}

#[test]
fn test_bf16_gs64_8bit_zp() {
    test_basic(64, 8, true, false);
}

#[test]
fn test_bf16_gs128_8bit_zp() {
    test_basic(128, 8, true, false);
}

// -- 4-bit, mlx quant ----------------------------------------------------------

#[test]
fn test_bf16_gs32_4bit_mlx() {
    test_basic(32, 4, false, true);
}

#[test]
fn test_bf16_gs64_4bit_mlx() {
    test_basic(64, 4, false, true);
}

#[test]
fn test_bf16_gs128_4bit_mlx() {
    test_basic(128, 4, false, true);
}

// -- 8-bit, mlx quant ----------------------------------------------------------

#[test]
fn test_bf16_gs32_8bit_mlx() {
    test_basic(32, 8, false, true);
}

#[test]
fn test_bf16_gs64_8bit_mlx() {
    test_basic(64, 8, false, true);
}

#[test]
fn test_bf16_gs128_8bit_mlx() {
    test_basic(128, 8, false, true);
}

// -- Edge cases ----------------------------------------------------------------

#[test]
fn test_edge_bf16_4bit_zp() {
    test_edge(32, 4, true, false);
}

#[test]
fn test_edge_bf16_8bit_zp() {
    test_edge(32, 8, true, false);
}

#[test]
fn test_edge_bf16_4bit_mlx() {
    test_edge(32, 4, false, true);
}

#[test]
fn test_edge_bf16_8bit_mlx() {
    test_edge(32, 8, false, true);
}
