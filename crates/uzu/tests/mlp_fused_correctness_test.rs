//! Correctness tests for MLP fused kernels
//! Compares fused (matmul + activation) output against separate operations

use half::f16;
use metal::{Device, MTLResourceOptions};
use objc2::rc::autoreleasepool;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{
            mlp::MlpActivationType,
            mlp_fused::{MlpFusedArguments, MlpFusedKernel},
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

/// CPU reference implementation for SiLU activation
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// CPU reference implementation for GELU activation
fn gelu(x: f32) -> f32 {
    let k0 = 0.044715f32;
    let k1 = 0.7978845608f32;
    let t = k1 * (x + k0 * x * x * x);
    0.5 * x * (1.0 + t.tanh())
}

/// CPU reference for MLP fused: up * activation(gate)
fn mlp_fused_reference(
    input: &[f32],
    weights_up: &[f32],
    weights_gate: &[f32],
    m: usize,
    k: usize,
    hidden_dim: usize,
    activation: MlpActivationType,
) -> Vec<f32> {
    let mut result = vec![0.0f32; m * hidden_dim];

    for row in 0..m {
        for col in 0..hidden_dim {
            // Compute up and gate dot products
            let mut up_sum = 0.0f32;
            let mut gate_sum = 0.0f32;
            for i in 0..k {
                let input_val = input[row * k + i];
                up_sum += input_val * weights_up[col * k + i];
                gate_sum += input_val * weights_gate[col * k + i];
            }

            // Apply fused activation
            let activated_gate = match activation {
                MlpActivationType::SiLU => silu(gate_sum),
                MlpActivationType::Gelu => gelu(gate_sum),
            };
            result[row * hidden_dim + col] = up_sum * activated_gate;
        }
    }

    result
}

/// Run MLP fused kernel (decode path, M=1)
fn run_fused_gemv(
    ctx: &MTLContext,
    input: &[f16],
    weights: &[f16], // [2*hidden_dim, k] - up then gate weights concatenated
    k: usize,
    hidden_dim: usize,
    activation: MlpActivationType,
) -> Vec<f16> {
    let input_buf = ctx.device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (input.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buf = ctx.device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        (weights.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (hidden_dim * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MlpFusedKernel::new(DataType::F16, true).expect("kernel");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();

    let args = MlpFusedArguments {
        input: &input_buf,
        input_offset: 0,
        weights: &weights_buf,
        output: &output_buf,
        batch: 1,
        input_dim: k as i32,
        hidden_dim: hidden_dim as i32,
        lda: k as i32,
        ldb: k as i32,
        ldd: hidden_dim as i32,
        batch_count: 1,
        activation,
    };

    kernel.encode(ctx, &enc, &args).expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = output_buf.contents() as *const f16;
        std::slice::from_raw_parts(ptr, hidden_dim).to_vec()
    }
}

/// Run MLP fused kernel (prefill path, M>1)
fn run_fused_gemm(
    ctx: &MTLContext,
    input: &[f16],
    weights: &[f16], // [2*hidden_dim, k] - up then gate weights concatenated
    m: usize,
    k: usize,
    hidden_dim: usize,
    activation: MlpActivationType,
) -> Vec<f16> {
    let input_buf = ctx.device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (input.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buf = ctx.device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        (weights.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (m * hidden_dim * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MlpFusedKernel::new(DataType::F16, true).expect("kernel");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();

    let args = MlpFusedArguments {
        input: &input_buf,
        input_offset: 0,
        weights: &weights_buf,
        output: &output_buf,
        batch: m as i32,
        input_dim: k as i32,
        hidden_dim: hidden_dim as i32,
        lda: k as i32,
        ldb: k as i32,
        ldd: hidden_dim as i32,
        batch_count: 1,
        activation,
    };

    kernel.encode(ctx, &enc, &args).expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = output_buf.contents() as *const f16;
        std::slice::from_raw_parts(ptr, m * hidden_dim).to_vec()
    }
}

fn compare_outputs(
    expected: &[f32],
    actual: &[f16],
    rtol: f32,
    atol: f32,
) -> (bool, f32) {
    let mut max_diff = 0.0f32;
    let mut all_close = true;

    for (i, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let act_f32 = f32::from(act);
        let diff = (exp - act_f32).abs();
        let rel_diff = diff / (exp.abs() + 1e-8);

        if diff > atol && rel_diff > rtol {
            if all_close {
                eprintln!(
                    "First mismatch at index {}: expected {}, got {} (diff={}, rel={})",
                    i, exp, act_f32, diff, rel_diff
                );
            }
            all_close = false;
        }
        max_diff = max_diff.max(diff);
    }

    (all_close, max_diff)
}

#[test]
fn test_mlp_fused_gemv_silu_small() {
    autoreleasepool(|_| {
        let ctx = match create_test_context() {
            Some(c) => c,
            None => {
                eprintln!("No Metal device available, skipping test");
                return;
            },
        };

        let k = 64;
        let hidden_dim = 32;
        let m = 1;

        // Generate random-ish test data
        let input: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let weights_up: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.017).cos() * 0.3)
            .collect();
        let weights_gate: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.023).sin() * 0.3)
            .collect();

        // Create concatenated weights [up; gate] row-major
        let mut weights_concat: Vec<f32> =
            Vec::with_capacity(2 * hidden_dim * k);
        weights_concat.extend(&weights_up);
        weights_concat.extend(&weights_gate);

        // CPU reference
        let expected = mlp_fused_reference(
            &input,
            &weights_up,
            &weights_gate,
            m,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        // Convert to f16
        let input_f16: Vec<f16> =
            input.iter().map(|&x| f16::from_f32(x)).collect();
        let weights_f16: Vec<f16> =
            weights_concat.iter().map(|&x| f16::from_f32(x)).collect();

        // Run fused kernel
        let actual = run_fused_gemv(
            &ctx,
            &input_f16,
            &weights_f16,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        let (close, max_diff) = compare_outputs(&expected, &actual, 0.01, 0.01);
        println!("GEMV SiLU small: max_diff = {}", max_diff);
        assert!(close, "GEMV SiLU small test failed, max_diff = {}", max_diff);
    });
}

#[test]
fn test_mlp_fused_gemv_gelu_small() {
    autoreleasepool(|_| {
        let ctx = match create_test_context() {
            Some(c) => c,
            None => {
                eprintln!("No Metal device available, skipping test");
                return;
            },
        };

        let k = 64;
        let hidden_dim = 32;
        let m = 1;

        let input: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let weights_up: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.017).cos() * 0.3)
            .collect();
        let weights_gate: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.023).sin() * 0.3)
            .collect();

        let mut weights_concat: Vec<f32> =
            Vec::with_capacity(2 * hidden_dim * k);
        weights_concat.extend(&weights_up);
        weights_concat.extend(&weights_gate);

        let expected = mlp_fused_reference(
            &input,
            &weights_up,
            &weights_gate,
            m,
            k,
            hidden_dim,
            MlpActivationType::Gelu,
        );

        let input_f16: Vec<f16> =
            input.iter().map(|&x| f16::from_f32(x)).collect();
        let weights_f16: Vec<f16> =
            weights_concat.iter().map(|&x| f16::from_f32(x)).collect();

        let actual = run_fused_gemv(
            &ctx,
            &input_f16,
            &weights_f16,
            k,
            hidden_dim,
            MlpActivationType::Gelu,
        );

        let (close, max_diff) = compare_outputs(&expected, &actual, 0.01, 0.01);
        println!("GEMV GELU small: max_diff = {}", max_diff);
        assert!(close, "GEMV GELU small test failed, max_diff = {}", max_diff);
    });
}

#[test]
fn test_mlp_fused_gemm_silu_medium() {
    autoreleasepool(|_| {
        let ctx = match create_test_context() {
            Some(c) => c,
            None => {
                eprintln!("No Metal device available, skipping test");
                return;
            },
        };

        let m = 32;
        let k = 64;
        let hidden_dim = 64;

        let input: Vec<f32> =
            (0..m * k).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let weights_up: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.017).cos() * 0.3)
            .collect();
        let weights_gate: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.023).sin() * 0.3)
            .collect();

        let mut weights_concat: Vec<f32> =
            Vec::with_capacity(2 * hidden_dim * k);
        weights_concat.extend(&weights_up);
        weights_concat.extend(&weights_gate);

        let expected = mlp_fused_reference(
            &input,
            &weights_up,
            &weights_gate,
            m,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        let input_f16: Vec<f16> =
            input.iter().map(|&x| f16::from_f32(x)).collect();
        let weights_f16: Vec<f16> =
            weights_concat.iter().map(|&x| f16::from_f32(x)).collect();

        let actual = run_fused_gemm(
            &ctx,
            &input_f16,
            &weights_f16,
            m,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        let (close, max_diff) = compare_outputs(&expected, &actual, 0.05, 0.05);
        println!("GEMM SiLU medium: max_diff = {}", max_diff);
        assert!(close, "GEMM SiLU medium test failed, max_diff = {}", max_diff);
    });
}

#[test]
fn test_mlp_fused_gemv_realistic_dimensions() {
    autoreleasepool(|_| {
        let ctx = match create_test_context() {
            Some(c) => c,
            None => {
                eprintln!("No Metal device available, skipping test");
                return;
            },
        };

        // Realistic LLM dimensions (scaled down for test speed)
        let k = 512; // input dim
        let hidden_dim = 256; // hidden dim (half of up+gate)
        let m = 1;

        let input: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let weights_up: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.001).cos() * 0.1)
            .collect();
        let weights_gate: Vec<f32> = (0..hidden_dim * k)
            .map(|i| (i as f32 * 0.002).sin() * 0.1)
            .collect();

        let mut weights_concat: Vec<f32> =
            Vec::with_capacity(2 * hidden_dim * k);
        weights_concat.extend(&weights_up);
        weights_concat.extend(&weights_gate);

        let expected = mlp_fused_reference(
            &input,
            &weights_up,
            &weights_gate,
            m,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        let input_f16: Vec<f16> =
            input.iter().map(|&x| f16::from_f32(x)).collect();
        let weights_f16: Vec<f16> =
            weights_concat.iter().map(|&x| f16::from_f32(x)).collect();

        let actual = run_fused_gemv(
            &ctx,
            &input_f16,
            &weights_f16,
            k,
            hidden_dim,
            MlpActivationType::SiLU,
        );

        let (close, max_diff) = compare_outputs(&expected, &actual, 0.05, 0.02);
        println!("GEMV realistic: max_diff = {}", max_diff);
        assert!(close, "GEMV realistic test failed, max_diff = {}", max_diff);
    });
}
