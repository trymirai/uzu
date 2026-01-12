//! Correctness tests comparing Metal matmul kernels against ndarray

use half::bf16;
use metal::{Device, MTLResourceOptions};
use ndarray::Array2;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn run_metal_matmul(
    ctx: &MTLContext,
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Vec<bf16> {
    let a_buf = ctx.device.new_buffer_with_data(
        a_data.as_ptr() as *const _,
        (a_data.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        (b_data.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let ldb = if transpose_b {
        k
    } else {
        n
    };

    let mut kernel = MatmulKernel::new(ctx, DataType::BF16, false, transpose_b)
        .expect("kernel");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            ctx,
            &enc,
            MatmulArguments {
                a: &a_buf,
                a_offset: 0,
                b: &b_buf,
                c: None,
                d: &d_buf,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: ldb as i32,
                ldd: n as i32,
                batch_count: 1,
                alpha: 1.0,
                beta: 0.0,
            },
        )
        .expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents() as *const bf16;
        std::slice::from_raw_parts(ptr, m * n).to_vec()
    }
}

fn run_ndarray_matmul(
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Vec<bf16> {
    let a_f32: Vec<f32> = a_data.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_data.iter().map(|x| x.to_f32()).collect();

    let a_arr = Array2::from_shape_vec((m, k), a_f32).expect("A shape");

    let result = if transpose_b {
        let b_arr = Array2::from_shape_vec((n, k), b_f32).expect("B shape");
        a_arr.dot(&b_arr.t())
    } else {
        let b_arr = Array2::from_shape_vec((k, n), b_f32).expect("B shape");
        a_arr.dot(&b_arr)
    };

    result.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn generate_test_data(
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> (Vec<bf16>, Vec<bf16>) {
    let a: Vec<bf16> = (0..(m * k))
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06))
        .collect();
    let b_size = if transpose_b {
        n * k
    } else {
        k * n
    };
    let b: Vec<bf16> = (0..b_size)
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15))
        .collect();
    (a, b)
}

struct TestCase {
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
    tolerance: f32,
}

fn test_cases() -> Vec<TestCase> {
    // Shapes derived from actual LLM models in the registry.
    // Tolerance scales with sqrt(k) due to floating point accumulation.
    let base_tolerance = 0.01;

    let cases: Vec<(usize, usize, usize, bool)> = vec![
        // === Decode shapes (m=1) from actual models ===
        // Qwen2.5-Coder-0.5B
        (1, 896, 896, true),
        (1, 896, 1152, true),
        (1, 896, 4864, true),
        (1, 4864, 896, true),
        // Qwen3-0.6B
        (1, 1024, 1024, true),
        (1, 1024, 3072, true),
        (1, 1024, 4096, true),
        (1, 3072, 1024, true),
        // Gemma-3-1B
        (1, 1152, 1152, true),
        (1, 1152, 1536, true),
        (1, 1152, 6912, true),
        (1, 6912, 1152, true),
        // Qwen2.5-Coder-1.5B
        (1, 1536, 1536, true),
        (1, 1536, 2048, true),
        (1, 1536, 8960, true),
        (1, 8960, 1536, true),
        // Llama-3.2-1B / Qwen3-1.7B / SmolLM2-1.7B
        (1, 2048, 2048, true),
        (1, 2048, 2560, true),
        (1, 2048, 3072, true),
        (1, 2048, 4096, true),
        (1, 2048, 6144, true),
        (1, 2048, 8192, true),
        (1, 2048, 11008, true),
        (1, 6144, 2048, true),
        (1, 8192, 2048, true),
        (1, 11008, 2048, true),
        // Qwen3-4B / Gemma-3-4B
        (1, 2560, 2560, true),
        (1, 2560, 4096, true),
        (1, 2560, 6144, true),
        (1, 2560, 9728, true),
        (1, 2560, 10240, true),
        (1, 9728, 2560, true),
        (1, 10240, 2560, true),
        // Llama-3.2-3B
        (1, 3072, 3072, true),
        (1, 3072, 5120, true),
        (1, 3072, 8192, true),
        (1, 8192, 3072, true),
        // Qwen2.5-Coder-7B
        (1, 3584, 3584, true),
        (1, 3584, 4608, true),
        (1, 3584, 18944, true),
        (1, 18944, 3584, true),
        // Llama-3.1-8B / Qwen3-8B
        (1, 4096, 4096, true),
        (1, 4096, 6144, true),
        (1, 4096, 12288, true),
        (1, 4096, 14336, true),
        (1, 12288, 4096, true),
        (1, 14336, 4096, true),
        // Qwen3-14B
        (1, 5120, 5120, true),
        (1, 5120, 7168, true),
        (1, 5120, 17408, true),
        (1, 17408, 5120, true),
        // === Prefill shapes (m>1) for representative models ===
        // Llama-3.2-1B prefill
        (8, 2048, 3072, true),
        (16, 2048, 3072, true),
        (32, 2048, 3072, true),
        (64, 2048, 3072, true),
        (128, 2048, 3072, true),
        (256, 2048, 8192, true),
        (512, 2048, 8192, true),
        // Qwen3-4B prefill
        (8, 2560, 6144, true),
        (32, 2560, 9728, true),
        (128, 2560, 9728, true),
        // Llama-3.1-8B prefill
        (8, 4096, 6144, true),
        (32, 4096, 14336, true),
        (128, 4096, 14336, true),
        (256, 4096, 14336, true),
        // === Non-transposed B (for attention AV matmul) ===
        (1, 1024, 1024, false),
        (1, 2048, 2048, false),
        (1, 4096, 4096, false),
        (64, 2048, 2048, false),
        (128, 4096, 4096, false),
        // === Edge cases ===
        (1, 128, 128, true),
        (2, 2048, 2048, true),
        (33, 2048, 2048, true),
        (17, 4096, 4096, true),
    ];

    cases
        .into_iter()
        .map(|(m, k, n, transpose_b)| {
            // Tolerance scales with sqrt(k) for accumulation error
            let tolerance = base_tolerance * (k as f32 / 1024.0).sqrt();
            // Larger m also contributes slightly to tolerance needs
            let tolerance = tolerance * (1.0 + (m as f32).log2() * 0.02);
            TestCase {
                m,
                k,
                n,
                transpose_b,
                tolerance,
            }
        })
        .collect()
}

fn compare_results(
    metal: &[bf16],
    reference: &[bf16],
    tolerance: f32,
) -> Result<(), (f32, usize, usize)> {
    if metal.len() != reference.len() {
        return Err((f32::INFINITY, 0, metal.len()));
    }

    let mut max_diff: f32 = 0.0;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;

    for (i, (&m_val, &r_val)) in metal.iter().zip(reference.iter()).enumerate()
    {
        let mf = m_val.to_f32();
        let rf = r_val.to_f32();
        let diff = (mf - rf).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        if diff > tolerance {
            diff_count += 1;
        }
    }

    if max_diff <= tolerance {
        Ok(())
    } else {
        Err((max_diff, max_diff_idx, diff_count))
    }
}

#[test]
fn matmul_correctness_comprehensive() {
    let Some(ctx) = create_test_context() else {
        eprintln!("No Metal device available, skipping test");
        return;
    };

    let cases = test_cases();
    let mut passed = 0;
    let mut failed = Vec::new();

    for case in &cases {
        let (a, b) =
            generate_test_data(case.m, case.k, case.n, case.transpose_b);
        let metal_result = run_metal_matmul(
            &ctx,
            &a,
            &b,
            case.m,
            case.k,
            case.n,
            case.transpose_b,
        );
        let reference = run_ndarray_matmul(
            &a,
            &b,
            case.m,
            case.k,
            case.n,
            case.transpose_b,
        );

        let trans_str = if case.transpose_b {
            "T"
        } else {
            "N"
        };

        match compare_results(&metal_result, &reference, case.tolerance) {
            Ok(()) => {
                passed += 1;
                eprintln!(
                    "✓ m={} k={} n={} B={}",
                    case.m, case.k, case.n, trans_str
                );
            },
            Err((max_diff, idx, count)) => {
                failed.push((case, max_diff, idx, count));
                eprintln!(
                    "✗ m={} k={} n={} B={} max_diff={:.6} at idx {} ({} exceed tol {})",
                    case.m,
                    case.k,
                    case.n,
                    trans_str,
                    max_diff,
                    idx,
                    count,
                    case.tolerance
                );
            },
        }
    }

    eprintln!("\n{}/{} tests passed", passed, cases.len());

    if !failed.is_empty() {
        eprintln!("\nFailed tests:");
        for (case, max_diff, idx, count) in &failed {
            let trans_str = if case.transpose_b {
                "T"
            } else {
                "N"
            };
            eprintln!(
                "  m={} k={} n={} B={}: max_diff={:.6} at idx {}, {} exceed tol {}",
                case.m,
                case.k,
                case.n,
                trans_str,
                max_diff,
                idx,
                count,
                case.tolerance
            );
        }
        panic!("{} tests failed", failed.len());
    }
}
