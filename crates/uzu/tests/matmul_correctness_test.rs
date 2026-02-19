//! Correctness tests comparing Metal matmul kernels against ndarray

use bytemuck;
use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use ndarray::Array2;
use uzu::{
    DataType,
    backends::{
        common::Context,
        metal::{
            MetalContext,
            kernel::{MatmulArguments, MatmulKernel},
        },
    },
};

fn run_metal_matmul(
    ctx: &MetalContext,
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Vec<bf16> {
    let a_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(a_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let d_buf = ctx
        .device
        .new_buffer(m * n * core::mem::size_of::<bf16>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let ldb = if transpose_b {
        k
    } else {
        n
    };

    let mut kernel = MatmulKernel::new(DataType::BF16).expect("kernel");

    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer").to_owned();
    let enc = cb.new_compute_command_encoder().expect("Failed to create compute encoder");
    let encode_result = kernel.encode(
        ctx,
        &enc,
        MatmulArguments {
            a: &a_buf,
            a_offset: 0,
            b: &b_buf,
            c: None,
            d: &d_buf,
            bias: None,
            batch: m as i32,
            input_dim: k as i32,
            output_dim: n as i32,
            lda: k as i32,
            ldb: ldb as i32,
            ldd: n as i32,
            batch_count: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b,
        },
    );
    enc.end_encoding();
    encode_result.expect("encode");
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents().as_ptr() as *const bf16;
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
    let a: Vec<bf16> = (0..(m * k)).map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06)).collect();
    let b_size = if transpose_b {
        n * k
    } else {
        k * n
    };
    let b: Vec<bf16> = (0..b_size).map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15)).collect();
    (a, b)
}

struct TestCase {
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
    tolerance: f32,
}

const MATMUL_CORRECTNESS_FULL_ENV: &str = "UZU_MATMUL_CORRECTNESS_FULL";

const QUICK_BATCH_SIZES: [usize; 2] = [1, 16];
const FULL_BATCH_SIZES: [usize; 12] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

const QUICK_MODEL_SHAPES: [(usize, usize); 8] = [
    (896, 896),
    (896, 4864),
    (1024, 4096),
    (1152, 6912),
    (1536, 8960),
    (2048, 8192),
    (4096, 14336),
    (5120, 17408),
];

const FULL_MODEL_SHAPES: [(usize, usize); 51] = [
    // Qwen2.5-Coder-0.5B
    (896, 896),
    (896, 1152),
    (896, 4864),
    (4864, 896),
    // Qwen3-0.6B
    (1024, 1024),
    (1024, 3072),
    (1024, 4096),
    (3072, 1024),
    // Gemma-3-1B
    (1152, 1152),
    (1152, 1536),
    (1152, 6912),
    (6912, 1152),
    // Qwen2.5-Coder-1.5B
    (1536, 1536),
    (1536, 2048),
    (1536, 8960),
    (8960, 1536),
    // Llama-3.2-1B / Qwen3-1.7B / SmolLM2-1.7B
    (2048, 2048),
    (2048, 2560),
    (2048, 3072),
    (2048, 4096),
    (2048, 6144),
    (2048, 8192),
    (2048, 11008),
    (6144, 2048),
    (8192, 2048),
    (11008, 2048),
    // Qwen3-4B / Gemma-3-4B
    (2560, 2560),
    (2560, 4096),
    (2560, 6144),
    (2560, 9728),
    (2560, 10240),
    (9728, 2560),
    (10240, 2560),
    // Llama-3.2-3B
    (3072, 3072),
    (3072, 5120),
    (3072, 8192),
    (8192, 3072),
    // Qwen2.5-Coder-7B
    (3584, 3584),
    (3584, 4608),
    (3584, 18944),
    (18944, 3584),
    // Llama-3.1-8B / Qwen3-8B
    (4096, 4096),
    (4096, 6144),
    (4096, 12288),
    (4096, 14336),
    (12288, 4096),
    (14336, 4096),
    // Qwen3-14B
    (5120, 5120),
    (5120, 7168),
    (5120, 17408),
    (17408, 5120),
];

fn run_full_case_matrix() -> bool {
    std::env::var(MATMUL_CORRECTNESS_FULL_ENV)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

fn test_cases(full_case_matrix: bool) -> Vec<TestCase> {
    let base_tolerance = 0.01;
    let (batch_sizes, model_shapes): (&[usize], &[(usize, usize)]) = if full_case_matrix {
        (&FULL_BATCH_SIZES, &FULL_MODEL_SHAPES)
    } else {
        (&QUICK_BATCH_SIZES, &QUICK_MODEL_SHAPES)
    };

    let mut cases: Vec<(usize, usize, usize, bool)> =
        model_shapes.iter().flat_map(|&(k, n)| batch_sizes.iter().map(move |&m| (m, k, n, true))).collect();

    // Edge case: small matrix
    for &m in batch_sizes {
        cases.push((m, 128, 128, true));
    }

    cases
        .into_iter()
        .map(|(m, k, n, transpose_b)| {
            let tolerance = base_tolerance * (k as f32 / 1024.0).sqrt();
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

    for (i, (&m_val, &r_val)) in metal.iter().zip(reference.iter()).enumerate() {
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
#[ignore]
fn matmul_correctness_comprehensive() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("No Metal device available, skipping test");
        return;
    };

    let full_case_matrix = run_full_case_matrix();
    let cases = test_cases(full_case_matrix);
    eprintln!(
        "Running {} matmul correctness case matrix (set {}=1 for exhaustive run)",
        if full_case_matrix {
            "exhaustive"
        } else {
            "quick"
        },
        MATMUL_CORRECTNESS_FULL_ENV
    );
    let mut passed = 0;
    let mut failed = Vec::new();

    for case in &cases {
        let (a, b) = generate_test_data(case.m, case.k, case.n, case.transpose_b);
        let metal_result = run_metal_matmul(&ctx, &a, &b, case.m, case.k, case.n, case.transpose_b);
        let reference = run_ndarray_matmul(&a, &b, case.m, case.k, case.n, case.transpose_b);

        let trans_str = if case.transpose_b {
            "T"
        } else {
            "N"
        };

        match compare_results(&metal_result, &reference, case.tolerance) {
            Ok(()) => {
                passed += 1;
                eprintln!("✓ m={} k={} n={} B={}", case.m, case.k, case.n, trans_str);
            },
            Err((max_diff, idx, count)) => {
                failed.push((case, max_diff, idx, count));
                eprintln!(
                    "✗ m={} k={} n={} B={} max_diff={:.6} at idx {} ({} exceed tol {})",
                    case.m, case.k, case.n, trans_str, max_diff, idx, count, case.tolerance
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
                case.m, case.k, case.n, trans_str, max_diff, idx, count, case.tolerance
            );
        }
        panic!("{} tests failed", failed.len());
    }
}
