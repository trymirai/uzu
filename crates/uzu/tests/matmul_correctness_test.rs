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
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
    tolerance: f32,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // GEMV shapes (m=1)
        TestCase {
            name: "gemv_small",
            m: 1,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.01,
        },
        TestCase {
            name: "gemv_medium",
            m: 1,
            k: 4096,
            n: 4096,
            transpose_b: true,
            tolerance: 0.01,
        },
        TestCase {
            name: "gemv_large",
            m: 1,
            k: 8192,
            n: 8192,
            transpose_b: true,
            tolerance: 0.02,
        },
        // MLP shapes
        TestCase {
            name: "mlp_up",
            m: 1,
            k: 2048,
            n: 8192,
            transpose_b: true,
            tolerance: 0.01,
        },
        TestCase {
            name: "mlp_down",
            m: 1,
            k: 8192,
            n: 2048,
            transpose_b: true,
            tolerance: 0.02,
        },
        // Small batch sizes
        TestCase {
            name: "batch_8",
            m: 8,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.02,
        },
        TestCase {
            name: "batch_16",
            m: 16,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.03,
        },
        TestCase {
            name: "batch_32",
            m: 32,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.05,
        },
        TestCase {
            name: "batch_64",
            m: 64,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.08,
        },
        TestCase {
            name: "batch_128",
            m: 128,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.1,
        },
        // Prefill shapes
        TestCase {
            name: "prefill_256",
            m: 256,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.15,
        },
        TestCase {
            name: "prefill_512",
            m: 512,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.2,
        },
        TestCase {
            name: "prefill_1024",
            m: 1024,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.25,
        },
        // Square matrices
        TestCase {
            name: "square_128",
            m: 128,
            k: 128,
            n: 128,
            transpose_b: true,
            tolerance: 0.05,
        },
        TestCase {
            name: "square_256",
            m: 256,
            k: 256,
            n: 256,
            transpose_b: true,
            tolerance: 0.1,
        },
        TestCase {
            name: "square_512",
            m: 512,
            k: 512,
            n: 512,
            transpose_b: true,
            tolerance: 0.1,
        },
        TestCase {
            name: "square_1024",
            m: 1024,
            k: 1024,
            n: 1024,
            transpose_b: true,
            tolerance: 0.1,
        },
        // Attention shapes
        TestCase {
            name: "attention_qk",
            m: 512,
            k: 64,
            n: 512,
            transpose_b: true,
            tolerance: 0.1,
        },
        TestCase {
            name: "attention_av",
            m: 512,
            k: 512,
            n: 64,
            transpose_b: false,
            tolerance: 0.1,
        },
        // Non-transposed B
        TestCase {
            name: "non_transposed_small",
            m: 64,
            k: 128,
            n: 64,
            transpose_b: false,
            tolerance: 0.15,
        },
        TestCase {
            name: "non_transposed_medium",
            m: 128,
            k: 256,
            n: 128,
            transpose_b: false,
            tolerance: 0.15,
        },
        // Non-aligned sizes
        TestCase {
            name: "unaligned_m",
            m: 17,
            k: 2048,
            n: 2048,
            transpose_b: true,
            tolerance: 0.05,
        },
        TestCase {
            name: "unaligned_n",
            m: 32,
            k: 2048,
            n: 1537,
            transpose_b: true,
            tolerance: 0.1,
        },
        TestCase {
            name: "unaligned_k",
            m: 32,
            k: 1999,
            n: 2048,
            transpose_b: true,
            tolerance: 0.1,
        },
        TestCase {
            name: "unaligned_all",
            m: 33,
            k: 127,
            n: 65,
            transpose_b: true,
            tolerance: 0.1,
        },
        // Edge cases
        TestCase {
            name: "thin_m",
            m: 2,
            k: 4096,
            n: 4096,
            transpose_b: true,
            tolerance: 0.02,
        },
        TestCase {
            name: "thin_n",
            m: 128,
            k: 2048,
            n: 1,
            transpose_b: true,
            tolerance: 0.02,
        },
        TestCase {
            name: "small_k",
            m: 64,
            k: 32,
            n: 64,
            transpose_b: true,
            tolerance: 0.05,
        },
        // Large K (tests SplitK path)
        TestCase {
            name: "large_k_small_mn",
            m: 4,
            k: 16384,
            n: 4,
            transpose_b: true,
            tolerance: 0.1,
        },
    ]
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

        match compare_results(&metal_result, &reference, case.tolerance) {
            Ok(()) => {
                passed += 1;
                eprintln!(
                    "✓ {} (m={}, k={}, n={}, transpose_b={})",
                    case.name, case.m, case.k, case.n, case.transpose_b
                );
            },
            Err((max_diff, idx, count)) => {
                failed.push((case, max_diff, idx, count));
                eprintln!(
                    "✗ {} (m={}, k={}, n={}, transpose_b={}) - max_diff={:.6} at idx {} ({} elements exceed tolerance {})",
                    case.name,
                    case.m,
                    case.k,
                    case.n,
                    case.transpose_b,
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
            eprintln!(
                "  {} ({}x{}x{}, transpose_b={}): max_diff={:.6} at idx {}, {} elements exceed tolerance {}",
                case.name,
                case.m,
                case.k,
                case.n,
                case.transpose_b,
                max_diff,
                idx,
                count,
                case.tolerance
            );
        }
        panic!("{} tests failed", failed.len());
    }
}
