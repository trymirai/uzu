use std::{hint::black_box, time::Instant};

use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

fn random_matrix(
    elements: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..elements).map(|_| rng.random_range(-0.5f32..0.5)).collect()
}

fn run_matmul(
    threads: usize,
    m: u32,
    n: u32,
    k: u32,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let context = <Cpu as Backend>::Context::with_threads(threads);
    let a_allocation = alloc_allocation_with_data::<Cpu, f32>(&context, a);
    let b_allocation = alloc_allocation_with_data::<Cpu, f32>(&context, b);
    let mut d_allocation = alloc_allocation::<Cpu, f32>(&context, (m * n) as usize);

    let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::F32,
        DataType::F32,
        DataType::F32,
    )
    .expect("matmul kernel");

    let mut encoder = Encoder::<Cpu>::new(&context).expect("encoder");
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a_allocation,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &b_allocation,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d_allocation,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                m,
                n,
                k,
            },
            &mut encoder,
        )
        .expect("encode");
    encoder.end_encoding().submit().wait_until_completed().expect("run");

    allocation_to_vec::<Cpu, f32>(&d_allocation)
}

fn reference_matmul(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut d = vec![0.0f32; m * n];
    let mut magnitude = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut accumulator = 0.0f32;
            let mut absolute = 0.0f32;
            for inner in 0..k {
                let term = a[row * k + inner] * b[col * k + inner];
                accumulator += term;
                absolute += term.abs();
            }
            d[row * n + col] = accumulator;
            magnitude[row * n + col] = absolute;
        }
    }
    (d, magnitude)
}

#[uzu_test]
fn threaded_matmul_is_bit_identical() {
    for (m, n, k) in [(1u32, 512u32, 256u32), (8, 300, 128), (64, 129, 64), (3, 1024, 512)] {
        let a = random_matrix((m * k) as usize, 0x9E3779B97F4A7C15 ^ u64::from(n));
        let b = random_matrix((n * k) as usize, 0xD1B54A32D192ED03 ^ u64::from(k));

        let single = run_matmul(1, m, n, k, &a, &b);
        let (reference, magnitude) = reference_matmul(m as usize, n as usize, k as usize, &a, &b);
        for (index, (value, expected)) in single.iter().zip(&reference).enumerate() {
            let tolerance = 2.0 * k as f32 * f32::EPSILON * magnitude[index];
            assert!(
                (value - expected).abs() <= tolerance,
                "{m}x{n}x{k} element {index}: {value} vs {expected}, tolerance {tolerance}"
            );
        }

        for threads in [2usize, 3, 8, 16] {
            let threaded = run_matmul(threads, m, n, k, &a, &b);
            assert_eq!(threaded, single, "threads={threads} changed the result ({m}x{n}x{k})");
        }
    }
}

#[uzu_test]
fn threaded_matmul_handles_ragged_splits() {
    let (m, n, k) = (2u32, 37u32, 19u32);
    let a = random_matrix((m * k) as usize, 0x2545F4914F6CDD1D);
    let b = random_matrix((n * k) as usize, 0x14057B7EF767814F);

    let single = run_matmul(1, m, n, k, &a, &b);
    for threads in 2..=40usize {
        assert_eq!(run_matmul(threads, m, n, k, &a, &b), single, "threads={threads}");
    }
}

#[uzu_test]
fn threaded_matmul_speedup() {
    let (m, n, k) = (1u32, 8192u32, 1024u32);
    let a = random_matrix((m * k) as usize, 0x3C79AC492BA7B653);
    let b = random_matrix((n * k) as usize, 0x76E15D3EFEFDCBBF);

    let single_started = Instant::now();
    let single = run_matmul(1, m, n, k, black_box(&a), black_box(&b));
    let single_elapsed = single_started.elapsed();

    let threads = std::thread::available_parallelism().map(|threads| threads.get()).unwrap_or(1);
    let threaded_started = Instant::now();
    let threaded = run_matmul(threads, m, n, k, black_box(&a), black_box(&b));
    let threaded_elapsed = threaded_started.elapsed();

    assert_eq!(threaded, single);
    println!(
        "cpu matmul {m}x{n}x{k}: 1 thread {single_elapsed:?} -> {threads} threads {threaded_elapsed:?} ({:.2}x)",
        single_elapsed.as_secs_f64() / threaded_elapsed.as_secs_f64()
    );
}
