use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rstest::rstest;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        matmul::{QuantBuffers, QuantInput},
    },
};

struct Input<T: ArrayElement + Float> {
    a: Box<[T]>,
    b: Box<[T]>,
    m: usize,
    k: usize,
    n: usize,
    ids: Option<Box<[u32]>>,
    soft_cap: Option<f32>,
}

fn get_test_data<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
) -> (Input<T>, Vec<T>) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();

    let input = Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        m,
        k,
        n,
        ids: None,
        soft_cap: None,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

// Encode one GEMV (dense, or a per-row B-row gather when `gather_indices` is set) and copy out.
fn run_gemv<'a, B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    a: &'a Allocation<B>,
    b: MatmulB<'a, B>,
    gather_indices: Option<&'a Allocation<B>>,
    m: usize,
    n_out: usize,
    k: usize,
    soft_cap: Option<f32>,
) -> Vec<T> {
    let mut d = alloc_allocation::<B, T>(context, m * n_out);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context, T::data_type(), T::data_type(), T::data_type())
            .expect("MatmulKernel");
    let mut encoder = Encoder::new(context).expect("encoder");
    kernel
        .encode(
            MatmulArguments {
                a,
                a_offset: 0,
                b,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d,
                d_transform: MatmulDOps {
                    soft_cap,
                    ..MatmulDOps::none()
                },
                gather_indices,
                m: m as u32,
                n: n_out as u32,
                k: k as u32,
            },
            &mut encoder,
        )
        .expect("encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, T>(&d)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let a = alloc_allocation_with_data::<B, T>(&context, &input.a);
    let weights = alloc_allocation_with_data::<B, T>(&context, &input.b);
    let ids = input.ids.as_ref().map(|ids| alloc_allocation_with_data::<B, u32>(&context, ids));
    run_gemv::<B, T>(
        &context,
        &a,
        MatmulB::FullPrecision {
            b: &weights,
        },
        ids.as_ref(),
        input.m,
        input.n,
        input.k,
        input.soft_cap,
    )
}

fn test<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    let (input, expected) = get_test_data::<T>(m, k, n);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float(&expected, &output, eps, &format!("backend {}", std::any::type_name::<B>()));
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
fn gemv_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<bf16>(m, k, n, 0.1);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
fn gemv_f32(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<f32>(m, k, n, 0.01);
}

fn assert_gather<T: ArrayElement + Float + Debug + Display>(
    dense: &[T],
    gather: &[T],
    ids: &[u32],
    m: usize,
    vocab: usize,
    ids_per_row: usize,
    eps: f32,
    name: &str,
) {
    let mut expected = vec![T::from(0.0).unwrap(); m * ids_per_row];
    for r in 0..m {
        for c in 0..ids_per_row {
            expected[r * ids_per_row + c] = dense[r * vocab + ids[r * ids_per_row + c] as usize];
        }
    }
    assert_eq_float(&expected, gather, eps, &format!("gather vs dense ({name})"));
}

macro_rules! check_gather {
    ($m:expr, $vocab:expr, $ids:expr, $ids_per_row:expr, $eps:expr, |$B:ident| $run:block) => {{
        {
            #[allow(non_camel_case_types)]
            type $B = Cpu;
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, "Cpu");
        }
        for_each_non_cpu_backend!(|$B| {
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, std::any::type_name::<$B>());
        });
    }};
}

fn fp_gather_case<T: ArrayElement + Float + Debug + Display>(
    soft_cap: Option<f32>,
    eps: f32,
) {
    let (m, k, vocab, ids_per_row) = (4usize, 128usize, 256usize, 8usize);
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let weights: Vec<T> = (0..vocab * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();

    // Dense (`n = vocab`, no ids) and gather (`n = ids_per_row`, ids) share `a`/`weights`/soft-cap.
    let make = |n: usize, ids: Option<Box<[u32]>>| Input {
        a: a.clone().into_boxed_slice(),
        b: weights.clone().into_boxed_slice(),
        m,
        k,
        n,
        ids,
        soft_cap,
    };
    let dense_input = make(vocab, None);
    let gather_input = make(ids_per_row, Some(ids.clone().into_boxed_slice()));

    check_gather!(m, vocab, ids, ids_per_row, eps, |B| {
        (get_output::<T, B>(&dense_input), get_output::<T, B>(&gather_input))
    });
}

#[uzu_test]
fn gemv_gather() {
    // Full precision: one call per dtype (generic over T, so bf16/f32 can't be a runtime loop).
    for soft_cap in [None, Some(15.0)] {
        fp_gather_case::<bf16>(soft_cap, 0.1);
        fp_gather_case::<f32>(soft_cap, 0.01);
    }
    // Quantized (bf16, per bits/method) — inline, since it isn't type-generic.
    for (bits, method) in [
        (4, QuantizationMethod::ScaleBias),
        (4, QuantizationMethod::ScaleZeroPoint),
        (4, QuantizationMethod::ScaleSymmetric),
        (8, QuantizationMethod::ScaleZeroPoint),
    ] {
        let (m, k, vocab, ids_per_row, group_size) = (4usize, 128usize, 64usize, 8usize, 32u32);
        let input = QuantInput::<bf16>::new(m, k, vocab, group_size, bits, method, 0x5EED);
        let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();
        // K_SPLIT == 1 keeps k in one reduction, so gather and dense share the exact accumulation.
        check_gather!(m, vocab, ids, ids_per_row, 0.05, |B| {
            let context = <B as Backend>::Context::new().expect("context");
            let buffers = QuantBuffers::<B, bf16>::allocate(&context, &input);
            let ids_alloc = alloc_allocation_with_data::<B, u32>(&context, &ids);
            let variant = || match method {
                QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
                    b: &buffers.w,
                    scales: &buffers.scales,
                    biases: buffers.bias.as_ref().expect("bias buffer"),
                    mode: input.mode,
                    group_size: input.group_size,
                },
                QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                    b: &buffers.w,
                    scales: &buffers.scales,
                    zero_points: buffers.zp.as_ref().expect("zp buffer"),
                    mode: input.mode,
                    group_size: input.group_size,
                },
                QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
                    b: &buffers.w,
                    scales: &buffers.scales,
                    mode: input.mode,
                    group_size: input.group_size,
                },
            };
            (
                run_gemv::<B, bf16>(&context, &buffers.x, variant(), None, m, vocab, k, None),
                run_gemv::<B, bf16>(&context, &buffers.x, variant(), Some(&ids_alloc), m, ids_per_row, k, None),
            )
        });
    }
}
