use std::{
    collections::HashSet,
    fmt::{Debug, Display},
};

use half::{bf16, f16};
use num_traits::Float;
use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            AllocationType, Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
};

use crate::{
    common::{assert::assert_eq_float, helpers::{alloc_allocation_with_data, allocation_to_vec}},
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    a: Box<[T]>,
    b: Box<[T]>,
    m: usize,
    k: usize,
    n: usize,
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
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let m = input.m as u32;
    let k = input.k as u32;
    let n = input.n as u32;

    let b_array = context.create_array_from(&[input.n, input.k], &input.b);
    let a_allocation = alloc_allocation_with_data::<B, T>(&context, &input.a);
    let mut d_allocation = context
        .create_allocation(input.m * input.n * std::mem::size_of::<T>(), AllocationType::Global)
        .expect("Failed to create allocation");

    let mut kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel
        .encode(
            MatmulArguments {
                a: &a_allocation,
                a_offset: 0,
                a_prologue: HashSet::new(),
                b: MatmulB::FullPrecision {
                    b: b_array.allocation(),
                },
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d_allocation,
                d_transform: HashSet::new(),
                m,
                n,
                k,
            },
            &mut encoder,
        )
        .expect("encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, T>(&d_allocation)
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

#[uzu_test]
fn test_f16_m1() {
    test::<f16>(1, 128, 64, 0.01);
}

#[uzu_test]
fn test_bf16_m1() {
    test::<bf16>(1, 128, 64, 0.1);
}

#[uzu_test]
fn test_f16_batched() {
    test::<f16>(4, 128, 64, 0.01);
}

#[uzu_test]
fn test_bf16_batched() {
    test::<bf16>(4, 128, 64, 0.1);
}

#[uzu_test]
fn test_f16_max_batch() {
    test::<f16>(8, 128, 64, 0.01);
}

#[uzu_test]
fn test_bf16_max_batch() {
    test::<bf16>(8, 128, 64, 0.1);
}

#[uzu_test]
fn test_f16_unaligned_k() {
    test::<f16>(1, 33, 64, 0.01);
}

#[uzu_test]
fn test_bf16_unaligned_k() {
    test::<bf16>(1, 33, 64, 0.1);
}

#[uzu_test]
fn test_f16_unaligned_n() {
    test::<f16>(1, 128, 11, 0.01);
}

#[uzu_test]
fn test_bf16_unaligned_n() {
    test::<bf16>(1, 128, 11, 0.1);
}

#[uzu_test]
fn test_f16_large() {
    test::<f16>(1, 4096, 2048, 0.5);
}

#[uzu_test]
fn test_bf16_large() {
    test::<bf16>(1, 4096, 2048, 1.0);
}

#[uzu_test]
fn test_f16_small_n() {
    test::<f16>(1, 128, 3, 0.01);
}

#[uzu_test]
fn test_bf16_small_n() {
    test::<bf16>(1, 128, 3, 0.1);
}
