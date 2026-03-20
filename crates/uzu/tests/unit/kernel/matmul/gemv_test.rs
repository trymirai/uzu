use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context,
            kernel::matmul::{MatmulArguments, MatmulKernel, MatmulKernels},
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

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

    let m = input.m as i32;
    let k = input.k as i32;
    let n = input.n as i32;

    let a_array = context.create_array_from(&[input.m, input.k], &input.a, "");
    let b_array = context.create_array_from(&[input.n, input.k], &input.b, "");
    let d_array = context.create_array_uninitialized(&[input.m, input.n], T::data_type(), "");

    let a_buf = a_array.buffer();
    let a_ref = a_buf.borrow();
    let b_buf = b_array.buffer();
    let b_ref = b_buf.borrow();
    let d_buf = d_array.buffer();
    let mut d_ref = d_buf.borrow_mut();

    let mut kernel = <B::Kernels as MatmulKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        &context,
        MatmulArguments {
            a: a_ref.deref(),
            a_offset: 0,
            b: b_ref.deref(),
            d: d_ref.deref_mut(),
            bias: None,
            batch: m,
            input_dim: k,
            output_dim: n,
            leading_dimension_a: k,
            leading_dimension_b: k,
            leading_dimension_d: n,
            transpose_b: true,
        },
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    drop(d_ref);
    d_array.as_slice().to_vec()
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

#[test]
fn test_f32_m1() {
    test::<f32>(1, 128, 64, 0.01);
}

#[test]
fn test_f16_m1() {
    test::<f16>(1, 128, 64, 0.01);
}

#[test]
fn test_bf16_m1() {
    test::<bf16>(1, 128, 64, 0.1);
}

#[test]
fn test_f32_batched() {
    test::<f32>(4, 128, 64, 0.01);
}

#[test]
fn test_f16_batched() {
    test::<f16>(4, 128, 64, 0.01);
}

#[test]
fn test_bf16_batched() {
    test::<bf16>(4, 128, 64, 0.1);
}

#[test]
fn test_f32_max_batch() {
    test::<f32>(8, 128, 64, 0.01);
}

#[test]
fn test_f16_max_batch() {
    test::<f16>(8, 128, 64, 0.01);
}

#[test]
fn test_bf16_max_batch() {
    test::<bf16>(8, 128, 64, 0.1);
}

#[test]
fn test_f32_unaligned_k() {
    test::<f32>(1, 33, 64, 0.01);
}

#[test]
fn test_f16_unaligned_k() {
    test::<f16>(1, 33, 64, 0.01);
}

#[test]
fn test_bf16_unaligned_k() {
    test::<bf16>(1, 33, 64, 0.1);
}

#[test]
fn test_f32_unaligned_n() {
    test::<f32>(1, 128, 11, 0.01);
}

#[test]
fn test_f16_unaligned_n() {
    test::<f16>(1, 128, 11, 0.01);
}

#[test]
fn test_bf16_unaligned_n() {
    test::<bf16>(1, 128, 11, 0.1);
}

#[test]
fn test_f32_large() {
    test::<f32>(1, 4096, 2048, 0.05);
}

#[test]
fn test_f16_large() {
    test::<f16>(1, 4096, 2048, 0.5);
}

#[test]
fn test_bf16_large() {
    test::<bf16>(1, 4096, 2048, 1.0);
}

#[test]
fn test_f32_small_n() {
    test::<f32>(1, 128, 3, 0.01);
}

#[test]
fn test_f16_small_n() {
    test::<f16>(1, 128, 3, 0.01);
}

#[test]
fn test_bf16_small_n() {
    test::<bf16>(1, 128, 3, 0.1);
}
