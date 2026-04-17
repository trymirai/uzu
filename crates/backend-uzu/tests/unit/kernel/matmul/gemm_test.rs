use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

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

    let expected = get_output::<T, Cpu>(&input, 1.0);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    ab_scale: f32,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let m = input.m as u32;
    let k = input.k as u32;
    let n = input.n as u32;

    let a_array = context.create_array_from(&[input.m, input.k], &input.a, "");
    let b_array = context.create_array_from(&[input.n, input.k], &input.b, "");
    let d_array = context.create_array_uninitialized(&[input.m, input.n], T::data_type(), "");

    let a_buf = a_array.buffer();
    let a_ref = a_buf.borrow();
    let b_buf = b_array.buffer();
    let b_ref = b_buf.borrow();
    let d_buf = d_array.buffer();
    let mut d_ref = d_buf.borrow_mut();

    let mut kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &context,
        MatmulArguments {
            a: a_ref.deref(),
            a_offset: 0,
            b: b_ref.deref(),
            ab_scale,
            c: MatmulArgumentC::None,
            d: d_ref.deref_mut(),
            batch_dim: m,
            input_dim: k,
            output_dim: n,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

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
        let output = get_output::<T, B>(&input, 1.0);
        assert_eq_float(&expected, &output, eps, &format!("backend {}", std::any::type_name::<B>()));
    });
}

fn test_with_scale<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    ab_scale: f32,
    eps: f32,
) {
    let (input, _) = get_test_data::<T>(m, k, n);
    let expected = get_output::<T, Cpu>(&input, ab_scale);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input, ab_scale);
        assert_eq_float(
            &expected,
            &output,
            eps,
            &format!("backend {} ab_scale={ab_scale}", std::any::type_name::<B>()),
        );
    });
}

#[uzu_test]
fn test_f32_aligned() {
    test::<f32>(64, 64, 64, 0.01);
}

#[uzu_test]
fn test_f16_aligned() {
    test::<f16>(64, 64, 64, 0.01);
}

#[uzu_test]
fn test_bf16_aligned() {
    test::<bf16>(64, 64, 64, 0.1);
}

#[uzu_test]
fn test_f32_unaligned() {
    test::<f32>(7, 33, 11, 0.01);
}

#[uzu_test]
fn test_f16_unaligned() {
    test::<f16>(7, 33, 11, 0.01);
}

#[uzu_test]
fn test_bf16_unaligned() {
    test::<bf16>(7, 33, 11, 0.1);
}

#[uzu_test]
fn test_f32_large() {
    test::<f32>(16, 128, 256, 0.01);
}

#[uzu_test]
fn test_f16_large() {
    test::<f16>(16, 128, 256, 0.01);
}

#[uzu_test]
fn test_bf16_large() {
    test::<bf16>(16, 128, 256, 0.1);
}

// ab_scale tests

#[uzu_test]
fn test_f32_ab_scale() {
    test_with_scale::<f32>(16, 128, 256, 0.5, 0.01);
}

#[uzu_test]
fn test_bf16_ab_scale() {
    test_with_scale::<bf16>(16, 128, 256, 0.5, 0.1);
}
