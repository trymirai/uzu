use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context,
            gpu_types::GemmParams,
            kernel::matmul::{
                GridSize, MatmulArguments,
                gemm::{GemmDispatchDescriptor, GemmKernel, GemmSpecialization},
            },
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

    // Pick a precompile config matching alignment requirements
    let configs = GemmSpecialization::precompile_configs(T::data_type());
    let base = configs[0];
    let align_m = (m % base.block_rows) == 0;
    let align_n = (n % base.block_cols) == 0;
    let align_k = (k % base.block_depth) == 0;
    // Find a config that matches alignment, or fall back to overriding the first one
    let config = configs
        .iter()
        .find(|c| c.align_m == align_m && c.align_n == align_n && c.align_k == align_k)
        .copied()
        .unwrap_or(GemmSpecialization {
            align_m,
            align_n,
            align_k,
            ..base
        });

    let threadgroups_per_row = (n + config.block_cols - 1) / config.block_cols;
    let threadgroups_per_column = (m + config.block_rows - 1) / config.block_rows;

    let params = GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        threadgroups_per_row,
        threadgroups_per_column,
        swizzle_log: 0,
        aligned_inner_iterations: k / config.block_depth,
    };

    let descriptor = GemmDispatchDescriptor {
        specialization: config,
        params,
        threadgroups: GridSize {
            x: threadgroups_per_row as usize,
            y: threadgroups_per_column as usize,
            z: 1,
        },
    };

    let mut kernel = GemmKernel::<B>::new(T::data_type()).expect("Failed to create GemmKernel");

    let a_buf = a_array.buffer();
    let a_ref = a_buf.borrow();
    let b_buf = b_array.buffer();
    let b_ref = b_buf.borrow();
    let d_buf = d_array.buffer();
    let mut d_ref = d_buf.borrow_mut();

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    let mut arguments = MatmulArguments {
        a: a_ref.deref(),
        a_offset: 0,
        b: b_ref.deref(),
        d: d_ref.deref_mut(),
        bias: None,
        batch: m,
        input_dim: k,
        output_dim: n,
        lda: k,
        ldb: k,
        ldd: n,
        transpose_b: true,
    };
    kernel.encode(&context, &mut arguments, &descriptor, &mut command_buffer).expect("Failed to encode");
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

// Aligned dimensions (divisible by common block sizes)
#[test]
fn test_f32_aligned() {
    test::<f32>(64, 64, 64, 0.01);
}

#[test]
fn test_f16_aligned() {
    test::<f16>(64, 64, 64, 0.01);
}

#[test]
fn test_bf16_aligned() {
    test::<bf16>(64, 64, 64, 0.1);
}

// Unaligned dimensions
#[test]
fn test_f32_unaligned() {
    test::<f32>(7, 33, 11, 0.01);
}

#[test]
fn test_f16_unaligned() {
    test::<f16>(7, 33, 11, 0.01);
}

#[test]
fn test_bf16_unaligned() {
    test::<bf16>(7, 33, 11, 0.1);
}

// Larger matrix
#[test]
fn test_f32_large() {
    test::<f32>(16, 128, 256, 0.01);
}

#[test]
fn test_f16_large() {
    test::<f16>(16, 128, 256, 0.01);
}

#[test]
fn test_bf16_large() {
    test::<bf16>(16, 128, 256, 0.1);
}
