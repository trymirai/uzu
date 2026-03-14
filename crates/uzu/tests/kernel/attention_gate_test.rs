use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels, kernel::AttentionGateKernel,
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    qkv: Box<[T]>,
    output: Box<[T]>,
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    suffix_length: u32,
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    suffix_length: u32,
) -> Input<T> {
    let qkv_stride = (2 * num_heads + 2 * num_groups) as usize * head_dim as usize;
    let qkv_size = suffix_length as usize * qkv_stride;
    let output_size = (suffix_length * num_heads * head_dim) as usize;

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from(((i as f32) * 0.1).sin() * 2.0).unwrap();
    }

    let mut output = vec![T::zero(); output_size];
    for i in 0..output_size {
        output[i] = T::from(((i as f32) * 0.07 + 1.0).cos() * 3.0).unwrap();
    }

    Input {
        qkv: qkv.into_boxed_slice(),
        output: output.into_boxed_slice(),
        num_heads,
        num_groups,
        head_dim,
        suffix_length,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionGateKernel::new(&context, T::data_type())
        .expect("Failed to create AttentionGateKernel");

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;

    let qkv_array = context.create_array_from(&[input.qkv.len()], &input.qkv, "");
    let output_array = context.create_array_from(&[output_size], &input.output, "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        qkv_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.num_heads,
        input.num_groups,
        input.head_dim,
        input.suffix_length,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(input: &Input<T>) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.01f32
    } else {
        1e-6
    };

    let expected = get_output::<T, Cpu>(input);

    for_each_non_cpu_backend!(|B| {
        let result = get_output::<T, B>(input);
        let backend_name = std::any::type_name::<B>();
        assert_eq_float::<T>(&expected, &result, eps, &format!("attention_gate mismatch (backend={})", backend_name));
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // Typical GQA config
    let input = get_test_data::<T>(8, 4, 4, 3);
    test_internal(&input);
}

fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    // Single token (decode)
    let input = get_test_data::<T>(4, 2, 8, 1);
    test_internal(&input);
}

fn test_small<T: ArrayElement + Float + Debug + Display>() {
    // Minimal config
    let input = get_test_data::<T>(2, 1, 4, 2);
    test_internal(&input);
}

// f32 tests
#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[test]
fn test_single_token_f32() {
    test_single_token::<f32>();
}

#[test]
fn test_small_f32() {
    test_small::<f32>();
}

// f16 tests
#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_single_token_f16() {
    test_single_token::<f16>();
}

#[test]
fn test_small_f16() {
    test_small::<f16>();
}

// bf16 tests
#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[test]
fn test_single_token_bf16() {
    test_single_token::<bf16>();
}

#[test]
fn test_small_bf16() {
    test_small::<bf16>();
}
