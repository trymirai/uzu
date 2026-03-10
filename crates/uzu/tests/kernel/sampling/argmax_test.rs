use std::{
    fmt::Display,
    mem::size_of,
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
            Context, Kernels,
            gpu_types::ArgmaxPair,
            kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel},
        },
        cpu::Cpu,
    },
};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
}

fn reference_argmax<T: ArrayElement + Float>(input: &Input<T>) -> Vec<u32> {
    let mut tokens = vec![0u32; input.batch_size as usize];
    for batch_idx in 0..input.batch_size as usize {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = u32::MAX;
        for vocab_idx in 0..input.vocab_size as usize {
            let global_idx = batch_idx * input.vocab_size as usize + vocab_idx;
            let value = input.logits[global_idx].to_f32().unwrap();
            if value > best_value || (value == best_value && (vocab_idx as u32) < best_index) {
                best_value = value;
                best_index = vocab_idx as u32;
            }
        }
        tokens[batch_idx] = best_index;
    }
    tokens
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
) -> (Input<T>, Vec<u32>) {
    let len = (batch_size * vocab_size) as usize;
    let mut logits: Vec<T> = vec![T::zero(); len];
    for i in 0..len {
        logits[i] = T::from((i as f32 * 0.1).sin() * 10.0f32).unwrap();
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        batch_size,
        vocab_size,
    };

    let expected = reference_argmax(&input);

    (input, expected)
}

fn get_output_single<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<u32> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::ArgmaxSingleKernel::new(&context, T::data_type())
        .expect("Failed to create ArgmaxSingleKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let output_array = context.create_array_uninitialized(&[input.batch_size as usize], DataType::U32, "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        logits_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        &mut command_buffer,
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice::<u32>().to_vec()
}

fn get_output_two_pass<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<u32> {
    let context = B::Context::new().expect("Failed to create Context");

    let main_kernel = <<B as Backend>::Kernels as Kernels>::ArgmaxMainKernel::new(&context, T::data_type())
        .expect("Failed to create ArgmaxMainKernel");
    let final_kernel = <<B as Backend>::Kernels as Kernels>::ArgmaxFinalKernel::new(&context)
        .expect("Failed to create ArgmaxFinalKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let output_array = context.create_array_uninitialized(&[input.batch_size as usize], DataType::U32, "");

    let block_size = 1024;
    let grain_size = 4;
    let elements_per_group = block_size * grain_size;
    let vocab_groups_per_batch = (input.vocab_size as usize + elements_per_group - 1) / elements_per_group;
    let partial_results_count = input.batch_size as usize * vocab_groups_per_batch;
    let mut partial_results_buffer = context
        .create_buffer(partial_results_count * size_of::<ArgmaxPair>())
        .expect("Failed to create partial results buffer");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    main_kernel.encode(
        logits_array.buffer().borrow().deref(),
        &mut partial_results_buffer,
        input.batch_size,
        input.vocab_size,
        &mut command_buffer,
    );
    final_kernel.encode(
        &partial_results_buffer,
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        &mut command_buffer,
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice::<u32>().to_vec()
}

fn test_single_pass<T: ArrayElement + Float + Display>(
    batch_size: u32,
    vocab_size: u32,
) {
    let (input, expected) = get_test_data::<T>(batch_size, vocab_size);

    let cpu_output = get_output_single::<T, Cpu>(&input);
    assert_eq!(cpu_output, expected, "CPU single-pass output does not match reference");

    for_each_backend!(|B| {
        let output = get_output_single::<T, B>(&input);
        assert_eq!(output, expected, "Single-pass output does not match for backend {}", std::any::type_name::<B>());
    });
}

fn test_two_pass<T: ArrayElement + Float + Display>(
    batch_size: u32,
    vocab_size: u32,
) {
    let (input, expected) = get_test_data::<T>(batch_size, vocab_size);

    let cpu_output = get_output_two_pass::<T, Cpu>(&input);
    assert_eq!(cpu_output, expected, "CPU two-pass output does not match reference");

    for_each_backend!(|B| {
        let output = get_output_two_pass::<T, B>(&input);
        assert_eq!(output, expected, "Two-pass output does not match for backend {}", std::any::type_name::<B>());
    });
}

// Single-pass tests
#[test]
fn test_single_pass_f32() {
    test_single_pass::<f32>(4, 1024);
}

#[test]
fn test_single_pass_f16() {
    test_single_pass::<f16>(4, 1024);
}

#[test]
fn test_single_pass_bf16() {
    test_single_pass::<bf16>(4, 1024);
}

// Two-pass tests
#[test]
fn test_two_pass_f32() {
    test_two_pass::<f32>(4, 1024);
}

#[test]
fn test_two_pass_f16() {
    test_two_pass::<f16>(4, 1024);
}

#[test]
fn test_two_pass_bf16() {
    test_two_pass::<bf16>(4, 1024);
}

// Large vocab tests (exercises multi-group reduction)
#[test]
fn test_single_pass_large_vocab_f32() {
    test_single_pass::<f32>(4, 128 * 1024);
}

#[test]
fn test_two_pass_large_vocab_f32() {
    test_two_pass::<f32>(4, 128 * 1024);
}

// Edge cases
#[test]
fn test_single_batch_f32() {
    test_single_pass::<f32>(1, 1024);
    test_two_pass::<f32>(1, 1024);
}

#[test]
fn test_small_vocab_f32() {
    test_single_pass::<f32>(4, 4);
    test_two_pass::<f32>(4, 4);
}
