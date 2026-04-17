use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MinPKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    min_p: f32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    min_p: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * vocab_size) as usize;
    let logits: Vec<T> = (0..len).map(|i| T::from((i as f32 * 0.1).sin() * 2.0).unwrap()).collect();

    let input = Input {
        logits: logits.into_boxed_slice(),
        batch_size,
        vocab_size,
        min_p,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MinPKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create MinPKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let logits_array_buffer_rc = logits_array.buffer();
    let logits_array_borrow = logits_array_buffer_rc.borrow();
    let logits_array_deref = logits_array_borrow.deref();
    let logits_buffer = (!input.in_place).then(|| logits_array_deref);
    let output_array = match input.in_place {
        true => context.create_array_from(&[len], &input.logits, ""),
        false => context.create_array_uninitialized(&[len], T::data_type(), ""),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        logits_buffer,
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        input.min_p,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    min_p: f32,
    in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, vocab_size, min_p, in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("MinP test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

// Out-of-place tests
#[uzu_test]
fn test_f32() {
    test_internal::<f32>(4, 128, 0.1, false);
}

#[uzu_test]
fn test_f16() {
    test_internal::<f16>(4, 128, 0.1, false);
}

#[uzu_test]
fn test_bf16() {
    test_internal::<bf16>(4, 128, 0.1, false);
}

// In-place tests
#[uzu_test]
fn test_in_place_f32() {
    test_internal::<f32>(4, 128, 0.1, true);
}

#[uzu_test]
fn test_in_place_f16() {
    test_internal::<f16>(4, 128, 0.1, true);
}

#[uzu_test]
fn test_in_place_bf16() {
    test_internal::<bf16>(4, 128, 0.1, true);
}

// High min_p — filters most tokens
#[uzu_test]
fn test_high_min_p_f32() {
    test_internal::<f32>(2, 128, 0.9, false);
}

#[uzu_test]
fn test_high_min_p_f16() {
    test_internal::<f16>(2, 128, 0.9, false);
}

#[uzu_test]
fn test_high_min_p_bf16() {
    test_internal::<bf16>(2, 128, 0.9, false);
}

// Low min_p — keeps most tokens
#[uzu_test]
fn test_low_min_p_f32() {
    test_internal::<f32>(2, 128, 0.01, false);
}

#[uzu_test]
fn test_low_min_p_f16() {
    test_internal::<f16>(2, 128, 0.01, false);
}

#[uzu_test]
fn test_low_min_p_bf16() {
    test_internal::<bf16>(2, 128, 0.01, false);
}

// Single batch
#[uzu_test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 256, 0.1, false);
}

// Large vocab
#[uzu_test]
fn test_large_vocab_f32() {
    test_internal::<f32>(2, 2048, 0.1, false);
}
