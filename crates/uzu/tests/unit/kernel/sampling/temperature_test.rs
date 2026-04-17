use std::{
    fmt::{Debug, Display},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::TemperatureKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * vocab_size) as usize;
    let mut logits: Vec<T> = vec![T::zero(); len];
    for i in 0..len {
        logits[i] = T::from((i as f32 * 0.1).sin() * 2.0f32).unwrap();
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        batch_size,
        vocab_size,
        temperature,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TemperatureKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create TemperatureKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let input_logits = (!input.in_place).then(|| context.create_array_from(&[len], &input.logits, "").into_allocation());
    let mut output = match input.in_place {
        true => context.create_array_from(&[len], &input.logits, "").into_allocation(),
        false => context.create_array_uninitialized(&[len], T::data_type(), "").into_allocation(),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_logits.as_ref(),
        &mut output,
        input.batch_size,
        input.vocab_size,
        input.temperature,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().unwrap();

    crate::common::helpers::allocation_to_vec(&output)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, vocab_size, temperature, in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Results are not equal for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

// Out-of-place tests
#[uzu_test]
fn test_f32() {
    test_internal::<f32>(4, 128, 0.7, false);
}

#[uzu_test]
fn test_f16() {
    test_internal::<f16>(4, 128, 0.7, false);
}

#[uzu_test]
fn test_bf16() {
    test_internal::<bf16>(4, 128, 0.7, false);
}

// In-place tests
#[uzu_test]
fn test_in_place_f32() {
    test_internal::<f32>(4, 128, 0.7, true);
}

#[uzu_test]
fn test_in_place_f16() {
    test_internal::<f16>(4, 128, 0.7, true);
}

#[uzu_test]
fn test_in_place_bf16() {
    test_internal::<bf16>(4, 128, 0.7, true);
}

// Edge cases
#[uzu_test]
fn test_temperature_1_f32() {
    test_internal::<f32>(2, 64, 1.0, false);
}

#[uzu_test]
fn test_high_temperature_f32() {
    test_internal::<f32>(2, 64, 100.0, false);
}

#[uzu_test]
fn test_low_temperature_f32() {
    test_internal::<f32>(2, 64, 0.1, false);
}

#[uzu_test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 256, 0.7, false);
}
