use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BitmaskKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    bitmask: Box<[u32]>,
    batch_size: u32,
    vocab_size: u32,
    in_place: bool,
}

fn get_test_data_basic<T: ArrayElement + Float>(in_place: bool) -> (Input<T>, Vec<T>) {
    let batch_size = 2u32;
    let vocab_size = 128u32;

    let len = (batch_size * vocab_size) as usize;
    let logits: Vec<T> = (0..len).map(|i| T::from((i as f32 * 0.1).sin() * 2.0).unwrap()).collect();

    let bitmask_size = ((vocab_size + 31) / 32) as usize;
    let mut bitmask = vec![0u32; batch_size as usize * bitmask_size];
    // Allow every other token
    for b in 0..batch_size as usize {
        for word in 0..bitmask_size {
            bitmask[b * bitmask_size + word] = 0x55555555; // bits 0,2,4,...
        }
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        bitmask: bitmask.into_boxed_slice(),
        batch_size,
        vocab_size,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_all_allowed<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let batch_size = 1u32;
    let vocab_size = 64u32;

    let len = (batch_size * vocab_size) as usize;
    let logits: Vec<T> = (0..len).map(|i| T::from(i as f32 * 0.5).unwrap()).collect();

    let bitmask_size = ((vocab_size + 31) / 32) as usize;
    let bitmask = vec![0xFFFFFFFFu32; batch_size as usize * bitmask_size];

    let input = Input {
        logits: logits.into_boxed_slice(),
        bitmask: bitmask.into_boxed_slice(),
        batch_size,
        vocab_size,
        in_place: false,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_none_allowed<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let batch_size = 1u32;
    let vocab_size = 64u32;

    let len = (batch_size * vocab_size) as usize;
    let logits: Vec<T> = (0..len).map(|i| T::from(i as f32 * 0.5).unwrap()).collect();

    let bitmask_size = ((vocab_size + 31) / 32) as usize;
    let bitmask = vec![0u32; batch_size as usize * bitmask_size];

    let input = Input {
        logits: logits.into_boxed_slice(),
        bitmask: bitmask.into_boxed_slice(),
        batch_size,
        vocab_size,
        in_place: false,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_non_aligned<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    // vocab_size not a multiple of 32
    let batch_size = 2u32;
    let vocab_size = 100u32;

    let len = (batch_size * vocab_size) as usize;
    let logits: Vec<T> = (0..len).map(|i| T::from((i as f32 * 0.3).cos() * 3.0).unwrap()).collect();

    let bitmask_size = ((vocab_size + 31) / 32) as usize;
    let mut bitmask = vec![0u32; batch_size as usize * bitmask_size];
    // Set specific bits
    for b in 0..batch_size as usize {
        for word in 0..bitmask_size {
            bitmask[b * bitmask_size + word] = 0xAAAAAAAA; // bits 1,3,5,...
        }
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        bitmask: bitmask.into_boxed_slice(),
        batch_size,
        vocab_size,
        in_place: false,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::BitmaskKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create BitmaskKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let bitmask_len = input.bitmask.len();

    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let bitmask_array = context.create_array_from(&[bitmask_len], &input.bitmask, "");
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
        bitmask_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_basic<T: ArrayElement + Float + Debug + Display>(in_place: bool) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    let (input, expected) = get_test_data_basic::<T>(in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Bitmask basic test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

fn test_all_allowed<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    let (input, expected) = get_test_data_all_allowed::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Bitmask all-allowed test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

fn test_none_allowed<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    let (input, expected) = get_test_data_none_allowed::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Bitmask none-allowed test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

fn test_non_aligned<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    let (input, expected) = get_test_data_non_aligned::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Bitmask non-aligned test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

// Out-of-place basic tests
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>(false);
}

#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>(false);
}

#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>(false);
}

// In-place basic tests
#[uzu_test]
fn test_in_place_f32() {
    test_basic::<f32>(true);
}

#[uzu_test]
fn test_in_place_f16() {
    test_basic::<f16>(true);
}

#[uzu_test]
fn test_in_place_bf16() {
    test_basic::<bf16>(true);
}

// All-allowed tests
#[uzu_test]
fn test_all_allowed_f32() {
    test_all_allowed::<f32>();
}

#[uzu_test]
fn test_all_allowed_f16() {
    test_all_allowed::<f16>();
}

#[uzu_test]
fn test_all_allowed_bf16() {
    test_all_allowed::<bf16>();
}

// None-allowed tests
#[uzu_test]
fn test_none_allowed_f32() {
    test_none_allowed::<f32>();
}

#[uzu_test]
fn test_none_allowed_f16() {
    test_none_allowed::<f16>();
}

#[uzu_test]
fn test_none_allowed_bf16() {
    test_none_allowed::<bf16>();
}

// Non-aligned vocab_size tests
#[uzu_test]
fn test_non_aligned_f32() {
    test_non_aligned::<f32>();
}

#[uzu_test]
fn test_non_aligned_f16() {
    test_non_aligned::<f16>();
}

#[uzu_test]
fn test_non_aligned_bf16() {
    test_non_aligned::<bf16>();
}
