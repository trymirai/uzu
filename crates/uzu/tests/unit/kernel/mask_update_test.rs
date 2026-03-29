use std::{
    fmt::{Debug, Display},
    ops::DerefMut,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MaskUpdateKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    mask: Box<[T]>,
    unmask_col: i32,
    mask_col: i32,
}

fn make_mask<T: ArrayElement + Float>(size: usize) -> Vec<T> {
    let mut mask = Vec::with_capacity(size);
    for i in 0..size {
        mask.push(T::from(i as f32 * 0.5).unwrap());
    }
    mask
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::MaskUpdateKernel::new(&context, T::data_type())
        .expect("Failed to create MaskUpdateKernel");

    let len = input.mask.len();
    let mask_array = context.create_array_from(&[len], &input.mask, "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
    kernel.encode(mask_array.buffer().borrow_mut().deref_mut(), input.unmask_col, input.mask_col, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    mask_array.as_slice().to_vec()
}

/// Unmask one column and mask another.
fn get_test_data_unmask_and_mask<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let size = 8;
    let mask = make_mask::<T>(size);

    let input = Input {
        mask: mask.into_boxed_slice(),
        unmask_col: 2,
        mask_col: 5,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Only unmask (mask_col negative).
fn get_test_data_unmask_only<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let size = 8;
    let mask = make_mask::<T>(size);

    let input = Input {
        mask: mask.into_boxed_slice(),
        unmask_col: 3,
        mask_col: -1,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Only mask (unmask_col negative).
fn get_test_data_mask_only<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let size = 8;
    let mask = make_mask::<T>(size);

    let input = Input {
        mask: mask.into_boxed_slice(),
        unmask_col: -1,
        mask_col: 4,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Both columns negative — data should remain unchanged.
fn get_test_data_no_op<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let size = 8;
    let mask = make_mask::<T>(size);

    let expected = mask.clone();

    let input = Input {
        mask: mask.into_boxed_slice(),
        unmask_col: -1,
        mask_col: -1,
    };

    (input, expected)
}

/// Larger mask to exercise more threads.
fn get_test_data_large<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let size = 1024;
    let mask = make_mask::<T>(size);

    let input = Input {
        mask: mask.into_boxed_slice(),
        unmask_col: 100,
        mask_col: 900,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    test_name: &str,
) {
    let eps = 1e-5;
    let expected_norm = expected.to_vec();
    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(input);
        let msg = format!("MaskUpdate {} failed with backend={}", test_name, std::any::type_name::<B>());
        assert_eq_float::<T>(&expected_norm, &actual, eps, &msg);
    });
}

fn test_unmask_and_mask_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_unmask_and_mask::<T>();
    test_internal::<T>(&input, &expected, "unmask_and_mask");
}

fn test_unmask_only_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_unmask_only::<T>();
    test_internal::<T>(&input, &expected, "unmask_only");
}

fn test_mask_only_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_mask_only::<T>();
    test_internal::<T>(&input, &expected, "mask_only");
}

fn test_no_op_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_no_op::<T>();
    test_internal::<T>(&input, &expected, "no_op");
}

fn test_large_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_large::<T>();
    test_internal::<T>(&input, &expected, "large");
}

// Unmask and mask tests
#[test]
fn test_unmask_and_mask_f32() {
    test_unmask_and_mask_internal::<f32>();
}

#[test]
fn test_unmask_and_mask_f16() {
    test_unmask_and_mask_internal::<f16>();
}

#[test]
fn test_unmask_and_mask_bf16() {
    test_unmask_and_mask_internal::<bf16>();
}

// Unmask only tests
#[test]
fn test_unmask_only_f32() {
    test_unmask_only_internal::<f32>();
}

#[test]
fn test_unmask_only_f16() {
    test_unmask_only_internal::<f16>();
}

#[test]
fn test_unmask_only_bf16() {
    test_unmask_only_internal::<bf16>();
}

// Mask only tests
#[test]
fn test_mask_only_f32() {
    test_mask_only_internal::<f32>();
}

#[test]
fn test_mask_only_f16() {
    test_mask_only_internal::<f16>();
}

#[test]
fn test_mask_only_bf16() {
    test_mask_only_internal::<bf16>();
}

// No-op tests
#[test]
fn test_no_op_f32() {
    test_no_op_internal::<f32>();
}

#[test]
fn test_no_op_f16() {
    test_no_op_internal::<f16>();
}

#[test]
fn test_no_op_bf16() {
    test_no_op_internal::<bf16>();
}

// Large dimension tests
#[test]
fn test_large_f32() {
    test_large_internal::<f32>();
}

#[test]
fn test_large_f16() {
    test_large_internal::<f16>();
}

#[test]
fn test_large_bf16() {
    test_large_internal::<bf16>();
}
