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
            Context, Kernels, kernel::TopKKernel,
        },
        cpu::Cpu,
    },
};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    top_k: u32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    top_k: u32,
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
        top_k,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TopKKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create TopKKernel");

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

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        logits_buffer,
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        input.top_k,
        &mut command_buffer,
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn assert_top_k_equal<T: ArrayElement + Float + Display>(
    expected: &[T],
    actual: &[T],
    vocab_size: u32,
    top_k: u32,
    eps: f32,
    msg: &str,
) {
    assert_eq!(expected.len(), actual.len(), "Slices size mismatch");
    let vocab_size = vocab_size as usize;
    let num_batches = expected.len() / vocab_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * vocab_size;
        let end = start + vocab_size;

        // Both should keep approximately top_k values and mask the rest
        let expected_kept: usize = expected[start..end].iter().filter(|v| v.is_finite()).count();
        let actual_kept: usize = actual[start..end].iter().filter(|v| v.is_finite()).count();

        // Allow small differences due to binary search convergence
        let diff = (expected_kept as i64 - actual_kept as i64).unsigned_abs();
        assert!(
            diff <= 3,
            "{msg}. Batch {batch_idx}: expected {expected_kept} kept values, got {actual_kept} (top_k={top_k})"
        );

        // For values that are kept by both, they should match the original values
        for i in start..end {
            let e = expected[i].to_f32().unwrap();
            let a = actual[i].to_f32().unwrap();
            if e.is_finite() && a.is_finite() {
                let d = (e - a).abs();
                assert!(d < eps, "{msg}. Mismatch at index {i}: expected {e}, got {a}, diff {d}");
            }
        }
    }
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    top_k: u32,
    in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, vocab_size, top_k, in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Results are not equal for backend {}", std::any::type_name::<B>());
        assert_top_k_equal::<T>(&expected, &output, vocab_size, top_k, eps, &msg);
    });
}

fn test_cpu_correctness<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    top_k: u32,
    in_place: bool,
) {
    let (input, output) = get_test_data::<T>(batch_size, vocab_size, top_k, in_place);
    let vocab_size = vocab_size as usize;
    let top_k = top_k as usize;

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size;
        let batch_output = &output[batch_start..batch_start + vocab_size];

        // Count non-masked values
        let num_kept: usize = batch_output.iter().filter(|v| v.to_f32().unwrap() > f32::NEG_INFINITY).count();

        // The number of kept values should be approximately top_k
        // (binary search may not be exact due to duplicate values and finite iterations)
        assert!(
            num_kept >= top_k && num_kept <= top_k + 5,
            "batch {batch_idx}: expected ~{top_k} kept values, got {num_kept}"
        );

        // All kept values should match the original logits
        for i in 0..vocab_size {
            let out_val = batch_output[i].to_f32().unwrap();
            if out_val > f32::NEG_INFINITY {
                let in_val = input.logits[batch_start + i].to_f32().unwrap();
                assert!(
                    (out_val - in_val).abs() < 1e-5,
                    "batch {batch_idx}, idx {i}: kept value {out_val} != input {in_val}"
                );
            }
        }
    }
}

// Out-of-place tests
#[test]
fn test_f32() {
    test_internal::<f32>(4, 128, 10, false);
}

#[test]
fn test_f16() {
    test_internal::<f16>(4, 128, 10, false);
}

#[test]
fn test_bf16() {
    test_internal::<bf16>(4, 128, 10, false);
}

// In-place tests
#[test]
fn test_in_place_f32() {
    test_internal::<f32>(4, 128, 10, true);
}

#[test]
fn test_in_place_f16() {
    test_internal::<f16>(4, 128, 10, true);
}

#[test]
fn test_in_place_bf16() {
    test_internal::<bf16>(4, 128, 10, true);
}

// CPU correctness tests
#[test]
fn test_cpu_correctness_f32() {
    test_cpu_correctness::<f32>(4, 128, 10, false);
}

#[test]
fn test_cpu_correctness_small_k() {
    test_cpu_correctness::<f32>(2, 256, 1, false);
}

#[test]
fn test_cpu_correctness_large_k() {
    test_cpu_correctness::<f32>(2, 256, 100, false);
}

#[test]
fn test_cpu_correctness_in_place() {
    test_cpu_correctness::<f32>(4, 128, 10, true);
}

// Edge cases
#[test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 256, 10, false);
}

#[test]
fn test_large_vocab_f32() {
    test_internal::<f32>(2, 1024, 50, false);
}

#[test]
fn test_top_k_1_f32() {
    test_internal::<f32>(2, 128, 1, false);
}

#[test]
fn test_top_k_equals_vocab_f32() {
    test_internal::<f32>(2, 64, 64, false);
}
