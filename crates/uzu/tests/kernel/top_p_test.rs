use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayElement, DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels, kernel::TopPKernel,
        },
        cpu::Cpu,
    },
};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * vocab_size) as usize;
    let mut rng = StdRng::seed_from_u64(42);
    let mut logits: Vec<T> = vec![T::zero(); len];
    for x in logits.iter_mut() {
        *x = T::from(rng.random_range(-16.0f32..16.0f32)).unwrap();
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        batch_size,
        vocab_size,
        top_p,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TopPKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create TopPKernel");

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
        input.top_p,
        &mut command_buffer,
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn assert_top_p_equal<T: ArrayElement + Float + Display>(
    expected: &[T],
    actual: &[T],
    vocab_size: u32,
    eps: f32,
    msg: &str,
) {
    assert_eq!(expected.len(), actual.len(), "Slices size mismatch");
    let vocab_size = vocab_size as usize;
    let num_batches = expected.len() / vocab_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * vocab_size;
        let end = start + vocab_size;

        let expected_kept: usize = expected[start..end].iter().filter(|v| v.is_finite()).count();
        let actual_kept: usize = actual[start..end].iter().filter(|v| v.is_finite()).count();

        // Allow small differences due to binary search convergence
        let diff = (expected_kept as i64 - actual_kept as i64).unsigned_abs();
        assert!(diff <= 3, "{msg}. Batch {batch_idx}: expected {expected_kept} kept values, got {actual_kept}");

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
    top_p: f32,
    in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, vocab_size, top_p, in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Results are not equal for backend {}", std::any::type_name::<B>());
        assert_top_p_equal::<T>(&expected, &output, vocab_size, eps, &msg);
    });
}

// Out-of-place tests (same data as sampling_test::test_topp_gpu_cpu_match)
#[test]
fn test_f32() {
    test_internal::<f32>(4, 1024, 0.9, false);
}

#[test]
fn test_f16() {
    test_internal::<f16>(4, 1024, 0.9, false);
}

#[test]
fn test_bf16() {
    test_internal::<bf16>(4, 1024, 0.9, false);
}

// In-place tests
#[test]
fn test_in_place_f32() {
    test_internal::<f32>(4, 1024, 0.9, true);
}

#[test]
fn test_in_place_f16() {
    test_internal::<f16>(4, 1024, 0.9, true);
}

#[test]
fn test_in_place_bf16() {
    test_internal::<bf16>(4, 1024, 0.9, true);
}

// Edge cases
#[test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 1024, 0.9, false);
}

#[test]
fn test_top_p_very_small() {
    test_internal::<f32>(4, 1024, 0.01, false);
}

#[test]
fn test_top_p_near_1() {
    test_internal::<f32>(4, 1024, 0.99, false);
}
