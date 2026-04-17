use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Encoder, Kernels, kernel::MoeFinalizeKernel},
        cpu::Cpu,
    },
};

use crate::{
    common::{assert::assert_eq_float, helpers::create_context},
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    tok2row: Box<[i32]>,
    probs: Box<[T]>,
    y_partial: Box<[T]>,
    t: usize,
    d_model: usize,
    k: usize,
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = create_context::<B>();
    let tok2row_array = context.create_array_from(&[input.tok2row.len()], &input.tok2row, "");
    let probs_array = context.create_array_from(&[input.probs.len()], &input.probs, "");
    let y_partial_array = context.create_array_from(&[input.y_partial.len()], &input.y_partial, "");
    let mut y_out = context.create_array_uninitialized(&[input.t * input.d_model], T::data_type(), "").into_allocation();

    let finalize = <B::Kernels as Kernels>::MoeFinalizeKernel::new(&context, DataType::BF16).expect("finalize kernel");
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    finalize.encode(
        tok2row_array.allocation(),
        probs_array.allocation(),
        y_partial_array.allocation(),
        &mut y_out,
        input.t as u32,
        input.d_model as u32,
        input.k as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    crate::common::helpers::allocation_to_vec(&y_out)
}

fn test_finalize_internal(
    t: usize,
    k: usize,
    d_model: usize,
    sum_k: usize,
) {
    let mut rng = StdRng::seed_from_u64(2026);

    // Generate random tok2row mapping: maps (token, k_idx) → row in y_partial
    // Some entries can be -1 (no expert selected)
    let mut tok2row: Vec<i32> = (0..t * k)
        .map(|_| {
            if rng.random_bool(0.9) {
                rng.random_range(0..sum_k as i32)
            } else {
                -1 // No expert selected
            }
        })
        .collect();

    // Ensure we use all rows in sum_k (avoid unused rows)
    for row in 0..sum_k.min(t * k) {
        tok2row[row] = row as i32;
    }

    // Generate random probabilities (should sum to 1 per token, but not critical for unit test)
    let probs: Vec<bf16> = (0..t * k).map(|_| bf16::from_f32(rng.random_range(0.0..1.0))).collect();

    // Generate random y_partial (expert outputs)
    let y_partial: Vec<bf16> = (0..sum_k * d_model).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();

    // CPU reference
    let input = Input {
        tok2row: tok2row.into_boxed_slice(),
        probs: probs.into_boxed_slice(),
        y_partial: y_partial.into_boxed_slice(),
        t,
        d_model,
        k,
    };
    let y_cpu = get_output::<Cpu, bf16>(&input);

    for_each_non_cpu_backend!(|B| {
        let y_gpu = get_output::<B, bf16>(&input);
        assert_eq_float(&y_cpu, &y_gpu, 1e-2, "finalize output");
    });
}

#[uzu_test]
fn test_finalize_single_token() {
    test_finalize_internal(1, 2, 64, 2)
}

#[uzu_test]
fn test_finalize_small_batch() {
    test_finalize_internal(4, 2, 128, 8)
}

#[uzu_test]
fn test_finalize_medium() {
    test_finalize_internal(8, 4, 256, 32)
}

#[uzu_test]
fn test_finalize_large() {
    test_finalize_internal(16, 2, 512, 32)
}
