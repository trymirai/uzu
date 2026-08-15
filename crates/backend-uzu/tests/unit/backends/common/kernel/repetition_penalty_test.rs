use std::fmt::Display;

use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::RepetitionPenaltyKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    context_ring: Box<[u32]>,
    token_ids: Box<[u32]>,
    repetition_penalty: f32,
    suffix_repetition_length: u32,
    vocab_size: u32,
    sampling_start: u32,
    sampling_length: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel::new(&context, T::data_type())
        .expect("Failed to create RepetitionPenaltyKernel");

    let original_logits = alloc_allocation_with_data::<B, T>(&context, &input.logits);
    let mut copy = alloc_allocation_with_data::<B, T>(&context, &input.logits);
    let context_ring = alloc_allocation_with_data::<B, u32>(&context, &input.context_ring);
    let token_ids = alloc_allocation_with_data::<B, u32>(&context, &input.token_ids);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &original_logits,
        &mut copy,
        &context_ring,
        &token_ids,
        input.repetition_penalty,
        input.suffix_repetition_length,
        input.vocab_size,
        input.sampling_start,
        input.sampling_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    allocation_to_vec(&copy)
}

fn ascending_logits<T: ArrayElement + Float>(len: usize) -> Box<[T]> {
    (0..len).map(|index| T::from(1.0 + index as f32).unwrap()).collect()
}

fn context_ring(
    ring_offset: u32,
    ring_length: u32,
    slots: &[u32],
) -> Box<[u32]> {
    [&[ring_offset, ring_length], slots].concat().into_boxed_slice()
}

fn penalized<T: ArrayElement + Float>(
    logit: T,
    repetition_penalty: f32,
) -> T {
    T::from(logit.to_f32().unwrap() / repetition_penalty).unwrap()
}

fn check<T: ArrayElement + Float + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    assert_eq_float::<T>(expected, &get_output::<T, Cpu>(input), 1e-5, "RepetitionPenalty failed with backend=Cpu");
    for_each_non_cpu_backend!(|B| {
        let msg = format!("RepetitionPenalty failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(expected, &get_output::<T, B>(input), 1e-5, &msg);
    });
}

#[uzu_test]
fn test_out_of_vocab() {
    let logits = ascending_logits::<f32>(8);
    let expected = logits.to_vec();
    let input = Input {
        logits,
        context_ring: context_ring(0, 0, &[0; 8]),
        token_ids: vec![3_000_000].into_boxed_slice(),
        repetition_penalty: 2.0,
        suffix_repetition_length: 8,
        vocab_size: 8,
        sampling_start: 0,
        sampling_length: 1,
    };
    check(&input, &expected);
}

#[uzu_test]
fn test_boundary() {
    let (vocab_size, penalty) = (4u32, 2.0f32);
    let logits = ascending_logits::<f32>((vocab_size * 2) as usize);
    let valid_offset = vocab_size as usize + 2;
    let mut expected = logits.to_vec();
    expected[valid_offset] = penalized(expected[valid_offset], penalty);
    let input = Input {
        logits,
        context_ring: context_ring(0, 0, &[0; 4]),
        token_ids: vec![vocab_size, 2].into_boxed_slice(),
        repetition_penalty: penalty,
        suffix_repetition_length: 1,
        vocab_size,
        sampling_start: 0,
        sampling_length: 2,
    };
    check(&input, &expected);
}
