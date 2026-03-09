use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels, kernel::GumbelKernel,
        },
        cpu::Cpu,
    },
    language_model::gumbel::{gumbel_float, revidx},
};

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    seeds: Box<[u64]>,
    batch_size: u32,
    vocab_size: u32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * vocab_size) as usize;
    let mut logits: Vec<T> = vec![T::zero(); len];
    for i in 0..len {
        logits[i] = T::from((i as f32 * 0.1).sin() * 2.0f32).unwrap();
    }

    let seeds: Vec<u64> = (0..batch_size as u64).collect();

    let input = Input {
        logits: logits.into_boxed_slice(),
        seeds: seeds.into_boxed_slice(),
        batch_size,
        vocab_size,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::GumbelKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create GumbelKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let logits_array_buffer_rc = logits_array.buffer();
    let logits_array_borrow = logits_array_buffer_rc.borrow();
    let logits_array_deref = logits_array_borrow.deref();
    let logits_buffer = (!input.in_place).then(|| logits_array_deref);

    let seeds_array = context.create_array_from(&[input.batch_size as usize], &input.seeds, "");

    let output_array = match input.in_place {
        true => context.create_array_from(&[len], &input.logits, ""),
        false => context.create_array_uninitialized(&[len], T::data_type(), ""),
    };

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        logits_buffer,
        seeds_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        &mut command_buffer,
    );

    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

// Verify that the CPU kernel produces valid Gumbel noise:
// output should differ from input and be finite.
fn test_cpu_produces_valid_output<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    in_place: bool,
) {
    let (input, output) = get_test_data::<T>(batch_size, vocab_size, in_place);
    let len = (batch_size * vocab_size) as usize;

    for i in 0..len {
        let out = output[i].to_f32().unwrap();
        assert!(out.is_finite(), "Output at index {} is not finite: {}", i, out);
    }

    // At least some outputs should differ from logits (Gumbel noise was added)
    let num_different = (0..len).filter(|&i| output[i] != input.logits[i]).count();
    assert!(num_different > len / 2, "Too few outputs differ from logits: {}/{}", num_different, len);
}

// Verify determinism: same seeds produce same output.
fn test_determinism<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
) {
    let (input, output1) = get_test_data::<T>(batch_size, vocab_size, false);
    let output2 = get_output::<T, Cpu>(&input);
    assert_eq!(output1, output2, "Determinism check failed");
}

// Test that Metal output matches the gumbel_float/revidx reference implementation.
fn test_gpu_reference_match(
    batch_size: usize,
    vocab_size: usize,
) {
    const RTOL: f32 = 0.01;
    const ATOL: f32 = 1e-6;

    let logits = vec![0.0f32; batch_size * vocab_size];
    let seeds: Vec<u64> = (0_u64..batch_size as u64).collect();

    let input = Input {
        logits: logits.into_boxed_slice(),
        seeds: seeds.clone().into_boxed_slice(),
        batch_size: batch_size as u32,
        vocab_size: vocab_size as u32,
        in_place: false,
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<f32, B>(&input);

        for (batch_idx, batch_seed) in seeds.iter().copied().enumerate() {
            let results = &output[batch_idx * vocab_size..(batch_idx + 1) * vocab_size];
            for (logit_idx, gpu_logit_value) in results.iter().copied().enumerate() {
                let cpu_logit_value = gumbel_float(batch_seed, revidx(logit_idx as u32));
                let abs_diff = (cpu_logit_value - gpu_logit_value).abs();
                let tolerance = ATOL + RTOL * cpu_logit_value.abs();
                assert!(
                    abs_diff <= tolerance,
                    "Mismatch at batch {batch_idx} element {logit_idx}: CPU={cpu_logit_value} GPU={gpu_logit_value} (abs_diff={abs_diff}, tolerance={tolerance})"
                );
            }
        }
    });
}

#[test]
fn test_f32() {
    test_cpu_produces_valid_output::<f32>(4, 128, false);
}

#[test]
fn test_f16() {
    test_cpu_produces_valid_output::<f16>(4, 128, false);
}

#[test]
fn test_bf16() {
    test_cpu_produces_valid_output::<bf16>(4, 128, false);
}

#[test]
fn test_in_place_f32() {
    test_cpu_produces_valid_output::<f32>(4, 128, true);
}

#[test]
fn test_in_place_f16() {
    test_cpu_produces_valid_output::<f16>(4, 128, true);
}

#[test]
fn test_in_place_bf16() {
    test_cpu_produces_valid_output::<bf16>(4, 128, true);
}

#[test]
fn test_determinism_f32() {
    test_determinism::<f32>(4, 128);
}

#[test]
fn test_single_batch_f32() {
    test_cpu_produces_valid_output::<f32>(1, 256, false);
}

#[test]
fn test_gumbel_gpu_cpu_match() {
    test_gpu_reference_match(7, 16 * 1024 * 64);
}
