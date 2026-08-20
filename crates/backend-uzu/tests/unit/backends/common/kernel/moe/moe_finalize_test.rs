use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Encoder, Kernels, kernel::MoeFinalizeKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context, for_each_non_cpu_backend,
        },
    },
};

struct Input<T: ArrayElement + Float> {
    probs: Box<[T]>,
    route_outputs: Box<[T]>,
    t: usize,
    d_model: usize,
    k: usize,
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = create_context::<B>();
    let probs = alloc_allocation_with_data::<B, T>(&context, &input.probs);
    let route_outputs = alloc_allocation_with_data::<B, T>(&context, &input.route_outputs);
    let mut y_out = alloc_allocation::<B, T>(&context, input.t * input.d_model);

    let finalize = <B::Kernels as Kernels>::MoeFinalizeKernel::new(&context, DataType::BF16).expect("finalize kernel");
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    finalize.encode(
        &probs,
        &route_outputs,
        &mut y_out,
        input.t as u32,
        input.d_model as u32,
        input.k as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&y_out)
}

fn test_finalize_internal(
    t: usize,
    k: usize,
    d_model: usize,
) {
    let mut rng = StdRng::seed_from_u64(2026);

    let probs: Vec<bf16> = (0..t * k).map(|_| bf16::from_f32(rng.random_range(0.0..1.0))).collect();
    let route_outputs: Vec<bf16> = (0..t * k * d_model).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();

    let input = Input {
        probs: probs.into_boxed_slice(),
        route_outputs: route_outputs.into_boxed_slice(),
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
    test_finalize_internal(1, 2, 64)
}

#[uzu_test]
fn test_finalize_small_batch() {
    test_finalize_internal(4, 2, 128)
}

#[uzu_test]
fn test_finalize_medium() {
    test_finalize_internal(8, 4, 256)
}

#[uzu_test]
fn test_finalize_large() {
    test_finalize_internal(16, 2, 512)
}
