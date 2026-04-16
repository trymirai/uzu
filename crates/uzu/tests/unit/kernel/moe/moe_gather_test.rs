use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Encoder,
            kernel::moe::{MoeGatherArguments, MoeGatherKernels},
        },
        cpu::Cpu,
    },
};

use crate::{
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context},
    },
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    bucket_ids: Box<[i32]>,
    t: usize,
    sum_k: usize,
    d_model: usize,
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = create_context::<B>();

    let sumk_data: [u32; 1] = [input.sum_k as u32];
    let x_allocation = alloc_allocation_with_data::<B, T>(&context, &input.x);
    let ids_allocation = alloc_allocation_with_data::<B, i32>(&context, &input.bucket_ids);
    let sumk_allocation = alloc_allocation_with_data::<B, u32>(&context, &sumk_data);
    let mut x_perm_allocation = alloc_allocation::<B, T>(&context, input.sum_k * input.d_model);

    let gather = MoeGatherKernels::<B>::new(&context).expect("MoeGatherKernel::new");
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    gather.encode(
        &mut encoder,
        T::data_type(),
        MoeGatherArguments {
            x: &x_allocation,
            bucketed_ids: &ids_allocation,
            x_perm: &mut x_perm_allocation,
            sumk: &sumk_allocation,
            t: input.t,
            k: input.sum_k / input.t, // Decompose sum_k into k per token
            d_model: input.d_model,
        },
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&x_perm_allocation)
}

fn test_gather_internal(
    t: usize,
    sum_k: usize,
    d_model: usize,
) {
    let mut rng = StdRng::seed_from_u64(2027);
    let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();
    let bucketed_ids: Vec<i32> = (0..sum_k).map(|_| rng.random_range(0..t as i32)).collect();

    let input = Input::<bf16> {
        x: x.clone().into_boxed_slice(),
        bucket_ids: bucketed_ids.clone().into_boxed_slice(),
        t,
        sum_k,
        d_model,
    };

    let cpu_ref_output = get_output::<Cpu, bf16>(&input);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, bf16>(&input);
        assert_eq_float(&cpu_ref_output, &output, 1e-6, "Moe gather");
    })
}

#[uzu_test]
fn test_gather_single_token() {
    test_gather_internal(1, 2, 64);
}

#[uzu_test]
fn test_gather_small_batch() {
    test_gather_internal(4, 8, 128);
}

#[uzu_test]
fn test_gather_medium_batch() {
    test_gather_internal(16, 32, 256)
}

#[uzu_test]
fn test_gather_large_batch() {
    test_gather_internal(64, 128, 512)
}
