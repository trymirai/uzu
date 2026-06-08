use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use super::MoeGather;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Encoder},
        cpu::Cpu,
    },
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, create_context},
    },
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
    let gather = MoeGather::<B>::new(&context, T::data_type()).expect("MoeGather::new");
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let x_perm_allocation = gather
        .encode(
            &x_allocation,
            &ids_allocation,
            &sumk_allocation,
            input.t,
            input.sum_k / input.t,
            input.d_model,
            &mut encoder,
        )
        .expect("Failed to encode MoE gather");
    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output = allocation_to_vec::<B, T>(&x_perm_allocation);
    drop(x_perm_allocation);
    drop(completed);
    output
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

#[test]
fn test_gather_single_token() {
    test_gather_internal(1, 2, 64);
}

#[test]
fn test_gather_small_batch() {
    test_gather_internal(4, 8, 128);
}

#[test]
fn test_gather_medium_batch() {
    test_gather_internal(16, 32, 256)
}

#[test]
fn test_gather_large_batch() {
    test_gather_internal(64, 128, 512)
}
