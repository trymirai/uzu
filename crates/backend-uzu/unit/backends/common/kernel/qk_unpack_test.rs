use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use test_macros::uzu_test;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::QkUnpackKernel},
        cpu::Cpu,
    },
    common::assert::assert_eq_float,
};

fn run<T: ArrayElement + Float + Debug + Display, B: Backend>(
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    suffix_length: u32,
) -> (Vec<T>, Vec<T>) {
    let total_heads = (num_heads + 2 * num_groups) as usize;
    let qkv_len = suffix_length as usize * total_heads * head_dim as usize;
    let qkv: Box<[T]> = (0..qkv_len).map(|index| T::from(((index as f32) * 0.1).sin() * 2.0).unwrap()).collect();

    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::QkUnpackKernel::new(&context, T::data_type())
        .expect("Failed to create QkUnpackKernel");

    let queries_len = num_heads as usize * suffix_length as usize * head_dim as usize;
    let keys_len = num_groups as usize * suffix_length as usize * head_dim as usize;
    let qkv_array = context.create_array_from(&[qkv_len], &qkv);
    let mut queries = context.create_array_uninitialized(&[queries_len], T::data_type()).into_allocation();
    let mut keys = context.create_array_uninitialized(&[keys_len], T::data_type()).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        qkv_array.allocation(),
        &mut queries,
        &mut keys,
        head_dim,
        num_heads,
        num_groups,
        suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (crate::common::helpers::allocation_to_vec(&queries), crate::common::helpers::allocation_to_vec(&keys))
}

fn check<T: ArrayElement + Float + Debug + Display>(
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    suffix_length: u32,
) {
    let (expected_queries, expected_keys) = run::<T, Cpu>(num_heads, num_groups, head_dim, suffix_length);
    for_each_non_cpu_backend!(|B| {
        let (queries, keys) = run::<T, B>(num_heads, num_groups, head_dim, suffix_length);
        assert_eq_float::<T>(&expected_queries, &queries, 0.0, "QkUnpack queries");
        assert_eq_float::<T>(&expected_keys, &keys, 0.0, "QkUnpack keys");
    });
}

#[uzu_test]
fn test_qk_unpack_gqa_f32() {
    check::<f32>(8, 2, 64, 4);
}

#[uzu_test]
fn test_qk_unpack_single_token_f32() {
    check::<f32>(16, 4, 64, 1);
}

#[uzu_test]
fn test_qk_unpack_bf16() {
    check::<bf16>(8, 2, 64, 4);
}
