use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            kernel::{AttentionTwoPass1Kernel, AttentionTwoPass2Kernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

const TOTAL_BLOCKS_COUNT: u32 = 32;

// --- First pass types and helpers ---

struct FirstPassInput<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: u32,
    gqa_factor: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    scale: f32,
    do_causal: bool,
}

struct FirstPassOutput {
    partials: Vec<f32>,
    sums: Vec<f32>,
    maxs: Vec<f32>,
}

fn get_first_pass_input<T: ArrayElement + Float>(
    num_heads: u32,
    num_kv_heads: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    do_causal: bool,
) -> FirstPassInput<T> {
    let gqa_factor = num_heads / num_kv_heads;

    let q_size = (num_heads * suffix_length * head_dim) as usize;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    let k_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    let v_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut values = vec![T::zero(); v_size];
    for i in 0..v_size {
        values[i] = T::from((i as f32 * 0.11 + 2.0).sin() * 0.5).unwrap();
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    FirstPassInput {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        gqa_factor,
        sequence_length,
        suffix_length,
        head_dim,
        scale,
        do_causal,
    }
}

fn get_first_pass_output<T: ArrayElement + Float, B: Backend>(input: &FirstPassInput<T>) -> FirstPassOutput {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
        &context,
        T::data_type(),
        input.head_dim,
        false,
        false,
        input.do_causal,
        false,
        false,
    )
    .expect("Failed to create AttentionTwoPass1Kernel");

    let queries = alloc_allocation_with_data::<B, T>(&context, &input.queries);
    let keys = alloc_allocation_with_data::<B, T>(&context, &input.keys);
    let values = alloc_allocation_with_data::<B, T>(&context, &input.values);

    let total_offsets = (input.suffix_length * input.num_heads) as usize;
    let partials_size = total_offsets * TOTAL_BLOCKS_COUNT as usize * input.head_dim as usize;
    let sums_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;
    let maxs_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;

    let mut partials = alloc_allocation::<B, f32>(&context, partials_size);
    let mut sums = alloc_allocation::<B, f32>(&context, sums_size);
    let mut maxs = alloc_allocation::<B, f32>(&context, maxs_size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &queries,
        &keys,
        &values,
        &mut partials,
        &mut sums,
        &mut maxs,
        input.gqa_factor,
        input.sequence_length,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.sequence_length * input.head_dim,
        input.head_dim,
        None,
        input.scale,
        input.num_heads,
        input.suffix_length,
        None::<&Allocation<B>>,
        None,
        None::<&Allocation<B>>,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    FirstPassOutput {
        partials: allocation_to_vec(&partials),
        sums: allocation_to_vec(&sums),
        maxs: allocation_to_vec(&maxs),
    }
}

// --- Second pass types and helpers ---

struct SecondPassInput {
    partials: Box<[f32]>,
    sums: Box<[f32]>,
    maxs: Box<[f32]>,
    num_heads: u32,
    suffix_length: u32,
    head_dim: u32,
}

fn get_second_pass_input(
    num_heads: u32,
    suffix_length: u32,
    head_dim: u32,
) -> SecondPassInput {
    let total_offsets = (suffix_length * num_heads) as usize;
    let partials_size = total_offsets * TOTAL_BLOCKS_COUNT as usize * head_dim as usize;
    let sums_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;
    let maxs_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;

    let mut partials = vec![0.0f32; partials_size];
    for i in 0..partials_size {
        partials[i] = (i as f32 * 0.03 + 0.1).sin() * 0.5;
    }

    let mut sums = vec![0.0f32; sums_size];
    for i in 0..sums_size {
        // Ensure sums are positive (they represent sum of exp scores)
        sums[i] = (i as f32 * 0.07 + 1.0).cos().abs() + 0.1;
    }

    let mut maxs = vec![0.0f32; maxs_size];
    for i in 0..maxs_size {
        maxs[i] = (i as f32 * 0.05 + 0.5).sin() * 2.0;
    }

    SecondPassInput {
        partials: partials.into_boxed_slice(),
        sums: sums.into_boxed_slice(),
        maxs: maxs.into_boxed_slice(),
        num_heads,
        suffix_length,
        head_dim,
    }
}

fn get_second_pass_output<T: ArrayElement + Float, B: Backend>(input: &SecondPassInput) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel =
        <<B as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel::new(&context, T::data_type(), input.head_dim)
            .expect("Failed to create AttentionTwoPass2Kernel");

    let partials = alloc_allocation_with_data::<B, f32>(&context, &input.partials);
    let sums = alloc_allocation_with_data::<B, f32>(&context, &input.sums);
    let maxs = alloc_allocation_with_data::<B, f32>(&context, &input.maxs);

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let mut output = alloc_allocation::<B, T>(&context, output_size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&partials, &sums, &maxs, &mut output, input.num_heads, input.suffix_length, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&output)
}

// --- First pass tests ---

fn test_first_pass_internal<T: ArrayElement + Float + Debug + Display>(
    input: &FirstPassInput<T>,
    expected: &FirstPassOutput,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_first_pass_output::<T, B>(input);
        let msg = format!(
            "AttentionTwoPass1 failed (backend={}, heads={}, seq={}, suffix={}, head_dim={}, causal={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
            input.do_causal,
        );
        assert_eq_float::<f32>(&expected.partials, &output.partials, eps, &format!("{msg} [partials]"));
        assert_eq_float::<f32>(&expected.sums, &output.sums, eps, &format!("{msg} [sums]"));
        assert_eq_float::<f32>(&expected.maxs, &output.maxs, eps, &format!("{msg} [maxs]"));
    });
}

fn test_first_pass_basic<T: ArrayElement + Float + Debug + Display>() {
    let input = get_first_pass_input::<T>(4, 4, 8, 1, 64, false);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);

    let input = get_first_pass_input::<T>(4, 4, 8, 4, 64, false);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);
}

fn test_first_pass_causal<T: ArrayElement + Float + Debug + Display>() {
    let input = get_first_pass_input::<T>(4, 4, 16, 1, 64, true);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);

    let input = get_first_pass_input::<T>(4, 4, 8, 4, 64, true);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);
}

fn test_first_pass_gqa<T: ArrayElement + Float + Debug + Display>() {
    let input = get_first_pass_input::<T>(8, 2, 8, 1, 64, false);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);

    let input = get_first_pass_input::<T>(8, 2, 8, 4, 64, true);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);
}

fn test_first_pass_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: u32) {
    let input = get_first_pass_input::<T>(4, 4, 8, 2, head_dim, true);
    let expected = get_first_pass_output::<T, Cpu>(&input);
    test_first_pass_internal(&input, &expected);
}

// --- Second pass tests ---

fn test_second_pass_internal<T: ArrayElement + Float + Debug + Display>(
    input: &SecondPassInput,
    expected: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_second_pass_output::<T, B>(input);
        let msg = format!(
            "AttentionTwoPass2 failed (backend={}, heads={}, suffix={}, head_dim={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.suffix_length,
            input.head_dim,
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_second_pass_basic<T: ArrayElement + Float + Debug + Display>() {
    let input = get_second_pass_input(4, 1, 64);
    let expected = get_second_pass_output::<T, Cpu>(&input);
    test_second_pass_internal::<T>(&input, &expected);

    let input = get_second_pass_input(4, 4, 64);
    let expected = get_second_pass_output::<T, Cpu>(&input);
    test_second_pass_internal::<T>(&input, &expected);
}

fn test_second_pass_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: u32) {
    let input = get_second_pass_input(4, 2, head_dim);
    let expected = get_second_pass_output::<T, Cpu>(&input);
    test_second_pass_internal::<T>(&input, &expected);
}

// --- First pass test entries ---

#[uzu_test]
fn test_first_pass_basic_f32() {
    test_first_pass_basic::<f32>();
}

#[uzu_test]
fn test_first_pass_basic_f16() {
    test_first_pass_basic::<f16>();
}

#[uzu_test]
fn test_first_pass_basic_bf16() {
    test_first_pass_basic::<bf16>();
}

#[uzu_test]
fn test_first_pass_causal_f32() {
    test_first_pass_causal::<f32>();
}

#[uzu_test]
fn test_first_pass_causal_f16() {
    test_first_pass_causal::<f16>();
}

#[uzu_test]
fn test_first_pass_causal_bf16() {
    test_first_pass_causal::<bf16>();
}

#[uzu_test]
fn test_first_pass_gqa_f32() {
    test_first_pass_gqa::<f32>();
}

#[uzu_test]
fn test_first_pass_gqa_f16() {
    test_first_pass_gqa::<f16>();
}

#[uzu_test]
fn test_first_pass_gqa_bf16() {
    test_first_pass_gqa::<bf16>();
}

#[uzu_test]
fn test_first_pass_head_dim_128_f32() {
    test_first_pass_head_dim::<f32>(128);
}

#[uzu_test]
fn test_first_pass_head_dim_128_f16() {
    test_first_pass_head_dim::<f16>(128);
}

#[uzu_test]
fn test_first_pass_head_dim_128_bf16() {
    test_first_pass_head_dim::<bf16>(128);
}

#[uzu_test]
fn test_first_pass_head_dim_512_f32() {
    test_first_pass_head_dim::<f32>(512);
}

#[uzu_test]
fn test_first_pass_head_dim_512_f16() {
    test_first_pass_head_dim::<f16>(512);
}

#[uzu_test]
fn test_first_pass_head_dim_512_bf16() {
    test_first_pass_head_dim::<bf16>(512);
}

// --- Second pass test entries ---

#[uzu_test]
fn test_second_pass_basic_f32() {
    test_second_pass_basic::<f32>();
}

#[uzu_test]
fn test_second_pass_basic_f16() {
    test_second_pass_basic::<f16>();
}

#[uzu_test]
fn test_second_pass_basic_bf16() {
    test_second_pass_basic::<bf16>();
}

#[uzu_test]
fn test_second_pass_head_dim_128_f32() {
    test_second_pass_head_dim::<f32>(128);
}

#[uzu_test]
fn test_second_pass_head_dim_128_f16() {
    test_second_pass_head_dim::<f16>(128);
}

#[uzu_test]
fn test_second_pass_head_dim_128_bf16() {
    test_second_pass_head_dim::<bf16>(128);
}

#[uzu_test]
fn test_second_pass_head_dim_512_f32() {
    test_second_pass_head_dim::<f32>(512);
}

#[uzu_test]
fn test_second_pass_head_dim_512_f16() {
    test_second_pass_head_dim::<f16>(512);
}

#[uzu_test]
fn test_second_pass_head_dim_512_bf16() {
    test_second_pass_head_dim::<bf16>(512);
}
