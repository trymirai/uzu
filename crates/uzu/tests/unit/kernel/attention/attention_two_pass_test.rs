use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{AttentionTwoPass1Kernel, AttentionTwoPass2Kernel},
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

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

    let queries_array = context.create_array_from(&[input.queries.len()], &input.queries, "");
    let keys_array = context.create_array_from(&[input.keys.len()], &input.keys, "");
    let values_array = context.create_array_from(&[input.values.len()], &input.values, "");

    let total_offsets = (input.suffix_length * input.num_heads) as usize;
    let partials_size = total_offsets * TOTAL_BLOCKS_COUNT as usize * input.head_dim as usize;
    let sums_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;
    let maxs_size = total_offsets * TOTAL_BLOCKS_COUNT as usize;

    let partials_array = context.create_array_uninitialized(&[partials_size], DataType::F32, "");
    let sums_array = context.create_array_uninitialized(&[sums_size], DataType::F32, "");
    let maxs_array = context.create_array_uninitialized(&[maxs_size], DataType::F32, "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        queries_array.buffer().borrow().deref(),
        keys_array.buffer().borrow().deref(),
        values_array.buffer().borrow().deref(),
        partials_array.buffer().borrow_mut().deref_mut(),
        sums_array.buffer().borrow_mut().deref_mut(),
        maxs_array.buffer().borrow_mut().deref_mut(),
        input.gqa_factor,
        input.sequence_length,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.sequence_length * input.head_dim,
        input.head_dim,
        None,
        input.scale,
        input.num_heads,
        input.num_heads / input.gqa_factor,
        input.suffix_length,
        None::<&B::Buffer>,
        None,
        None::<&B::Buffer>,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    FirstPassOutput {
        partials: partials_array.as_slice().to_vec(),
        sums: sums_array.as_slice().to_vec(),
        maxs: maxs_array.as_slice().to_vec(),
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

    let partials_array = context.create_array_from(&[input.partials.len()], &input.partials, "");
    let sums_array = context.create_array_from(&[input.sums.len()], &input.sums, "");
    let maxs_array = context.create_array_from(&[input.maxs.len()], &input.maxs, "");

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let output_array = context.create_array_uninitialized(&[output_size], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        partials_array.buffer().borrow().deref(),
        sums_array.buffer().borrow().deref(),
        maxs_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.num_heads,
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
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

fn test_first_pass_head_dim_128<T: ArrayElement + Float + Debug + Display>() {
    let input = get_first_pass_input::<T>(4, 4, 8, 2, 128, true);
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

fn test_second_pass_head_dim_128<T: ArrayElement + Float + Debug + Display>() {
    let input = get_second_pass_input(4, 2, 128);
    let expected = get_second_pass_output::<T, Cpu>(&input);
    test_second_pass_internal::<T>(&input, &expected);
}

// --- First pass test entries ---

#[test]
fn test_first_pass_basic_f32() {
    test_first_pass_basic::<f32>();
}

#[test]
fn test_first_pass_basic_f16() {
    test_first_pass_basic::<f16>();
}

#[test]
fn test_first_pass_basic_bf16() {
    test_first_pass_basic::<bf16>();
}

#[test]
fn test_first_pass_causal_f32() {
    test_first_pass_causal::<f32>();
}

#[test]
fn test_first_pass_causal_f16() {
    test_first_pass_causal::<f16>();
}

#[test]
fn test_first_pass_causal_bf16() {
    test_first_pass_causal::<bf16>();
}

#[test]
fn test_first_pass_gqa_f32() {
    test_first_pass_gqa::<f32>();
}

#[test]
fn test_first_pass_gqa_f16() {
    test_first_pass_gqa::<f16>();
}

#[test]
fn test_first_pass_gqa_bf16() {
    test_first_pass_gqa::<bf16>();
}

#[test]
fn test_first_pass_head_dim_128_f32() {
    test_first_pass_head_dim_128::<f32>();
}

#[test]
fn test_first_pass_head_dim_128_f16() {
    test_first_pass_head_dim_128::<f16>();
}

#[test]
fn test_first_pass_head_dim_128_bf16() {
    test_first_pass_head_dim_128::<bf16>();
}

// --- Second pass test entries ---

#[test]
fn test_second_pass_basic_f32() {
    test_second_pass_basic::<f32>();
}

#[test]
fn test_second_pass_basic_f16() {
    test_second_pass_basic::<f16>();
}

#[test]
fn test_second_pass_basic_bf16() {
    test_second_pass_basic::<bf16>();
}

#[test]
fn test_second_pass_head_dim_128_f32() {
    test_second_pass_head_dim_128::<f32>();
}

#[test]
fn test_second_pass_head_dim_128_f16() {
    test_second_pass_head_dim_128::<f16>();
}

#[test]
fn test_second_pass_head_dim_128_bf16() {
    test_second_pass_head_dim_128::<bf16>();
}
