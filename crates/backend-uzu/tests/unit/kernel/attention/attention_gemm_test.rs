use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Backend, kernel::ManualKernels},
        cpu::Cpu,
    },
};
use num_traits::Float;

use crate::{
    common::{
        assert::assert_eq_float,
        helpers::{
            FlatAttentionInput, OutputStorage, expected_row_major_matmul, followup_matmul_weights,
            generate_boxed_slice, run_attention_gemm, run_attention_gemm_followed_by_matmul,
        },
    },
    uzu_test,
};

type Input<T> = FlatAttentionInput<T>;

macro_rules! assert_non_cpu_case {
    ($input:expr, $expected:expr, $eps:expr, $label:literal, $runner:ident $(, $arg:expr )* $(,)?) => {{
        let input = $input;
        let expected = $expected;
        let epsilon = $eps;

        for_each_non_cpu_backend!(|B| {
            let output = $runner::<T, B>(input $(, $arg)*);
            let message = format!(
                concat!(
                    $label,
                    " failed (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={}, causal={})"
                ),
                std::any::type_name::<B>(),
                input.num_heads,
                input.num_kv_heads,
                input.sequence_length,
                input.suffix_length,
                input.head_dim,
                input.do_causal,
            );
            assert_eq_float::<T>(expected, &output, epsilon, &message);
        });
    }};
}

macro_rules! define_float_case {
    ($test_name:ident, $runner:ident) => {
        #[uzu_test]
        fn $test_name() {
            for_each_float_type!(|FloatType| {
                $runner::<FloatType>();
            });
        }
    };
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
) -> (Input<T>, Vec<T>) {
    let queries =
        generate_boxed_slice(num_heads * suffix_length * head_dim, |index| (index as f32 * 0.13 + 0.5).sin() * 0.5);
    let keys = generate_boxed_slice(num_kv_heads * sequence_length * head_dim, |index| {
        (index as f32 * 0.07 + 1.0).cos() * 0.5
    });
    let values = generate_boxed_slice(num_kv_heads * sequence_length * head_dim, |index| {
        (index as f32 * 0.11 + 2.0).sin() * 0.5
    });

    let input = Input {
        queries,
        keys,
        values,
        num_heads,
        num_kv_heads,
        sequence_length,
        suffix_length,
        head_dim,
        scale: 1.0 / (head_dim as f32).sqrt(),
        do_causal,
    };
    let expected = run_attention::<T, Cpu>(&input);

    (input, expected)
}

fn epsilon<T: ArrayElement>() -> f32 {
    if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    }
}

fn chained_matmul_epsilon<T: ArrayElement>() -> f32 {
    if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2
    } else {
        1e-4
    }
}

fn run_attention<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    run_attention_gemm::<T, B>(input, input.sequence_length, OutputStorage::Global)
}

fn run_attention_with_padded_kv_cache<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    max_sequence_length: usize,
) -> Vec<T> {
    run_attention_gemm::<T, B>(input, max_sequence_length, OutputStorage::Global)
}

fn run_attention_with_pooled_scratch_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    run_attention_gemm::<T, B>(input, input.sequence_length, OutputStorage::ScratchCopy)
}

fn run_attention_followed_by_matmul<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    output_dim: usize,
) -> Vec<T>
where
    B::Kernels: ManualKernels,
{
    run_attention_gemm_followed_by_matmul::<T, B>(input, output_dim)
}

fn model_like_prefill_data<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data::<T>(32, 8, 38, 38, 64, true)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    assert_non_cpu_case!(input, expected, epsilon::<T>(), "AttentionGemm", run_attention);
}

fn test_basic_for<T: ArrayElement + Float + Debug + Display>() {
    for (sequence_length, suffix_length) in [(8, 1), (8, 4)] {
        let (input, expected) = get_test_data::<T>(4, 4, sequence_length, suffix_length, 64, false);
        test_internal(&input, &expected);
    }
}

fn test_causal_for<T: ArrayElement + Float + Debug + Display>() {
    for (sequence_length, suffix_length) in [(16, 1), (8, 4)] {
        let (input, expected) = get_test_data::<T>(4, 4, sequence_length, suffix_length, 64, true);
        test_internal(&input, &expected);
    }
}

fn test_gqa_for<T: ArrayElement + Float + Debug + Display>() {
    for (suffix_length, do_causal) in [(1, false), (4, true)] {
        let (input, expected) = get_test_data::<T>(8, 2, 8, suffix_length, 64, do_causal);
        test_internal(&input, &expected);
    }
}

fn test_head_dim_128_for<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 2, 128, true);
    test_internal(&input, &expected);
}

fn test_unaligned_for<T: ArrayElement + Float + Debug + Display>() {
    for (sequence_length, suffix_length, do_causal) in [(40, 7, true), (13, 4, false)] {
        let (input, expected) = get_test_data::<T>(4, 4, sequence_length, suffix_length, 64, do_causal);
        test_internal(&input, &expected);
    }
}

fn test_model_like_prefill_for<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = model_like_prefill_data::<T>();
    test_internal(&input, &expected);
}

fn test_model_like_prefill_with_padded_kv_cache_for<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = model_like_prefill_data::<T>();
    assert_non_cpu_case!(
        &input,
        &expected,
        epsilon::<T>(),
        "AttentionGemm with padded KV cache",
        run_attention_with_padded_kv_cache,
        256,
    );
}

fn test_model_like_prefill_with_pooled_scratch_output_for<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = model_like_prefill_data::<T>();
    assert_non_cpu_case!(
        &input,
        &expected,
        epsilon::<T>(),
        "AttentionGemm with pooled scratch output",
        run_attention_with_pooled_scratch_output,
    );
}

fn test_model_like_prefill_followed_by_matmul_for<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_attention) = model_like_prefill_data::<T>();
    let input_dim = input.num_heads * input.head_dim;
    let output_dim = 256;
    let weights = followup_matmul_weights::<T>(input_dim, output_dim);
    let expected = expected_row_major_matmul(&expected_attention, &weights, input_dim, output_dim);

    assert_non_cpu_case!(
        &input,
        &expected,
        chained_matmul_epsilon::<T>(),
        "AttentionGemm followed by Matmul",
        run_attention_followed_by_matmul,
        output_dim,
    );
}

define_float_case!(test_basic, test_basic_for);
define_float_case!(test_causal, test_causal_for);
define_float_case!(test_gqa, test_gqa_for);
define_float_case!(test_head_dim_128, test_head_dim_128_for);
define_float_case!(test_unaligned, test_unaligned_for);
define_float_case!(test_model_like_prefill, test_model_like_prefill_for);
define_float_case!(test_model_like_prefill_with_padded_kv_cache, test_model_like_prefill_with_padded_kv_cache_for);
define_float_case!(
    test_model_like_prefill_with_pooled_scratch_output,
    test_model_like_prefill_with_pooled_scratch_output_for
);
define_float_case!(test_model_like_prefill_followed_by_matmul, test_model_like_prefill_followed_by_matmul_for);
