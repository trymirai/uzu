use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::QKVNormKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::assert::assert_eq_float,
};

struct Input<InputT: ArrayElement + Float, ScaleT: ArrayElement + Float, OutputT: ArrayElement + Float> {
    qkv: Box<[OutputT]>,
    scales: Box<[ScaleT]>,
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    epsilon: f32,
    scale_offset: f32,
    head_offset: u32,
    head_count: u32,
    full_layer: bool,
    in_place: bool,
    scale_free: bool,
    _phantom: std::marker::PhantomData<InputT>,
}

fn get_test_data<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    head_offset: u32,
    head_count: u32,
    full_layer: bool,
    scale_free: bool,
) -> (Input<InputT, ScaleT, OutputT>, Vec<OutputT>) {
    let batch_size = 1u32;
    let num_q_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 8u32;
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;

    // QKV layout: [Q0, Q1, Q2, Q3, K0, K1, V0, V1] where each head has head_dim elements
    let qkv_width = ((num_q_heads + 2 * num_kv_heads) * head_dim) as usize;
    let total_size = (batch_size as usize) * qkv_width;

    let mut qkv_data_f32 = vec![0.0f32; total_size];

    // Q heads: values 1.0, 1.1, 1.2, 1.3, ...
    for head in 0..num_q_heads {
        for dim in 0..head_dim {
            let idx = (head * head_dim + dim) as usize;
            qkv_data_f32[idx] = 1.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // K heads: values 2.0, 2.1, 2.2, 2.3, ... (start after Q heads)
    let k_offset = (num_q_heads * head_dim) as usize;
    for head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let idx = k_offset + (head * head_dim + dim) as usize;
            qkv_data_f32[idx] = 2.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // V heads: values 3.0, 3.1, 3.2, 3.3, ... (start after K heads)
    let v_offset = ((num_q_heads + num_kv_heads) * head_dim) as usize;
    for head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let idx = v_offset + (head * head_dim + dim) as usize;
            qkv_data_f32[idx] = 3.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // Scale data (identity scaling)
    let scale_data: Vec<ScaleT> = (0..head_dim).map(|_| ScaleT::one()).collect();
    let qkv_data: Vec<OutputT> = qkv_data_f32.iter().map(|&x| OutputT::from(x).unwrap()).collect();

    let input = Input {
        qkv: qkv_data.into_boxed_slice(),
        scales: scale_data.into_boxed_slice(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        epsilon,
        scale_offset,
        head_offset,
        head_count,
        full_layer,
        in_place: true,
        scale_free,
        _phantom: std::marker::PhantomData,
    };

    let expected = get_output::<Cpu, InputT, ScaleT, OutputT, AccumT>(&input);
    (input, expected)
}

fn get_output<
    B: Backend,
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    input: &Input<InputT, ScaleT, OutputT>
) -> Vec<OutputT> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::QKVNormKernel::new(
        &context,
        InputT::data_type(),
        ScaleT::data_type(),
        OutputT::data_type(),
        AccumT::data_type(),
        input.in_place,
        input.scale_free,
    )
    .expect("Failed to create QKVNormKernel");

    let qkv_len = input.qkv.len();
    let mut qkv = context.create_array_from(&[qkv_len], &input.qkv).into_allocation();
    let scales_array = context.create_array_from(&[input.scales.len()], &input.scales);

    let scales = if input.scale_free {
        None
    } else {
        Some(scales_array.allocation())
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        None::<&Allocation<B>>,
        scales,
        &mut qkv,
        input.batch_size,
        input.num_q_heads,
        input.num_kv_heads,
        input.head_dim,
        input.epsilon,
        input.scale_offset,
        input.head_offset,
        input.head_count,
        input.full_layer,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    crate::tests::helpers::allocation_to_vec(&qkv)
}

fn test_internal<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>(
    input: &Input<InputT, ScaleT, OutputT>,
    expected: &[OutputT],
) {
    let eps = if matches!(InputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(ScaleT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(OutputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(AccumT::data_type(), DataType::F16 | DataType::BF16)
    {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, InputT, ScaleT, OutputT, AccumT>(input);
        let msg = format!(
            "QKVNorm kernel test failed with backend={}, head_offset={}, head_count={}, full_layer={}, scale_free={}",
            std::any::type_name::<B>(),
            input.head_offset,
            input.head_count,
            input.full_layer,
            input.scale_free,
        );
        assert_eq_float::<OutputT>(expected, &output, eps, &msg);
    });
}

fn test_q_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    for full_layer in [true, false] {
        // Normalize Q heads: head_offset=0, head_count=num_q_heads(4)
        let (input, expected) = get_test_data::<InputT, ScaleT, OutputT, AccumT>(0, 4, full_layer, false);
        test_internal::<InputT, ScaleT, OutputT, AccumT>(&input, &expected);
    }
}

fn test_k_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    for full_layer in [true, false] {
        // Normalize K heads: head_offset=num_q_heads(4), head_count=num_kv_heads(2)
        let (input, expected) = get_test_data::<InputT, ScaleT, OutputT, AccumT>(4, 2, full_layer, false);
        test_internal::<InputT, ScaleT, OutputT, AccumT>(&input, &expected);
    }
}

fn test_v_norm_scale_free<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    let (input, expected) = get_test_data::<InputT, ScaleT, OutputT, AccumT>(6, 2, true, true);
    test_internal::<InputT, ScaleT, OutputT, AccumT>(&input, &expected);
}

fn test_addressing<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    // Test that Q norm only modifies Q heads and leaves K/V untouched
    let (input, _expected) = get_test_data::<InputT, ScaleT, OutputT, AccumT>(0, 4, false, false);

    let num_q_heads = input.num_q_heads as usize;
    let head_dim = input.head_dim as usize;
    let k_start = num_q_heads * head_dim;
    let qkv_len = input.qkv.len();

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, InputT, ScaleT, OutputT, AccumT>(&input);

        // Q heads should be modified (different from input)
        for i in 0..(num_q_heads * head_dim) {
            let orig = input.qkv[i].to_f32().unwrap();
            let out = output[i].to_f32().unwrap();
            assert!(
                (out - orig).abs() > 0.001,
                "Q head element {} was not modified by Q norm on backend {}",
                i,
                std::any::type_name::<B>(),
            );
        }

        // K and V heads should be unchanged
        for i in k_start..qkv_len {
            let orig = input.qkv[i].to_f32().unwrap();
            let out = output[i].to_f32().unwrap();
            assert!(
                (out - orig).abs() < 1e-6,
                "K/V element {} was incorrectly modified by Q norm on backend {}: expected {}, got {}",
                i,
                std::any::type_name::<B>(),
                orig,
                out,
            );
        }
    });
}

fn test_v_addressing<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    // Scale-free V norm must modify only V heads and leave Q/K untouched.
    let (input, _expected) = get_test_data::<InputT, ScaleT, OutputT, AccumT>(6, 2, true, true);

    let num_q_heads = input.num_q_heads as usize;
    let num_kv_heads = input.num_kv_heads as usize;
    let head_dim = input.head_dim as usize;
    let v_start = (num_q_heads + num_kv_heads) * head_dim;
    let qkv_len = input.qkv.len();

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, InputT, ScaleT, OutputT, AccumT>(&input);

        for i in 0..v_start {
            let orig = input.qkv[i].to_f32().unwrap();
            let out = output[i].to_f32().unwrap();
            assert!(
                (out - orig).abs() < 1e-6,
                "Q/K element {} was incorrectly modified by V norm on backend {}: expected {}, got {}",
                i,
                std::any::type_name::<B>(),
                orig,
                out,
            );
        }

        for i in v_start..qkv_len {
            let orig = input.qkv[i].to_f32().unwrap();
            let out = output[i].to_f32().unwrap();
            assert!(
                (out - orig).abs() > 0.001,
                "V head element {} was not modified by V norm on backend {}",
                i,
                std::any::type_name::<B>(),
            );
        }
    });
}

// Q norm tests
#[uzu_test]
fn test_q_norm_f32_f32_f32_f32() {
    test_q_norm::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_q_norm_f16_f16_f16_f32() {
    test_q_norm::<f16, f16, f16, f32>();
}

#[uzu_test]
fn test_q_norm_f16_f16_f16_f16() {
    test_q_norm::<f16, f16, f16, f16>();
}

#[uzu_test]
fn test_q_norm_bf16_bf16_bf16_f32() {
    test_q_norm::<bf16, bf16, bf16, f32>();
}

#[uzu_test]
fn test_q_norm_f32_f16_f32_f32() {
    test_q_norm::<f32, f16, f32, f32>();
}

#[uzu_test]
fn test_q_norm_f16_f32_f16_f32() {
    test_q_norm::<f16, f32, f16, f32>();
}

// K norm tests
#[uzu_test]
fn test_k_norm_f32_f32_f32_f32() {
    test_k_norm::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_k_norm_f16_f16_f16_f32() {
    test_k_norm::<f16, f16, f16, f32>();
}

#[uzu_test]
fn test_k_norm_f16_f16_f16_f16() {
    test_k_norm::<f16, f16, f16, f16>();
}

#[uzu_test]
fn test_k_norm_bf16_bf16_bf16_f32() {
    test_k_norm::<bf16, bf16, bf16, f32>();
}

#[uzu_test]
fn test_v_norm_scale_free_f32_f32_f32_f32() {
    test_v_norm_scale_free::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_v_norm_scale_free_f16_f16_f16_f32() {
    test_v_norm_scale_free::<f16, f16, f16, f32>();
}

#[uzu_test]
fn test_v_norm_scale_free_bf16_bf16_bf16_f32() {
    test_v_norm_scale_free::<bf16, bf16, bf16, f32>();
}

// Addressing tests (Q norm should not touch K/V)
#[uzu_test]
fn test_addressing_f32_f32_f32_f32() {
    test_addressing::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_addressing_f16_f16_f16_f32() {
    test_addressing::<f16, f16, f16, f32>();
}

#[uzu_test]
fn test_v_addressing_f32_f32_f32_f32() {
    test_v_addressing::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_v_addressing_bf16_bf16_bf16_f32() {
    test_v_addressing::<bf16, bf16, bf16, f32>();
}
