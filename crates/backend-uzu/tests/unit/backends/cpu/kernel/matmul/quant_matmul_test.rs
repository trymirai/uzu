use std::{hint::black_box, time::Instant};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{QuantizationMethod, QuantizationMode},
    tests::matmul::quant::{QuantInput, run_quant_cpu},
};

fn reference_quant_matmul<T: ArrayElement + Float>(input: &QuantInput<T>) -> Vec<T> {
    let (rows, inner_size, columns) = (input.m as usize, input.k as usize, input.n as usize);
    let group_size = input.group_size as usize;
    let bits = match input.mode {
        QuantizationMode::U4 => 4usize,
        _ => 8usize,
    };
    let groups = inner_size.div_ceil(group_size);
    let zero_point_stride = if bits == 4 {
        groups.div_ceil(2)
    } else {
        groups
    };
    let pack_factor = 32 / bits;
    let code_mask = (1u32 << bits) - 1;
    let midpoint = (1u32 << (bits - 1)) as f32;
    let words = input.weights_for_upload();
    let prepared = input.prepared_a.as_ref();

    let mut output = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let mut accumulator = 0.0f32;
            for index in 0..inner_size {
                let linear = column * inner_size + index;
                let word = words[linear / pack_factor];
                let mut code = ((word >> ((linear % pack_factor) * bits)) & code_mask) as u8;
                if input.signed_codes {
                    code ^= 1u8 << (bits - 1);
                }

                let group = index / group_size;
                let scale = input.scales[column * groups + group].to_f32().unwrap();
                let bias = if let Some(zero_points) = &input.zero_points {
                    let zero_point = if bits == 4 {
                        let byte = zero_points[column * zero_point_stride + (group >> 1)];
                        if group.is_multiple_of(2) {
                            (byte & 0x0F) as f32
                        } else {
                            ((byte >> 4) & 0x0F) as f32
                        }
                    } else {
                        zero_points[column * zero_point_stride + group] as f32
                    };
                    -scale * zero_point
                } else if let Some(biases) = &input.biases {
                    biases[column * groups + group].to_f32().unwrap()
                } else {
                    -scale * midpoint
                };

                let activation = match prepared {
                    Some(prepared) => {
                        let activation_groups = inner_size / prepared.activation_scale_group_size as usize;
                        let activation_group = index / prepared.activation_scale_group_size as usize;
                        f32::from(prepared.values[row * inner_size + index])
                            * prepared.scales[row * activation_groups + activation_group]
                    },
                    None => input.x[row * inner_size + index].to_f32().unwrap(),
                };

                accumulator += activation * (scale * f32::from(code) + bias);
            }
            output.push(T::from(accumulator).unwrap());
        }
    }
    output
}

const METHODS: [QuantizationMethod; 3] =
    [QuantizationMethod::ScaleBias, QuantizationMethod::ScaleZeroPoint, QuantizationMethod::ScaleSymmetric];

const SHAPES: [(u32, u32, u32); 6] = [
    (1, 128, 64),  // decode, every group full
    (1, 40, 17),   // k = one group of 32 plus a group of 8
    (3, 96, 33),   // k below a 128-group, exact for 32 and short for 64
    (8, 256, 128), // prefill-ish batch
    (2, 160, 9),   // k = 5 groups of 32, odd column count
    (1, 72, 7),    // short trailing group for every group size
];

fn check_all<T: ArrayElement + Float + std::fmt::Debug>(label: &str) {
    let mut seed = 0x9E3779B97F4A7C15u64;
    for method in METHODS {
        for bits in [4u32, 8] {
            for group_size in [32u32, 64, 128] {
                for (m, k, n) in SHAPES {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let input = QuantInput::<T>::new(m, k, n, group_size, bits, method, seed);

                    assert_eq!(
                        run_quant_cpu(&input),
                        reference_quant_matmul(&input),
                        "{label}: method={method:?} bits={bits} group={group_size} shape={m}x{n}x{k}"
                    );
                }
            }
        }
    }
}

#[uzu_test]
fn quant_matmul_matches_reference_f32() {
    check_all::<f32>("f32");
}

#[uzu_test]
fn quant_matmul_matches_reference_f16() {
    check_all::<f16>("f16");
}

#[uzu_test]
fn quant_matmul_matches_reference_bf16() {
    check_all::<bf16>("bf16");
}

#[uzu_test]
fn quant_matmul_matches_reference_with_signed_codes() {
    for bits in [4u32, 8] {
        for method in METHODS {
            for (m, k, n) in SHAPES {
                let input =
                    QuantInput::<f32>::new(m, k, n, 32, bits, method, 0xD1B54A32D192ED03).with_signed_weight_codes();

                assert_eq!(
                    run_quant_cpu(&input),
                    reference_quant_matmul(&input),
                    "bits={bits} method={method:?} shape={m}x{n}x{k}"
                );
            }
        }
    }
}

#[uzu_test]
fn quant_matmul_matches_reference_with_int8_activations() {
    for bits in [4u32, 8] {
        for group_size in [32u32, 64] {
            let input = QuantInput::<f32>::new(
                2,
                256,
                48,
                group_size,
                bits,
                QuantizationMethod::ScaleSymmetric,
                0x14057B7EF767814F,
            )
            .with_prepared_a(group_size, None);

            assert_eq!(
                run_quant_cpu(&input),
                reference_quant_matmul(&input),
                "int8 activations: bits={bits} group={group_size}"
            );
        }
    }
}

#[uzu_test]
fn quant_matmul_decode_speed() {
    let input = QuantInput::<f32>::new(1, 1024, 8192, 64, 4, QuantizationMethod::ScaleZeroPoint, 0x2545F4914F6CDD1D);

    black_box(run_quant_cpu(&input));

    let iterations = 5;
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(run_quant_cpu(black_box(&input)));
    }
    let elapsed = started.elapsed() / iterations;

    println!("cpu quant matmul 1x8192x1024 (4-bit, group 64): {elapsed:?} per call");
}
