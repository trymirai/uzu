use std::fmt::Display;

use half::{bf16, f16};
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::{
                EmbeddingTableKind::{self, D4S4, Dense, Quantized},
                QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
                QuantizationMode::{self, I8, U4, U8},
                d4s4,
            },
            kernel::InputEmbeddingLookupKernel,
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const VOCAB_SIZE: u32 = 4;
const MODEL_DIM: u32 = 128;
const TOKEN_IDS: [u32; 3] = [0, 2, VOCAB_SIZE];

type Quantization = (QuantizationMode, QuantizationMethod, u32);

fn pattern(
    len: u32,
    seed: u32,
) -> Vec<u8> {
    (0..len).map(|index| (index * seed + 9) as u8).collect()
}

fn floats<T: ArrayElement + Float>(
    len: u32,
    value: impl Fn(u32) -> f32,
) -> Vec<T> {
    (0..len).map(|index| T::from(value(index)).unwrap()).collect()
}

fn lookup<B: Backend, T: ArrayElement + Float>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    use_hadamard: bool,
) -> Vec<T> {
    let context = <B as Backend>::Context::new().unwrap();
    let context = context.as_ref();
    let bytes = |data: &[u8]| alloc_allocation_with_data::<B, u8>(context, data);
    let (mode, method, group_size) = quantization.unwrap_or((U4, ScaleSymmetric, 128));

    let (mut scales, mut zero_points, mut biases) = (None, None, None);
    let (mut ladder_indices, mut ladder, mut codebook) = (None, None, None);
    let values = match table_kind {
        Dense => {
            bytes(bytemuck::cast_slice(&floats::<T>(VOCAB_SIZE * MODEL_DIM, |index| (index as f32 * 0.37).sin() * 3.0)))
        },
        Quantized => {
            let groups = MODEL_DIM.div_ceil(group_size);
            let group_values = |value: fn(u32) -> f32| {
                alloc_allocation_with_data::<B, T>(context, &floats(VOCAB_SIZE * groups, value))
            };
            scales = Some(group_values(|index| 0.02 + (index % 5) as f32 * 0.01));
            if method == ScaleBias {
                biases = Some(group_values(|index| -0.1 + index as f32 * 0.03));
            } else if method == ScaleZeroPoint {
                zero_points = Some(bytes(&pattern(VOCAB_SIZE * groups.div_ceil(mode.packing_divisor()), 53)));
            }
            bytes(&pattern(VOCAB_SIZE * MODEL_DIM / mode.packing_divisor(), 37))
        },
        D4S4 => {
            let row_scales = floats::<T>(VOCAB_SIZE, |row| 0.02 + row as f32 * 0.01);
            let steps: Vec<f16> =
                (0..d4s4::LADDER_SIZE).map(|index| f16::from_f32(2f32.powf(index as f32 / 2.0 - 5.5))).collect();
            let points: Vec<i8> =
                (0..d4s4::CODEBOOK_SIZE * d4s4::VALUES_PER_CODE).map(|index| (index % 9) as i8 - 4).collect();
            scales = Some(alloc_allocation_with_data::<B, T>(context, &row_scales));
            ladder_indices = Some(bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::COLUMNS_PER_LADDER_INDEX_BYTE, 5)));
            ladder = Some(alloc_allocation_with_data::<B, f16>(context, &steps));
            codebook = Some(alloc_allocation_with_data::<B, i8>(context, &points));
            bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::VALUES_PER_CODE, 37))
        },
    };
    let hadamard_factors = if use_hadamard {
        let signs: Vec<i32> = (0..MODEL_DIM)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        Some(alloc_allocation_with_data::<B, i32>(context, &signs))
    } else {
        None
    };

    let kernel = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel::new(
        context,
        T::data_type(),
        table_kind,
        group_size,
        mode,
        method,
        use_hadamard,
    )
    .unwrap();
    let token_ids = alloc_allocation_with_data::<B, u32>(context, &TOKEN_IDS);
    let mut output = alloc_allocation::<B, T>(context, TOKEN_IDS.len() * MODEL_DIM as usize);
    let mut encoder = Encoder::<B>::new(context).unwrap();
    kernel.encode(
        &token_ids,
        &values,
        scales.as_ref(),
        zero_points.as_ref(),
        biases.as_ref(),
        hadamard_factors.as_ref(),
        ladder_indices.as_ref(),
        ladder.as_ref(),
        codebook.as_ref(),
        &mut output,
        TOKEN_IDS.len() as u32,
        VOCAB_SIZE,
        MODEL_DIM,
        1.5,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, T>(&output)
}

fn check<T: ArrayElement + Float + Display>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    use_hadamard: bool,
) {
    let expected = lookup::<Cpu, T>(table_kind, quantization, use_hadamard);
    let in_vocab = 2 * MODEL_DIM as usize;
    assert!(expected[..in_vocab].iter().any(|value| *value != T::zero()), "in-vocab rows are empty");
    assert!(expected[in_vocab..].iter().all(|value| *value == T::zero()), "out-of-vocab row is not zero");
    for_each_non_cpu_backend!(|B| {
        let actual = lookup::<B, T>(table_kind, quantization, use_hadamard);
        assert_eq_float(&expected, &actual, 0.02, std::any::type_name::<B>());
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, false)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), false)]
#[case::u4_zero_point_odd_group(Quantized, Some((U4, ScaleZeroPoint, 48)), false)]
#[case::u4_symmetric(Quantized, Some((U4, ScaleSymmetric, 32)), false)]
#[case::u8_bias(Quantized, Some((U8, ScaleBias, 32)), false)]
#[case::u8_zero_point(Quantized, Some((U8, ScaleZeroPoint, 32)), false)]
#[case::u8_symmetric(Quantized, Some((U8, ScaleSymmetric, 32)), false)]
#[case::i8_symmetric(Quantized, Some((I8, ScaleSymmetric, 32)), false)]
#[case::u4_bias_hadamard(Quantized, Some((U4, ScaleBias, 32)), true)]
#[case::d4s4(D4S4, None, true)]
fn input_embedding_lookup_bf16(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] use_hadamard: bool,
) {
    check::<bf16>(table_kind, quantization, use_hadamard);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, false)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), false)]
fn input_embedding_lookup_f32(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] use_hadamard: bool,
) {
    check::<f32>(table_kind, quantization, use_hadamard);
}
