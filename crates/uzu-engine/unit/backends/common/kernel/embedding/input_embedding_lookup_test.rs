use std::fmt::Display;

use futures::executor::block_on;
use half::{bf16, f16};
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{
                d4s4,
                embedding::EmbeddingTableKind::{self, D4S4, Dense, Quantized},
                quantization::QuantizationMode::{self, I8, U2, U4, U8},
                quantization_method::QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
            },
            kernel::InputEmbeddingLookupKernel,
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend, for_each_non_cpu_backend,
            submit_command_buffer,
        },
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

fn floats<T: DataTypeRepr + Float>(
    len: u32,
    value: impl Fn(u32) -> f32,
) -> Vec<T> {
    (0..len).map(|index| T::from(value(index)).unwrap()).collect()
}

fn lookup<B: Backend, T: DataTypeRepr + Float>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    hadamard_size: u32,
) -> Vec<T> {
    let context = block_on(<B as Backend>::Context::new()).unwrap();
    let context = context.as_ref();
    let bytes = |data: &[u8]| create_buffer_with_data::<B, u8>(context, data);
    let (mut scales, mut zero_points, mut biases) = (None, None, None);
    let (mut ladder_indices, mut ladder, mut codebook) = (None, None, None);
    let values = match table_kind {
        Dense => {
            bytes(bytemuck::cast_slice(&floats::<T>(VOCAB_SIZE * MODEL_DIM, |index| (index as f32 * 0.37).sin() * 3.0)))
        },
        Quantized => {
            let (mode, method, group_size) = quantization.expect("quantization settings");
            let groups = MODEL_DIM.div_ceil(group_size);
            let group_values =
                |value: fn(u32) -> f32| create_buffer_with_data::<B, T>(context, &floats(VOCAB_SIZE * groups, value));
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
            scales = Some(create_buffer_with_data::<B, T>(context, &row_scales));
            ladder_indices = Some(bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::COLUMNS_PER_LADDER_INDEX_BYTE, 5)));
            ladder = Some(create_buffer_with_data::<B, f16>(context, &steps));
            codebook = Some(create_buffer_with_data::<B, i8>(context, &points));
            bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::VALUES_PER_CODE, 37))
        },
    };
    let hadamard_factors = if hadamard_size != 0 {
        let signs: Vec<i32> = (0..MODEL_DIM)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        Some(create_buffer_with_data::<B, i32>(context, &signs))
    } else {
        None
    };

    let (mode, method, group_size) = quantization
        .map_or((None, None, None), |(mode, method, group_size)| (Some(mode), Some(method), Some(group_size)));
    let kernel = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel::new(
        context,
        T::data_type(),
        table_kind,
        group_size,
        mode,
        method,
        hadamard_size,
    )
    .unwrap();
    let token_ids = create_buffer_with_data::<B, u32>(context, &TOKEN_IDS);
    let mut output = create_buffer::<B, T>(context, TOKEN_IDS.len() * MODEL_DIM as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
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
        0,
        MODEL_DIM,
        1.5,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, T>(context, &output)
}

fn check<T: DataTypeRepr + Float + Display>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    hadamard_size: u32,
) {
    let expected = lookup::<Cpu, T>(table_kind, quantization, hadamard_size);
    if let Some((U2, method, group_size)) = quantization {
        let groups = MODEL_DIM.div_ceil(group_size);
        let codes = pattern(VOCAB_SIZE * MODEL_DIM / 4, 37);
        let zero_points = pattern(VOCAB_SIZE * groups.div_ceil(4), 53);
        let reference = TOKEN_IDS
            .into_iter()
            .flat_map(|token| {
                let codes = &codes;
                let zero_points = &zero_points;
                (0..MODEL_DIM).map(move |column| {
                    if token >= VOCAB_SIZE {
                        return T::zero();
                    }
                    let group = column / group_size;
                    let parameter = token * groups + group;
                    let scale = T::from(0.02 + (parameter % 5) as f32 * 0.01).unwrap().to_f32().unwrap();
                    let index = token * MODEL_DIM + column;
                    let code = (codes[index as usize / 4] >> (2 * (index % 4))) & 3;
                    let bias = match method {
                        ScaleBias => T::from(-0.1 + parameter as f32 * 0.03).unwrap().to_f32().unwrap(),
                        ScaleZeroPoint => {
                            -scale
                                * ((zero_points[(token * groups.div_ceil(4) + group / 4) as usize]
                                    >> (2 * (group % 4)))
                                    & 3) as f32
                        },
                        _ => -scale * 2.0,
                    };
                    T::from((scale * code as f32 + bias) * 1.5).unwrap()
                })
            })
            .collect::<Vec<_>>();
        assert_eq_float(&reference, &expected, 1e-6, "independent U2 decoding");
    }
    let in_vocab = 2 * MODEL_DIM as usize;
    assert!(expected[..in_vocab].iter().any(|value| *value != T::zero()), "in-vocab rows are empty");
    assert!(expected[in_vocab..].iter().all(|value| *value == T::zero()), "out-of-vocab row is not zero");
    for_each_non_cpu_backend!(|B| {
        let actual = lookup::<B, T>(table_kind, quantization, hadamard_size);
        assert!(actual[in_vocab..].iter().all(|value| *value == T::zero()), "out-of-vocab row is not zero");
        let tolerance = if hadamard_size != 0 && table_kind == Quantized {
            1e-6
        } else {
            0.02
        };
        assert_eq_float(&expected, &actual, tolerance, std::any::type_name::<B>());
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, 0)]
#[case::dense_hadamard(Dense, None, 32)]
#[case::dense_hadamard128(Dense, None, 128)]
#[case::u2_bias(Quantized, Some((U2, ScaleBias, 128)), 0)]
#[case::u2_zero_point(Quantized, Some((U2, ScaleZeroPoint, 48)), 0)]
#[case::u2_symmetric(Quantized, Some((U2, ScaleSymmetric, 128)), 0)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), 0)]
#[case::u4_zero_point_odd_group(Quantized, Some((U4, ScaleZeroPoint, 48)), 0)]
#[case::u4_symmetric(Quantized, Some((U4, ScaleSymmetric, 32)), 0)]
#[case::u8_bias(Quantized, Some((U8, ScaleBias, 32)), 0)]
#[case::u8_zero_point(Quantized, Some((U8, ScaleZeroPoint, 32)), 0)]
#[case::u8_symmetric(Quantized, Some((U8, ScaleSymmetric, 32)), 0)]
#[case::i8_symmetric(Quantized, Some((I8, ScaleSymmetric, 32)), 0)]
#[case::i8_hadamard128(Quantized, Some((I8, ScaleSymmetric, 64)), 128)]
#[case::u4_bias_hadamard(Quantized, Some((U4, ScaleBias, 32)), 32)]
#[case::d4s4(D4S4, None, 32)]
fn input_embedding_lookup_bf16(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] hadamard_size: u32,
) {
    check::<bf16>(table_kind, quantization, hadamard_size);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, 0)]
#[case::dense_hadamard(Dense, None, 32)]
#[case::u2_bias(Quantized, Some((U2, ScaleBias, 128)), 0)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), 0)]
fn input_embedding_lookup_f32(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] hadamard_size: u32,
) {
    check::<f32>(table_kind, quantization, hadamard_size);
}

#[uzu_test]
fn input_embedding_lookup_first_row() {
    const PART_ROWS: [u32; 2] = [3, 5];
    const STACK_TOKEN_IDS: [u32; 6] = [3, 0, 7, 2, 8, 4];
    let u8_codes = pattern(PART_ROWS[0] * MODEL_DIM, 37);
    let u8_scales = floats::<bf16>(PART_ROWS[0] * MODEL_DIM / 64, |index| 0.02 + (index % 5) as f32 * 0.01);
    let u4_codes = pattern(PART_ROWS[1] * MODEL_DIM / 2, 41);
    let u4_scales = floats::<bf16>(PART_ROWS[1] * MODEL_DIM / 32, |index| 0.03 + (index % 7) as f32 * 0.01);
    let u4_zero_points = pattern(PART_ROWS[1] * MODEL_DIM / 64, 53);
    let nibble = |bytes: &[u8], index: u32| (bytes[index as usize / 2] >> (4 * (index % 2))) & 15;
    let mut expected = Vec::new();
    for token in STACK_TOKEN_IDS {
        for column in 0..MODEL_DIM {
            let weight = if token < PART_ROWS[0] {
                let scale = u8_scales[(token * (MODEL_DIM / 64) + column / 64) as usize];
                (u8_codes[(token * MODEL_DIM + column) as usize] as f32 - 128.0) * scale.to_f32()
            } else if token < PART_ROWS[0] + PART_ROWS[1] {
                let row = token - PART_ROWS[0];
                let group = row * (MODEL_DIM / 32) + column / 32;
                let code = nibble(&u4_codes, row * MODEL_DIM + column);
                (code as f32 - nibble(&u4_zero_points, group) as f32) * u4_scales[group as usize].to_f32()
            } else {
                0.0
            };
            expected.push(bf16::from_f32(weight * 1.5));
        }
    }

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).unwrap();
        let context = context.as_ref();
        let kernel = |(mode, method, group_size): Quantization| {
            <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel::new(
                context,
                bf16::data_type(),
                Quantized,
                Some(group_size),
                Some(mode),
                Some(method),
                0,
            )
            .unwrap()
        };
        let bytes = |data: &[u8]| create_buffer_with_data::<B, u8>(context, data);
        let parts = [
            (
                kernel((U8, ScaleSymmetric, 64)),
                bytes(&u8_codes),
                create_buffer_with_data::<B, bf16>(context, &u8_scales),
                None,
            ),
            (
                kernel((U4, ScaleZeroPoint, 32)),
                bytes(&u4_codes),
                create_buffer_with_data::<B, bf16>(context, &u4_scales),
                Some(bytes(&u4_zero_points)),
            ),
        ];
        let token_ids = create_buffer_with_data::<B, u32>(context, &STACK_TOKEN_IDS);
        let mut output = create_buffer_with_data::<B, bf16>(context, &vec![bf16::ONE; expected.len()]);
        let unused = None::<&<B as Backend>::GlobalGpuBuffer>;
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        let mut first_row = 0;
        for ((kernel, values, scales, zero_points), rows) in parts.iter().zip(PART_ROWS) {
            kernel.encode(
                &token_ids,
                values,
                Some(scales),
                zero_points.as_ref(),
                unused,
                unused,
                unused,
                unused,
                unused,
                &mut output,
                STACK_TOKEN_IDS.len() as u32,
                rows,
                first_row,
                MODEL_DIM,
                1.5,
                &mut command_buffer,
            );
            first_row += rows;
        }
        submit_command_buffer(command_buffer);
        assert_eq!(buffer_to_vec::<B, bf16>(context, &output), expected, "{}", std::any::type_name::<B>());
    });
}
