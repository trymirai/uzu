use std::any::TypeId;

use half::{bf16, f16};
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulError, MatmulKernel},
            microfloat::{MicrofloatEncoding, MicrofloatError, MicrofloatFormat, MicrofloatMetadata},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation_with_data, allocation_to_vec, create_context, for_each_non_cpu_backend, submit_encoder,
        },
    },
};

fn check_dense_mxfp4<B: Backend, T: ArrayElement>(
    row_count: usize,
    group_size: usize,
    wide_scale: bool,
    tolerance: f32,
) {
    const K: usize = 32;
    const N: usize = 4;
    const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

    // CPU covers f32 intermediate overflow; Metal retains its f16-range regression.
    let (wide_codes, wide_exponent, wide_outer_scale, wide_expected) = if TypeId::of::<B>() == TypeId::of::<Cpu>() {
        (0x77, 254, 0.25, 3.0 * 2.0f32.powi(116))
    } else {
        (0x22, 147, 1.0 / 1024.0, 1.0)
    };
    let encoding =
        MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 4, group_size as u32).expect("valid MXFP4 encoding");
    let metadata = MicrofloatMetadata::new(encoding, N as u32, K as u32).expect("valid dense MXFP4 metadata");
    let context = create_context::<B>();
    let input: Vec<f32> = (0..row_count * K)
        .map(|index| {
            if wide_scale {
                if index % K == 0 {
                    1.0 / 1024.0
                } else {
                    0.0
                }
            } else {
                (index % 13) as f32 * 0.125 - 0.5
            }
        })
        .collect();
    let codes: Vec<u8> = (0..N * K / 2)
        .map(|index| {
            if wide_scale {
                return wide_codes;
            }
            let low = ((index * 5 + 1) % 16) as u8;
            let high = ((index * 7 + 3) % 16) as u8;
            low | (high << 4)
        })
        .collect();
    let scales: Vec<u8> = (0..N * K / group_size)
        .map(|index| {
            if wide_scale {
                wide_exponent
            } else {
                126 + (index % 3) as u8
            }
        })
        .collect();
    let outer_scale: f32 = if wide_scale {
        wide_outer_scale
    } else {
        1.25
    };
    let mut expected = vec![0.0; row_count * N];
    for row in 0..row_count {
        for output_row in 0..N {
            for inner in 0..K {
                let packed = codes[output_row * K / 2 + inner / 2];
                let code = (packed >> (4 * (inner % 2))) & 0x0f;
                let exponent = scales[output_row * K / group_size + inner / group_size];
                let weight = (f64::from(E2M1[code as usize])
                    * 2.0f64.powi(i32::from(exponent) - 127)
                    * f64::from(outer_scale)) as f32;
                expected[row * N + output_row] += input[row * K + inner] * weight;
            }
        }
    }
    let biases = [0.5, -1.0, 1.5, -2.0];
    if wide_scale {
        assert_eq!(expected, vec![wide_expected; row_count * N]);
    } else {
        for (index, value) in expected.iter_mut().enumerate() {
            *value = ((*value * 0.5 + biases[index % N]) / 16.0).tanh() * 16.0;
        }
    }

    let input: Vec<T> = input.into_iter().map(|value| T::from(value).expect("representable input")).collect();
    let outer_scale = T::from(outer_scale).expect("representable outer scale");
    let input = alloc_allocation_with_data::<B, T>(context.as_ref(), &input);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let outer_scales = alloc_allocation_with_data::<B, T>(context.as_ref(), &[outer_scale]);
    let biases = biases.map(|value| T::from(value).expect("representable bias"));
    let biases = alloc_allocation_with_data::<B, T>(context.as_ref(), &biases);
    let mut output = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![f32::NAN; row_count * N]);
    let d_transform = if wide_scale {
        MatmulDOps::none()
    } else {
        MatmulDOps {
            ab_scale: 0.5,
            bias: Some(&biases),
            soft_cap: Some(16.0),
            ..MatmulDOps::none()
        }
    };
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), T::data_type(), T::data_type(), DataType::F32)
            .expect("create matmul kernel");
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::<B>::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform,
                gather_indices: None,
                m: row_count as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .expect("encode MXFP4 matmul");
    submit_encoder(encoder);
    let actual = allocation_to_vec::<B, f32>(&output);
    assert_eq_float(
        &expected,
        &actual,
        tolerance,
        &format!("{} {:?} M={row_count} group={group_size}", std::any::type_name::<B>(), T::data_type()),
    );
}

#[uzu_test]
fn dense_mxfp4_matches_scalar_reference() {
    for row_count in [1, 9, 33] {
        for group_size in [16, 32] {
            check_dense_mxfp4::<Cpu, f16>(row_count, group_size, false, 1e-5);
            check_dense_mxfp4::<Cpu, bf16>(row_count, group_size, false, 1e-5);
            check_dense_mxfp4::<Cpu, f32>(row_count, group_size, false, 1e-5);
            check_dense_mxfp4::<Cpu, f16>(row_count, group_size, true, 1e-5);
            for_each_non_cpu_backend!(|B| {
                check_dense_mxfp4::<B, f16>(row_count, group_size, false, 0.01);
                check_dense_mxfp4::<B, bf16>(row_count, group_size, false, 0.01);
                check_dense_mxfp4::<B, f32>(row_count, group_size, false, 0.01);
                check_dense_mxfp4::<B, f16>(row_count, group_size, true, 0.01);
            });
        }
    }
}

#[uzu_test]
fn cpu_rejects_invalid_mxfp4_operands_before_dispatch() {
    let context = create_context::<Cpu>();
    let encoding = MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 4, 16).expect("valid MXFP4 encoding");
    let metadata = MicrofloatMetadata::new(encoding, 4, 32).expect("valid MXFP4 metadata");
    let input = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &[1.0; 32]);
    let mut output = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &[f32::NAN; 4]);
    let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context.as_ref(),
        DataType::F32,
        DataType::F32,
        DataType::F32,
    )
    .expect("create CPU matmul kernel");

    for (metadata, code_bytes, scale_bytes, outer_bytes, invalid_metadata) in [
        (
            MicrofloatMetadata {
                rows: 0,
                ..metadata
            },
            64,
            8,
            4,
            Some(MicrofloatError::EmptyShape),
        ),
        (
            MicrofloatMetadata {
                columns: 33,
                ..metadata
            },
            64,
            8,
            4,
            Some(MicrofloatError::MisalignedColumns {
                columns: 33,
                group_size: 16,
            }),
        ),
        (
            MicrofloatMetadata {
                encoding: MicrofloatEncoding {
                    group_size: 0,
                    ..encoding
                },
                ..metadata
            },
            64,
            8,
            4,
            Some(MicrofloatError::UnsupportedGroupSize {
                format: MicrofloatFormat::Mxfp4,
                group_size: 0,
            }),
        ),
        (metadata, 63, 8, 4, None),
        (metadata, 64, 7, 4, None),
        (metadata, 64, 8, 3, None),
    ] {
        let codes = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &vec![0x22; code_bytes]);
        let scales = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &vec![127; scale_bytes]);
        let outer_scales = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &vec![0; outer_bytes]);
        let mut encoder = Encoder::<Cpu>::new(context.as_ref()).expect("create CPU encoder");
        let result = kernel.encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::<Cpu>::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                m: 1,
                n: 4,
                k: 32,
            },
            &mut encoder,
        );
        let error = result.expect_err("reject invalid MXFP4 operand during encoding");
        let error = (&error as &dyn std::error::Error)
            .source()
            .and_then(|source| source.downcast_ref::<MatmulError<Cpu>>())
            .expect("preserve matmul error");
        match (invalid_metadata, error) {
            (Some(expected), MatmulError::InvalidMicrofloat(actual)) => assert_eq!(*actual, expected),
            (None, MatmulError::InvalidMicrofloatStorage) => {},
            (expected, actual) => panic!("expected metadata error {expected:?}, got {actual:?}"),
        }
    }
}
