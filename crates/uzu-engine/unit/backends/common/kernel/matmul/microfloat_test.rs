use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            microfloat::{MicrofloatAxisOrder, MicrofloatEncoding, MicrofloatFormat, MicrofloatMetadata, decode_mxfp4},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

const K: usize = 32;
const N: usize = 4;

fn packed_codes() -> Vec<u8> {
    (0..N * K / 2)
        .map(|index| {
            let low = (index % 7 + 1) as u8;
            let high = ((index * 3 + 1) % 7 + 1) as u8;
            low | (high << 4)
        })
        .collect()
}

#[uzu_test]
fn cpu_executes_dense_mxfp4_matmul() {
    for row_count in [1, 5] {
        for group_size in [16, 32] {
            let input_values: Vec<f32> = (0..row_count * K).map(|index| (index % 13) as f32 * 0.125 - 0.5).collect();
            let codes = packed_codes();
            let scales: Vec<u8> = (0..N * K / group_size).map(|index| 126 + (index % 3) as u8).collect();
            let outer_scales = [1.25f32];
            let encoding = MicrofloatEncoding::new(
                MicrofloatFormat::Mxfp4,
                4,
                group_size as u32,
                MicrofloatAxisOrder::OutputInput,
            )
            .expect("valid MXFP4 encoding");
            let metadata = MicrofloatMetadata::new(encoding, N as u32, K as u32).expect("valid dense MXFP4 metadata");

            let context = <Cpu as Backend>::Context::new().expect("create CPU context");
            let input = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &input_values);
            let codes = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &codes);
            let scales = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &scales);
            let outer_scales = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &outer_scales);
            let mut output = alloc_allocation::<Cpu, f32>(context.as_ref(), row_count * N);
            let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
                context.as_ref(),
                DataType::F32,
                DataType::F32,
                DataType::F32,
            )
            .expect("create CPU matmul kernel");
            let mut encoder = Encoder::<Cpu>::new(context.as_ref()).expect("create CPU encoder");
            kernel
                .encode(
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
                        m: row_count as u32,
                        n: N as u32,
                        k: K as u32,
                    },
                    &mut encoder,
                )
                .expect("encode dense MXFP4 matmul");
            encoder.end_encoding().submit().wait_until_completed().expect("execute dense MXFP4 matmul");
            let actual = allocation_to_vec::<Cpu, f32>(&output);

            let codes: &[u8] = codes.as_slice();
            let scales: &[u8] = scales.as_slice();
            let mut expected = vec![0.0f32; row_count * N];
            for row in 0..row_count {
                for output_row in 0..N {
                    for inner in 0..K {
                        let packed = codes[output_row * K / 2 + inner / 2];
                        let code = if inner.is_multiple_of(2) {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let scale = scales[output_row * K / group_size + inner / group_size];
                        expected[row * N + output_row] +=
                            input_values[row * K + inner] * decode_mxfp4(code, scale, 1.25);
                    }
                }
            }
            assert_eq_float(&expected, &actual, 1e-5, "CPU dense MXFP4");
        }
    }
}
