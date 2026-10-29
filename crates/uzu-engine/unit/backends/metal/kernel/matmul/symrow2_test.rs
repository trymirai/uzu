use futures::executor::block_on;
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{
                Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
            },
        },
        metal::{Metal, context::MetalContext},
    },
    data_type::DataTypeRepr,
    tests::{
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        matmul::{QuantInput, quant::PreparedInt8A},
    },
};

type Gpu = <Metal as Backend>::GlobalGpuBuffer;

/// Packs true codes `[n][k]` into the `[group][tile16][lane][uint4]` layout the kernels read.
fn pack_symrow2(
    codes: &[u8],
    n: usize,
    k: usize,
) -> Vec<u32> {
    let (tiles, groups) = (n / 16, k / 128);
    let mut out = vec![0u32; tiles * groups * 128];
    for t in 0..tiles {
        for g in 0..groups {
            for lane in 0..32usize {
                let (row, q) = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1), (lane & 1) + 2 * ((lane >> 3) & 1));
                for u in 0..4usize {
                    let mut word = 0u32;
                    for s in 0..4usize {
                        for i in 0..4usize {
                            // Codes within each 16-column block are stored in activation order.
                            let (block, p) = (g * 8 + u * 2 + (s >> 1), 4 * i + q);
                            let source = (t * 16 + row + 8 * (s & 1)) * k + 16 * block + 4 * (p & 3) + (p >> 2);
                            word |= u32::from(codes[source]) << (8 * i + 2 * s);
                        }
                    }
                    out[((g * tiles + t) * 32 + lane) * 4 + u] = word;
                }
            }
        }
    }
    out
}

struct SymFixture {
    m: u32,
    n: u32,
    k: u32,
    codes: Vec<u8>,
    packed: Vec<u32>,
    row_scales: Vec<bf16>,
    prepared: PreparedInt8A,
}

impl SymFixture {
    fn new(
        m: u32,
        n: u32,
        k: u32,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let codes: Vec<u8> = (0..n as usize * k as usize).map(|_| rng.random_range(0u8..4)).collect();
        let row_scales = (0..n).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.1))).collect();
        let (_, reference) = QuantInput::<bf16>::new(m, k, n, 64, 4, QuantizationMethod::ScaleZeroPoint, seed)
            .with_prepared_a_and_reference(ACTIVATION_SCALE_GROUP_SIZE, Some(128));
        Self {
            m,
            n,
            k,
            packed: pack_symrow2(&codes, n as usize, k as usize),
            codes,
            row_scales,
            prepared: reference.prepared_a.unwrap(),
        }
    }

    /// `(value, sum of |term|)` per output, in f64.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let (m, n, k) = (self.m as usize, self.n as usize, self.k as usize);
        let (mut out, mut l1) = (vec![0.0; m * n], vec![0.0; m * n]);
        for row in 0..m {
            for col in 0..n {
                let (mut total, mut magnitude) = (0.0f64, 0.0f64);
                for g in 0..k / 128 {
                    let dot: f64 = (g * 128..(g + 1) * 128)
                        .map(|i| {
                            f64::from(self.prepared.values[row * k + i]) * (f64::from(self.codes[col * k + i]) - 1.5)
                        })
                        .sum();
                    let term = f64::from(self.prepared.scales[row * k / 128 + g]) * dot;
                    total += term;
                    magnitude += term.abs();
                }
                let scale = f64::from(self.row_scales[col].to_f32());
                out[row * n + col] = total * scale;
                l1[row * n + col] = magnitude * scale;
            }
        }
        (out, l1)
    }
}

fn matmul_kernel(context: &MetalContext) -> <<Metal as Backend>::Kernels as Kernels>::MatmulKernel {
    <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .unwrap()
}

fn run_dense(
    context: &MetalContext,
    fixture: &SymFixture,
    rht_factors: Option<&[i32]>,
    bias: Option<&[bf16]>,
) -> Vec<f64> {
    let codes = create_buffer_with_data::<Metal, u32>(context, &fixture.packed);
    let row_scales = create_buffer_with_data::<Metal, bf16>(context, &fixture.row_scales);
    let values = create_buffer_with_data::<Metal, i8>(context, &fixture.prepared.values);
    let scales = create_buffer_with_data::<Metal, f32>(context, &fixture.prepared.scales);
    let sums = create_buffer_with_data::<Metal, i32>(context, &fixture.prepared.group_sums);
    let rht_factors = rht_factors.map(|factors| create_buffer_with_data::<Metal, i32>(context, factors));
    let bias = bias.map(|bias| create_buffer_with_data::<Metal, bf16>(context, bias));
    let mut output = create_buffer::<Metal, bf16>(context, (fixture.m * fixture.n) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    matmul_kernel(context)
        .encode(
            MatmulArguments {
                a: MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    group_sums: Some(&sums),
                    scale_group_size: ACTIVATION_SCALE_GROUP_SIZE,
                    code_layout: Int8CodeLayout::Sequential,
                },
                b: MatmulB::SymRow2 {
                    codes: &codes,
                    row_scales: &row_scales,
                },
                b_leading_dimension: None,
                b_transpose: true,
                output: MatmulOutput::new(
                    &mut output,
                    MatmulDOps {
                        rht_factors: rht_factors.as_ref(),
                        bias: bias.as_ref(),
                        ..MatmulDOps::none()
                    },
                ),
                gather_indices: None::<&Gpu>,
                m: fixture.m,
                n: fixture.n,
                k: fixture.k,
            },
            &mut command_buffer,
        )
        .unwrap();
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<Metal, bf16>(context, &output).into_iter().map(|v| f64::from(v.to_f32())).collect()
}

/// Error in units of the allowed one: half a bf16 spacing of the result plus 2^-18 of the summed term magnitudes.
fn worst_error(
    actual: &[f64],
    expected: &[f64],
    l1: &[f64],
) -> f64 {
    (0..actual.len())
        .map(|i| {
            (actual[i] - expected[i]).abs()
                / (expected[i].abs() * 2f64.powi(-8) * 1.02 + l1[i] * 2f64.powi(-18) + 1e-30)
        })
        .fold(0.0, f64::max)
}

#[uzu_test]
fn symrow2_dense_matches_f64_reference() {
    let context = crate::tests::util::shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    // Decode and verify row counts (with a row tail), 2- and 4-way K splits, long K.
    for (m, n, k) in [(1, 2048, 12288), (5, 6144, 2048), (16, 4096, 4096)] {
        let fixture = SymFixture::new(m, n, k, 7 + u64::from(m));
        let (reference, l1) = fixture.reference();
        let error = worst_error(&run_dense(&context, &fixture, None, None), &reference, &l1);
        assert!(error < 1.0, "m={m} n={n} k={k}: {error}");
    }
}

#[uzu_test]
fn symrow2_fused_rht_matches_plain_then_hadamard() {
    let context = crate::tests::util::shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    for (m, n, k) in [(1, 2048, 4096), (16, 6144, 4096)] {
        let fixture = SymFixture::new(m, n, k, 11 + u64::from(m));
        let mut rng = SmallRng::seed_from_u64(99);
        let factors: Vec<i32> = (0..n)
            .map(|_| {
                if rng.random() {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let bias: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0))).collect();
        let plain = run_dense(&context, &fixture, None, None);
        let fused = run_dense(&context, &fixture, Some(&factors), Some(&bias));
        let mut worst = 0.0f64;
        for row in 0..m as usize {
            for block in 0..n as usize / 32 {
                let base = row * n as usize + block * 32;
                let scale = plain[base..base + 32].iter().fold(1e-30f64, |max, v| max.max(v.abs())) + 1.0;
                for i in 0..32usize {
                    let transformed: f64 = (0..32)
                        .map(|j| {
                            if (i & j).count_ones() % 2 == 0 {
                                plain[base + j]
                            } else {
                                -plain[base + j]
                            }
                        })
                        .sum();
                    let expected = transformed / 32f64.sqrt() * f64::from(factors[block * 32 + i])
                        + f64::from(bias[block * 32 + i].to_f32());
                    worst = worst.max((fused[base + i] - expected).abs() / scale);
                }
            }
        }
        assert!(worst < 2f64.powi(-5), "m={m} n={n} k={k}: {worst}");
    }
}

#[uzu_test]
fn symrow2_gather_matches_f64_reference_with_tail_candidates() {
    let context = crate::tests::util::shared_metal_context();
    // 517 candidates drawn from 4096 physical rows: not a multiple of the 4-candidate tile.
    let (n, k, candidates) = (4096u32, 4096u32, 517u32);
    let fixture = SymFixture::new(1, n, k, 11);
    let mut rng = SmallRng::seed_from_u64(11);
    let codes = create_buffer_with_data::<Metal, u32>(&context, &fixture.packed);
    let row_scales = create_buffer_with_data::<Metal, bf16>(&context, &fixture.row_scales);
    for m in [1u32, 4] {
        let x: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0))).collect();
        let indices: Vec<u32> = (0..m * candidates).map(|_| rng.random_range(0..n)).collect();
        let x_buffer = create_buffer_with_data::<Metal, bf16>(&context, &x);
        let index_buffer = create_buffer_with_data::<Metal, u32>(&context, &indices);
        let mut output = create_buffer::<Metal, bf16>(&context, (m * candidates) as usize);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        matmul_kernel(&context)
            .encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &x_buffer,
                        offset: 0,
                    },
                    b: MatmulB::SymRow2 {
                        codes: &codes,
                        row_scales: &row_scales,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(&mut output, MatmulDOps::none()),
                    gather_indices: Some(&index_buffer),
                    m,
                    n: candidates,
                    k,
                },
                &mut command_buffer,
            )
            .unwrap();
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        let actual: Vec<f64> =
            buffer_to_vec::<Metal, bf16>(&context, &output).into_iter().map(|v| f64::from(v.to_f32())).collect();
        let (mut expected, mut l1) = (vec![0.0; actual.len()], vec![0.0; actual.len()]);
        for (i, slot) in indices.iter().enumerate() {
            let (row, index) = (i / candidates as usize, *slot as usize);
            let scale = f64::from(fixture.row_scales[index].to_f32());
            for kk in 0..k as usize {
                let term = f64::from(x[row * k as usize + kk].to_f32())
                    * (f64::from(fixture.codes[index * k as usize + kk]) - 1.5);
                expected[i] += term * scale;
                l1[i] += term.abs() * scale;
            }
        }
        let error = worst_error(&actual, &expected, &l1);
        assert!(error < 1.0, "m={m}: {error}");
    }
}
