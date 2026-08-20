#![cfg(backend = "metal")]

//! Production-shape oracle benches for the GPT-OSS MXFP4 expert decode path.
//! Reported throughput is sustained packed-weight traffic (bytes), not FLOP/s.

use std::num::NonZeroU32;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            kernel::matmul::{
                ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting,
            },
            microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata},
        },
        metal::{Metal, MetalContext},
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop,
        util::{shared_metal_context, type_short_name},
    },
};

type MetalKernels = <Metal as Backend>::Kernels;

struct Mxfp4Shape {
    label: &'static str,
    routes: u32,
    experts: u32,
    n: u32,
    k: u32,
    group_size: u32,
    input_is_bf16: bool,
}

fn mxfp4_expert_bytes(shape: &Mxfp4Shape) -> u64 {
    let per_expert = shape.n as u64 * (shape.k as u64 / 2 + shape.k as u64 / shape.group_size as u64);
    shape.routes as u64 * per_expert
}

fn bench_mxfp4_expert_projection(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    context: &MetalContext,
    shape: &Mxfp4Shape,
    expert_ids_value: &[i32],
    ids_label: &str,
) {
    let routes = shape.routes as usize;
    let experts = shape.experts as usize;
    let (n, k) = (shape.n as usize, shape.k as usize);
    let codes = alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * n * k / 2]);
    let scales =
        alloc_allocation_with_data::<Metal, u8>(context, &vec![127u8; experts * n * k / shape.group_size as usize]);
    let outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
    let biases = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * n]);
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        shape.group_size,
        MicrofloatLayout::OutputInput,
        shape.experts,
        shape.n,
        shape.k,
    )
    .unwrap();
    let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, expert_ids_value);
    let mut output = alloc_allocation::<Metal, u8>(context, routes * n * 4);
    let routes_per_token = NonZeroU32::new(shape.routes).unwrap();
    let expert_count = NonZeroU32::new(shape.experts).unwrap();

    let label = format!("{}_{}", shape.label, ids_label);
    group.throughput(Throughput::Bytes(mxfp4_expert_bytes(shape)));
    if shape.input_is_bf16 {
        let input = alloc_allocation::<Metal, bf16>(context, k);
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            crate::data_type::DataType::F32,
        )
        .expect("MatmulKernel");
        group.bench_function(BenchmarkId::new("mxfp4_routed", &label), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Tokens,
                            }),
                            m: shape.routes,
                            n: shape.n,
                            k: shape.k,
                        },
                        encoder,
                    )
                    .expect("mxfp4 routed matmul");
            });
        });
    } else {
        let input = alloc_allocation::<Metal, f32>(context, routes * k);
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            crate::data_type::DataType::F32,
            bf16::data_type(),
        )
        .expect("MatmulKernel");
        group.bench_function(BenchmarkId::new("mxfp4_routed", &label), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Routes,
                            }),
                            m: shape.routes,
                            n: shape.n,
                            k: shape.k,
                        },
                        encoder,
                    )
                    .expect("mxfp4 routed matmul");
            });
        });
    }
}

#[uzu_bench]
fn bench_mxfp4_expert_decode_production(c: &mut Criterion) {
    let context = &*shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/Mxfp4ExpertDecode", type_short_name::<Metal>()));

    const W13: Mxfp4Shape = Mxfp4Shape {
        label: "W13_N5760_K2880_G16",
        routes: 4,
        experts: 32,
        n: 5760,
        k: 2880,
        group_size: 16,
        input_is_bf16: true,
    };
    const W2: Mxfp4Shape = Mxfp4Shape {
        label: "W2_N2880_K2880_G32",
        routes: 4,
        experts: 32,
        n: 2880,
        k: 2880,
        group_size: 32,
        input_is_bf16: false,
    };

    let spread_ids: Vec<i32> = vec![0, 8, 16, 24];
    let fixed_ids: Vec<i32> = vec![3, 7, 11, 19];
    for shape in [&W13, &W2] {
        bench_mxfp4_expert_projection(&mut group, context, shape, &spread_ids, "spread");
        bench_mxfp4_expert_projection(&mut group, context, shape, &fixed_ids, "fixed");
    }

    // Fused W13: identical weight traffic, gate/up epilogue folded in, output
    // is the activated hidden half-width (routes x 2880 F32).
    {
        let routes = W13.routes as usize;
        let experts = W13.experts as usize;
        let (n, k) = (W13.n as usize, W13.k as usize);
        let hidden = n / 2;
        let codes = alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * n * k / 2]);
        let scales =
            alloc_allocation_with_data::<Metal, u8>(context, &vec![127u8; experts * n * k / W13.group_size as usize]);
        let outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
        let biases = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * n]);
        let metadata = MicrofloatMetadata::new(
            MicrofloatFormat::Mxfp4,
            4,
            W13.group_size,
            MicrofloatLayout::OutputInput,
            W13.experts,
            W13.n,
            W13.k,
        )
        .unwrap();
        let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, &spread_ids);
        let input = alloc_allocation::<Metal, bf16>(context, k);
        let mut output = alloc_allocation::<Metal, u8>(context, routes * hidden * 4);
        let routes_per_token = NonZeroU32::new(W13.routes).unwrap();
        let expert_count = NonZeroU32::new(W13.experts).unwrap();
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            crate::data_type::DataType::F32,
        )
        .expect("MatmulKernel");
        group.throughput(Throughput::Bytes(mxfp4_expert_bytes(&W13)));
        group.bench_function(BenchmarkId::new("mxfp4_routed_fused_gate_up", "W13_N5760_K2880_G16_spread"), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                gate_act: Some(crate::backends::common::kernel::matmul::GateActMulDOps {
                                    activation_alpha: Some(1.702),
                                    gate_clipping: Some((-7.0, 7.0)),
                                    value_clipping: Some((-7.0, 7.0)),
                                }),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Tokens,
                            }),
                            m: W13.routes,
                            n: W13.n,
                            k: W13.k,
                        },
                        encoder,
                    )
                    .expect("fused mxfp4 routed matmul");
            });
        });
    }
}

#[uzu_bench]
fn bench_bf16_lm_head_preflight(c: &mut Criterion) {
    let context = &*shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/Bf16LmHead", type_short_name::<Metal>()));

    // GPT-OSS LM head: dense BF16 GEMV, M=1, N=201088, K=2880.
    let (m, n, k) = (1usize, 201088usize, 2880usize);
    let input = alloc_allocation::<Metal, bf16>(context, m * k);
    let weights = alloc_allocation::<Metal, bf16>(context, n * k);
    let mut output = alloc_allocation::<Metal, bf16>(context, m * n);
    let mut matmul =
        <MetalKernels as Kernels>::MatmulKernel::new(context, bf16::data_type(), bf16::data_type(), bf16::data_type())
            .expect("MatmulKernel");
    let bytes = (n * k * 2) as u64;
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);
    group.bench_function(BenchmarkId::new("BF16", "M1_N201088_K2880"), |bencher| {
        iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
            matmul
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &input,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: &weights,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        routing: MatmulRouting::Dense,
                        m: m as u32,
                        n: n as u32,
                        k: k as u32,
                    },
                    encoder,
                )
                .expect("lm head matmul");
        });
    });
}
