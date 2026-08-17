#![cfg(backend = "metal")]

use std::num::NonZeroU32;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Kernels,
            kernel::{
                FullPrecisionEmbeddingLookupKernel,
                matmul::{
                    ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel,
                    MatmulRouting,
                },
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

fn bench_shape(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    context: &MetalContext,
    token_count: u32,
    routes_per_token: u32,
    expert_count: u32,
    n: u32,
    k: u32,
    microfloat: bool,
) {
    let route_count = token_count * routes_per_token;
    let input = alloc_allocation::<Metal, bf16>(context, token_count as usize * k as usize);
    let weights = alloc_allocation::<Metal, bf16>(context, expert_count as usize * n as usize * k as usize);
    let codes =
        alloc_allocation_with_data::<Metal, u8>(context, &vec![0; expert_count as usize * n as usize * k as usize / 2]);
    let scales = alloc_allocation_with_data::<Metal, u8>(
        context,
        &vec![127; expert_count as usize * n as usize * k as usize / 32],
    );
    let outer_scales =
        alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); expert_count as usize]);
    let microfloat_metadata =
        MicrofloatMetadata::new(MicrofloatFormat::Mxfp4, 4, 32, MicrofloatLayout::OutputInput, expert_count, n, k)
            .unwrap();
    let route_token_ids: Vec<u32> = (0..route_count).map(|route| route / routes_per_token).collect();
    let route_token_ids = alloc_allocation_with_data::<Metal, u32>(context, &route_token_ids);
    let expert_ids: Vec<i32> = (0..route_count).map(|route| (route % expert_count) as i32).collect();
    let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, &expert_ids);
    let mut expanded_input = alloc_allocation::<Metal, bf16>(context, route_count as usize * k as usize);
    let mut output = alloc_allocation::<Metal, bf16>(context, route_count as usize * n as usize);
    let mut matmul =
        <MetalKernels as Kernels>::MatmulKernel::new(context, bf16::data_type(), bf16::data_type(), bf16::data_type())
            .expect("MatmulKernel");
    let gather = <MetalKernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, bf16::data_type())
        .expect("FullPrecisionEmbeddingLookupKernel");
    let routes_per_token = NonZeroU32::new(routes_per_token).unwrap();
    let expert_count = NonZeroU32::new(expert_count).unwrap();
    let storage = if microfloat {
        "MXFP4"
    } else {
        "BF16"
    };
    let shape = format!("{storage}_T{token_count}_R{route_count}_N{n}_K{k}");

    group.throughput(Throughput::Elements(2 * u64::from(route_count) * u64::from(k) * u64::from(n)));
    group.bench_function(BenchmarkId::new("direct", &shape), |bencher| {
        iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
            matmul
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &input,
                            offset: 0,
                        },
                        b: if microfloat {
                            MatmulB::Microfloat {
                                codes: &codes,
                                scales: &scales,
                                outer_scales: &outer_scales,
                                metadata: microfloat_metadata,
                            }
                        } else {
                            MatmulB::FullPrecision {
                                b: &weights,
                            }
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        routing: MatmulRouting::Experts(ExpertRoutes {
                            expert_ids: &expert_ids,
                            routes_per_token,
                            expert_count,
                            input: ExpertInput::Tokens,
                        }),
                        m: route_count,
                        n,
                        k,
                    },
                    encoder,
                )
                .expect("direct expert matmul");
        });
    });
    group.bench_function(BenchmarkId::new("expand_then_route", &shape), |bencher| {
        iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
            gather.encode(&route_token_ids, &input, &mut expanded_input, route_count, token_count, k, 1.0, encoder);
            matmul
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &expanded_input,
                            offset: 0,
                        },
                        b: if microfloat {
                            MatmulB::Microfloat {
                                codes: &codes,
                                scales: &scales,
                                outer_scales: &outer_scales,
                                metadata: microfloat_metadata,
                            }
                        } else {
                            MatmulB::FullPrecision {
                                b: &weights,
                            }
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        routing: MatmulRouting::Experts(ExpertRoutes {
                            expert_ids: &expert_ids,
                            routes_per_token,
                            expert_count,
                            input: ExpertInput::Routes,
                        }),
                        m: route_count,
                        n,
                        k,
                    },
                    encoder,
                )
                .expect("expanded expert matmul");
        });
    });
}

#[uzu_bench]
fn bench_direct_expert_routes(c: &mut Criterion) {
    let context = shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/ExpertRoutes", type_short_name::<Metal>()));
    for microfloat in [false, true] {
        for token_count in [1, 32] {
            bench_shape(&mut group, &context, token_count, 4, 16, 1024, 1024, microfloat);
        }
    }
}
