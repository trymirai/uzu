#![cfg(metal_backend)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder, QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationsPrepareKernel, HadamardTransformKernel, Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{DeviceTier, GemmDispatchPath, GemvDispatch, GemvSpecialization, Metal, MetalContext},
    },
    data_type::DataType,
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::{QuantInput, iter_encode_loop_named},
        util::{shared_metal_context, type_short_name},
    },
};

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
type MetalPrepare = <<Metal as Backend>::Kernels as Kernels>::ActivationsPrepareKernel;
type MetalHadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel;

const LAYERS: &[(&str, usize, usize)] =
    &[("qkv", 2048, 3072), ("o", 2048, 2048), ("up", 2048, 12288), ("down", 6144, 2048)];
const BATCHES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
const WIDTHS: &[u32] = &[8, 4];

#[derive(Clone, Copy)]
enum BenchPath {
    A8Gemm,
    Bf16Gemm,
    Bf16Gemv,
}

const PATHS: &[BenchPath] = &[BenchPath::A8Gemm, BenchPath::Bf16Gemm, BenchPath::Bf16Gemv];

impl BenchPath {
    fn label(self) -> &'static str {
        match self {
            BenchPath::A8Gemm => "a8_gemm",
            BenchPath::Bf16Gemm => "bf16_gemm",
            BenchPath::Bf16Gemv => "bf16_gemv",
        }
    }
}

struct BenchmarkData {
    weights: Allocation<Metal>,
    weight_scales: Allocation<Metal>,
    activations: Allocation<Metal>,
    rht_factors: Allocation<Metal>,
    a_working: Allocation<Metal>,
    a_int8: Allocation<Metal>,
    a_scales: Allocation<Metal>,
    output: Allocation<Metal>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    mode: QuantizationMode,
}

impl BenchmarkData {
    fn new(
        context: &MetalContext,
        m: usize,
        k: usize,
        n: usize,
        bits: u32,
    ) -> Self {
        let group_size = HADAMARD_TRANSFORM_BLOCK_SIZE as u32;
        let seed = 0xA8_00 ^ u64::from(bits) ^ k as u64 ^ n as u64;
        let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleSymmetric, seed);
        let rht: Vec<i32> = (0..k).map(|index| if index % 3 == 0 { -1 } else { 1 }).collect();
        let groups = k / group_size as usize;

        Self {
            weights: alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed),
            weight_scales: alloc_allocation_with_data::<Metal, bf16>(context, &input.scales),
            activations: alloc_allocation_with_data::<Metal, bf16>(context, &input.x),
            rht_factors: alloc_allocation_with_data::<Metal, i32>(context, &rht),
            a_working: alloc_allocation::<Metal, bf16>(context, m * k),
            a_int8: alloc_allocation::<Metal, i8>(context, m * k),
            a_scales: alloc_allocation::<Metal, f32>(context, m * groups),
            output: alloc_allocation::<Metal, bf16>(context, m * n),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            group_size,
            mode: input.mode,
        }
    }

    fn arguments<'a>(
        &'a mut self,
        a8: bool,
    ) -> MatmulArguments<'a, 'a, 'a, Metal, &'a Allocation<Metal>> {
        let a = if a8 {
            MatmulA::Int8Symmetric {
                values: &self.a_int8,
                scales: &self.a_scales,
            }
        } else {
            MatmulA::FullPrecision {
                values: &self.a_working,
                offset: 0,
            }
        };
        MatmulArguments {
            a,
            b: MatmulB::ScaleSymmetricDequant {
                b: &self.weights,
                scales: &self.weight_scales,
                mode: self.mode,
                group_size: self.group_size,
            },
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut self.output,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }
}

struct BenchKernels {
    prepare: MetalPrepare,
    hadamard: MetalHadamard,
    matmul: MetalMatmul,
    gemv: GemvDispatch,
    device_tier: DeviceTier,
}

fn encode_step(
    path: BenchPath,
    data: &mut BenchmarkData,
    kernels: &mut BenchKernels,
    encoder: &mut Encoder<Metal>,
) {
    match path {
        BenchPath::A8Gemm => {
            kernels.prepare.encode(
                &data.activations,
                &mut data.a_int8,
                &mut data.a_scales,
                &data.rht_factors,
                data.m,
                data.k,
                data.group_size,
                encoder,
            );
            let args = data.arguments(true);
            kernels.matmul.gemm.encode_dispatch_path(args, GemmDispatchPath::Mxu, encoder).expect("a8 gemm encode");
        },
        BenchPath::Bf16Gemm => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            kernels.hadamard.encode(&mut data.a_working, &data.rht_factors, data.k, data.m, encoder);
            let args = data.arguments(false);
            kernels.matmul.gemm.encode_dispatch_path(args, GemmDispatchPath::Mxu, encoder).expect("bf16 gemm encode");
        },
        BenchPath::Bf16Gemv => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            kernels.hadamard.encode(&mut data.a_working, &data.rht_factors, data.k, data.m, encoder);
            let args = data.arguments(false);
            let specialization = GemvSpecialization::select_for_any_batch(
                &args,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
                kernels.device_tier,
            )
            .expect("gemv specialization");
            kernels.gemv.encode(args, specialization, encoder).expect("gemv encode");
        },
    }
}

#[uzu_bench]
fn bench_a8w(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }

    let mut kernels = BenchKernels {
        prepare: <MetalPrepare as ActivationsPrepareKernel>::new(&context, DataType::BF16).expect("prepare kernel"),
        hadamard: <MetalHadamard as HadamardTransformKernel>::new(
            &context,
            DataType::BF16,
            HadamardTransformOrder::Input,
        )
        .expect("hadamard kernel"),
        matmul: <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("matmul kernel"),
        gemv: GemvDispatch::new(DataType::BF16, DataType::BF16, DataType::BF16),
        device_tier: context.device_tier(),
    };

    let mut group = c.benchmark_group(format!("{}/Kernel/A8W", type_short_name::<Metal>()));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for &(layer, k, n) in LAYERS {
        for &m in BATCHES {
            let shape_label = format!("{layer}_m{m}_k{k}_n{n}");
            group.throughput(Throughput::Elements((m * k * n) as u64));
            for &bits in WIDTHS {
                let mut data = BenchmarkData::new(&context, m, k, n, bits);
                for &path in PATHS {
                    let function_id = format!("w{bits}_{}", path.label());
                    group.bench_function(BenchmarkId::new(&function_id, &shape_label), |bench| {
                        let benchmark_path =
                            format!("{}/Kernel/A8W/{function_id}/{shape_label}", type_short_name::<Metal>());
                        iter_encode_loop_named::<Metal, _>(&context, bench, &benchmark_path, |encoder| {
                            encode_step(path, &mut data, &mut kernels, encoder);
                        });
                    });
                }
            }
        }
    }
    group.finish();
}
