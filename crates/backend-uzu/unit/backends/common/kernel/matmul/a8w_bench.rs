#![cfg(metal_backend)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::{HadamardTransformOrder, QuantizationMethod, QuantizationMode},
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

// Symmetric int8 activations always quantize with this group size, independent
// of the weight group size (see the ActivationsPrepare kernel).
const ACTIVATION_GROUP_SIZE: u32 = 32;

const SHAPES: &[(usize, usize, usize)] = &[
    (1, 2048, 2048),
    (2, 2048, 2048),
    (4, 2048, 2048),
    (8, 2048, 2048),
    (16, 2048, 2048),
    (32, 2048, 2048),
    (64, 2048, 2048),
];

/// The three RHT-linear execution paths we compare on each shape.
#[derive(Clone, Copy)]
enum BenchPath {
    A8GemmMxu,
    Bf16GemmMxu,
    Bf16Gemv,
}

impl BenchPath {
    fn label(self) -> &'static str {
        match self {
            BenchPath::A8GemmMxu => "a8_gemm_mxu",
            BenchPath::Bf16GemmMxu => "abf16_gemm_mxu",
            BenchPath::Bf16Gemv => "abf16_gemv",
        }
    }
}

/// Device buffers for one RHT-linear shape. Weights are stored unsigned exactly
/// as they ship; every path (A8 MXU, BF16 GEMM, BF16 GEMV) shares the same
/// buffer, and the A8 shader applies the signed conversion just before the MMA.
struct BenchmarkData {
    weights_u8: Allocation<Metal>,
    weight_scales: Allocation<Metal>,
    activations: Allocation<Metal>,
    rht_factors: Allocation<Metal>,
    a_working: Allocation<Metal>,
    a_int8: Allocation<Metal>,
    a_scales: Allocation<Metal>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    mode: QuantizationMode,
}

impl BenchmarkData {
    /// Builds the unsigned weight/scale/activation buffers shared by all paths.
    fn new(
        context: &MetalContext,
        m: usize,
        k: usize,
        n: usize,
        bits: u32,
        seed: u64,
    ) -> Self {
        let group_size = ACTIVATION_GROUP_SIZE;
        let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleSymmetric, seed);

        let weights_u8 = alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed);
        let weight_scales = alloc_allocation_with_data::<Metal, bf16>(context, &input.scales);
        let activations = alloc_allocation_with_data::<Metal, bf16>(context, &input.x);
        let rht: Vec<i32> = (0..k)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let rht_factors = alloc_allocation_with_data::<Metal, i32>(context, &rht);

        let groups = k / group_size as usize;
        Self {
            weights_u8,
            weight_scales,
            activations,
            rht_factors,
            a_working: alloc_allocation::<Metal, bf16>(context, m * k),
            a_int8: alloc_allocation::<Metal, i8>(context, m * k),
            a_scales: alloc_allocation::<Metal, f32>(context, m * groups),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            group_size,
            mode: if bits == 4 {
                QuantizationMode::U4
            } else {
                QuantizationMode::U8
            },
        }
    }

    /// Full-precision (BF16) matmul arguments reading the already-transformed
    /// activation working buffer and the unsigned weights.
    fn bf16_arguments<'a>(
        &'a self,
        output: &'a mut Allocation<Metal>,
    ) -> MatmulArguments<'a, 'a, 'a, Metal, &'a Allocation<Metal>> {
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &self.a_working,
                offset: 0,
            },
            b: MatmulB::ScaleSymmetricDequant {
                b: &self.weights_u8,
                scales: &self.weight_scales,
                mode: self.mode,
                group_size: self.group_size,
            },
            b_leading_dimension: None,
            b_transpose: true,
            d: output,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_step(
    path: BenchPath,
    data: &mut BenchmarkData,
    output: &mut Allocation<Metal>,
    prepare: &MetalPrepare,
    hadamard: &MetalHadamard,
    matmul: &mut MetalMatmul,
    gemv: &mut GemvDispatch,
    device_tier: DeviceTier,
    encoder: &mut Encoder<Metal>,
) {
    match path {
        BenchPath::A8GemmMxu => {
            // Fused input RHT + int8 quantization, then the A8 GEMM-MXU matmul.
            // The shader converts the unsigned weights to signed just before the
            // MMA, so this shares the same buffer as the BF16 paths.
            prepare.encode(
                &data.activations,
                &mut data.a_int8,
                &mut data.a_scales,
                &data.rht_factors,
                data.m,
                data.k,
                ACTIVATION_GROUP_SIZE,
                encoder,
            );
            let args: MatmulArguments<'_, '_, '_, Metal, &Allocation<Metal>> = MatmulArguments {
                a: MatmulA::Int8Symmetric {
                    values: &data.a_int8,
                    scales: &data.a_scales,
                    group_size: ACTIVATION_GROUP_SIZE,
                },
                b: MatmulB::ScaleSymmetricDequant {
                    b: &data.weights_u8,
                    scales: &data.weight_scales,
                    mode: data.mode,
                    group_size: data.group_size,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                m: data.m,
                n: data.n,
                k: data.k,
            };
            matmul.gemm.encode_dispatch_path(args, GemmDispatchPath::Mxu, encoder).expect("a8 gemm mxu encode");
        },
        BenchPath::Bf16GemmMxu => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            hadamard.encode(&mut data.a_working, &data.rht_factors, data.k, data.m, encoder);
            let args = data.bf16_arguments(output);
            matmul.gemm.encode_dispatch_path(args, GemmDispatchPath::Mxu, encoder).expect("bf16 gemm mxu encode");
        },
        BenchPath::Bf16Gemv => {
            encoder.encode_copy(&data.activations, .., &mut data.a_working, ..);
            hadamard.encode(&mut data.a_working, &data.rht_factors, data.k, data.m, encoder);
            let args = data.bf16_arguments(output);
            let spec = GemvSpecialization::select(&args, DataType::BF16, DataType::BF16, DataType::BF16, device_tier)
                .expect("bf16 gemv specialization");
            gemv.encode(args, spec, encoder).expect("bf16 gemv encode");
        },
    }
}

fn bench_bits(
    c: &mut Criterion,
    context: &MetalContext,
    device_tier: DeviceTier,
    prepare: &MetalPrepare,
    hadamard: &MetalHadamard,
    bits: u32,
) {
    let mut matmul = <MetalMatmul as MatmulKernel>::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut gemv = GemvDispatch::new(DataType::BF16, DataType::BF16, DataType::BF16);

    let mut group = c.benchmark_group(format!("{}/Kernel/A8W/w{bits}", type_short_name::<Metal>()));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(800));

    for &(m, k, n) in SHAPES {
        let mut data = BenchmarkData::new(context, m, k, n, bits, 0xA8_00 ^ u64::from(bits) ^ k as u64);
        let mut output = alloc_allocation::<Metal, bf16>(context, m * n);
        let shape_label = format!("m{m}_k{k}_n{n}");

        // GEMV is the decode path: only emit it where the specialization applies
        // (small m); prefill shapes fall back to A8 vs BF16 GEMM-MXU only.
        let gemv_eligible = GemvSpecialization::select(
            &data.bf16_arguments(&mut output),
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            device_tier,
        )
        .is_some();

        let mut paths = vec![BenchPath::A8GemmMxu, BenchPath::Bf16GemmMxu];
        if gemv_eligible {
            paths.push(BenchPath::Bf16Gemv);
        }

        group.throughput(Throughput::Elements((m * k * n) as u64));
        for path in paths {
            group.bench_function(BenchmarkId::new(path.label(), &shape_label), |bench| {
                let benchmark_path =
                    format!("{}/Kernel/A8W/w{bits}/{}/{shape_label}", type_short_name::<Metal>(), path.label());
                iter_encode_loop_named::<Metal, _>(context, bench, &benchmark_path, |encoder| {
                    encode_step(
                        path,
                        &mut data,
                        &mut output,
                        prepare,
                        hadamard,
                        &mut matmul,
                        &mut gemv,
                        device_tier,
                        encoder,
                    );
                });
            });
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_a8w(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let device_tier = context.device_tier();

    let prepare = <MetalPrepare as ActivationsPrepareKernel>::new(&context, DataType::BF16).expect("prepare kernel");
    let hadamard =
        <MetalHadamard as HadamardTransformKernel>::new(&context, DataType::BF16, HadamardTransformOrder::Input)
            .expect("hadamard kernel");

    for bits in [8u32, 4u32] {
        bench_bits(c, &context, device_tier, &prepare, &hadamard, bits);
    }
}
