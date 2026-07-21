use std::time::Duration;

use backend_uzu::{
    backends::{
        common::{
            Allocation, AllocationType, Backend, Context, Encoder,
            gpu_types::{ACTIVATION_QUANTIZATION_GROUP_SIZE, ActivationPrepareOps, HadamardTransformOrder, QuantizationMode},
            kernel::{
                ActivationsPrepareKernel, HadamardTransformKernel, Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
    data_type::DataType,
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
type MetalPrepare = <<Metal as Backend>::Kernels as Kernels>::ActivationsPrepareKernel;
type MetalHadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel;

const SHAPES: &[(usize, usize, usize)] = &[
    (1, 2048, 2048),
    (2, 2048, 2048),
    (4, 2048, 2048),
    (8, 2048, 2048),
    (16, 2048, 2048),
    (32, 2048, 2048),
    (64, 2048, 2048),
];

#[derive(Clone, Copy)]
enum Path {
    A8Mxu,
    Bf16Gemv,
}

impl Path {
    fn label(
        self,
        bits: u32,
    ) -> String {
        match self {
            Self::A8Mxu => format!("a8w{bits}_mxu"),
            Self::Bf16Gemv => format!("bf16w{bits}_gemv"),
        }
    }
}

struct HostInputs {
    weights_u8: Vec<u8>,
    weight_scales: Vec<bf16>,
    activations: Vec<bf16>,
    rht_factors: Vec<i32>,
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
}

impl HostInputs {
    fn generate(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
        bits: u32,
        seed: u64,
    ) -> Self {
        assert!(bits == 4 || bits == 8);
        assert!(k.is_multiple_of(group_size as usize));
        let groups = k / group_size as usize;
        let weight_bytes = n * k * (bits as usize) / 8;
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            weights_u8: (0..weight_bytes).map(|_| rng.random_range(0u32..256) as u8).collect(),
            weight_scales: (0..n * groups).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.3f32))).collect(),
            activations: (0..m * k).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0f32))).collect(),
            rht_factors: (0..k)
                .map(|index| {
                    if index % 3 == 0 {
                        -1
                    } else {
                        1
                    }
                })
                .collect(),
            m,
            k,
            n,
            group_size,
            bits,
        }
    }
}

struct DeviceBuffers {
    weights_u8: Allocation<Metal>,
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
    bits: u32,
}

fn upload(
    context: &MetalContext,
    host: &HostInputs,
) -> DeviceBuffers {
    let groups = host.k / host.group_size as usize;
    let mut weights_u8 = context.create_allocation(host.weights_u8.len(), AllocationType::Global).expect("weights");
    weights_u8.copyin(&host.weights_u8);
    let mut weight_scales = context
        .create_allocation(host.weight_scales.len() * size_of::<bf16>(), AllocationType::Global)
        .expect("weight scales");
    weight_scales.copyin(&host.weight_scales);
    let mut activations = context
        .create_allocation(host.activations.len() * size_of::<bf16>(), AllocationType::Global)
        .expect("activations");
    activations.copyin(&host.activations);
    let mut rht_factors =
        context.create_allocation(host.rht_factors.len() * size_of::<i32>(), AllocationType::Global).expect("rht");
    rht_factors.copyin(&host.rht_factors);

    DeviceBuffers {
        weights_u8,
        weight_scales,
        activations,
        rht_factors,
        a_working: context
            .create_allocation(host.m * host.k * size_of::<bf16>(), AllocationType::Global)
            .expect("a working"),
        a_int8: context.create_allocation(host.m * host.k, AllocationType::Global).expect("a int8"),
        a_scales: context
            .create_allocation(host.m * groups * size_of::<f32>(), AllocationType::Global)
            .expect("a scales"),
        output: context.create_allocation(host.m * host.n * size_of::<bf16>(), AllocationType::Global).expect("output"),
        m: host.m as u32,
        k: host.k as u32,
        n: host.n as u32,
        group_size: host.group_size,
        bits: host.bits,
    }
}

fn weight_mode(bits: u32) -> QuantizationMode {
    if bits == 4 {
        QuantizationMode::U4
    } else {
        QuantizationMode::U8
    }
}

fn encode_a8_mxu(
    prepare: &MetalPrepare,
    matmul: &mut MetalMatmul,
    buffers: &mut DeviceBuffers,
    encoder: &mut Encoder<Metal>,
) {
    prepare.encode(
        &buffers.activations,
        &mut buffers.a_int8,
        &mut buffers.a_scales,
        Some(&buffers.rht_factors),
        buffers.m,
        buffers.k,
        buffers.group_size,
        encoder,
    );
    let arguments: MatmulArguments<'_, '_, '_, Metal> = MatmulArguments {
        a: MatmulA::Int8Symmetric {
            values: &buffers.a_int8,
            scales: &buffers.a_scales,
            group_size: buffers.group_size,
        },
        b: MatmulB::ScaleSymmetricDequant {
            b: &buffers.weights_u8,
            scales: &buffers.weight_scales,
            mode: weight_mode(buffers.bits),
            group_size: buffers.group_size,
        },
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.output,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m: buffers.m,
        n: buffers.n,
        k: buffers.k,
    };
    matmul
        .gemm
        .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
        .unwrap_or_else(|error| panic!("a8w{} MXU: {error}", buffers.bits));
}

fn encode_bf16_gemv(
    hadamard: &MetalHadamard,
    matmul: &mut MetalMatmul,
    buffers: &mut DeviceBuffers,
    encoder: &mut Encoder<Metal>,
) -> bool {
    encoder.encode_copy(&buffers.activations, .., &mut buffers.a_working, ..);
    hadamard.encode(&mut buffers.a_working, &buffers.rht_factors, buffers.k, buffers.m, encoder);
    let arguments: MatmulArguments<'_, '_, '_, Metal> = MatmulArguments {
        a: MatmulA::FullPrecision {
            values: &buffers.a_working,
            offset: 0,
        },
        b: MatmulB::ScaleSymmetricDequant {
            b: &buffers.weights_u8,
            scales: &buffers.weight_scales,
            mode: weight_mode(buffers.bits),
            group_size: buffers.group_size,
        },
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.output,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m: buffers.m,
        n: buffers.n,
        k: buffers.k,
    };
    matmul.try_encode_gemv(arguments, encoder).unwrap_or_else(|error| panic!("bf16w{} GEMV: {error}", buffers.bits))
}

fn gemv_eligible(
    hadamard: &MetalHadamard,
    matmul: &mut MetalMatmul,
    buffers: &mut DeviceBuffers,
    context: &MetalContext,
) -> bool {
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    let ok = encode_bf16_gemv(hadamard, matmul, buffers, &mut encoder);
    let _ = encoder.end_encoding().submit().wait_until_completed();
    ok
}

#[uzu_bench]
fn bench_a8w8(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let prepare = <MetalPrepare as ActivationsPrepareKernel>::new(
        &context,
        DataType::BF16,
        ActivationPrepareOps::INPUT_RHT | ActivationPrepareOps::QUANTIZE,
    )
    .expect("prepare kernel");
    let hadamard = <MetalHadamard as HadamardTransformKernel>::new(&context, DataType::BF16, HadamardTransformOrder::Input)
        .expect("hadamard kernel");

    let mut group = c.benchmark_group("Metal/Kernel/A8W");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(800));

    for &(m, k, n) in SHAPES {
        for bits in [8u32, 4u32] {
            let group_size = ACTIVATION_QUANTIZATION_GROUP_SIZE;
            let host = HostInputs::generate(m, k, n, group_size, bits, 0xA8_00 ^ (bits as u64) ^ k as u64);
            let mut buffers = upload(&context, &host);
            group.throughput(Throughput::Elements((m * k * n) as u64));

            let label = format!("m{m}_k{k}_n{n}_w{bits}_gs{group_size}");
            group.bench_with_input(BenchmarkId::new(Path::A8Mxu.label(bits), &label), &bits, |b, _| {
                b.iter_custom(|iterations| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iterations {
                        encode_a8_mxu(&prepare, &mut matmul, &mut buffers, &mut encoder);
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });

            if gemv_eligible(&hadamard, &mut matmul, &mut buffers, &context) {
                group.bench_with_input(BenchmarkId::new(Path::Bf16Gemv.label(bits), &label), &bits, |b, _| {
                    b.iter_custom(|iterations| {
                        let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                        for _ in 0..iterations {
                            assert!(encode_bf16_gemv(&hadamard, &mut matmul, &mut buffers, &mut encoder));
                        }
                        encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                    });
                });
            }
        }
    }
    group.finish();
}
