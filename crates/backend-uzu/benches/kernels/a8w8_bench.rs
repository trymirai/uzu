use std::time::Duration;

use backend_uzu::{
    backends::{
        common::{
            Allocation, AllocationType, Backend, Context, Encoder,
            gpu_types::{
                ACTIVATION_QUANTIZATION_GROUP_SIZE, ActivationPrepareOps, HadamardTransformOrder, QuantizationMode,
            },
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

const SHAPES: &[(usize, usize, usize, u32)] = &[
    (1, 2048, 2048, ACTIVATION_QUANTIZATION_GROUP_SIZE),
    (8, 2048, 2048, ACTIVATION_QUANTIZATION_GROUP_SIZE),
    (128, 2048, 2048, ACTIVATION_QUANTIZATION_GROUP_SIZE),
    (256, 4096, 2048, ACTIVATION_QUANTIZATION_GROUP_SIZE),
];

struct HostInputs {
    weights_u8: Vec<u8>,
    weight_scales: Vec<bf16>,
    activations: Vec<bf16>,
    rht_factors: Vec<i32>,
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
}

impl HostInputs {
    fn generate(
        m: usize,
        k: usize,
        n: usize,
        group_size: u32,
        seed: u64,
    ) -> Self {
        let groups = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            weights_u8: (0..n * k).map(|_| rng.random_range(0u32..256) as u8).collect(),
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
        }
    }
}

struct DeviceBuffers {
    weights_u8: Allocation<Metal>,
    weights_signed: Allocation<Metal>,
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
}

fn upload(
    context: &MetalContext,
    host: &HostInputs,
) -> DeviceBuffers {
    let groups = host.k.div_ceil(host.group_size as usize);
    let signed: Vec<u8> = host.weights_u8.iter().map(|code| code ^ 0x80).collect();
    let mut weights_u8 = context.create_allocation(host.weights_u8.len(), AllocationType::Global).expect("u8 w");
    weights_u8.copyin(&host.weights_u8);
    let mut weights_signed = context.create_allocation(signed.len(), AllocationType::Global).expect("signed w");
    weights_signed.copyin(&signed);
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
        weights_signed,
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
    }
}

fn encode_a8w8(
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
            b: &buffers.weights_signed,
            scales: &buffers.weight_scales,
            mode: QuantizationMode::U8,
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
    matmul.gemm.encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder).expect("a8w8 GEMM");
}

fn encode_abf16_w8(
    hadamard: &MetalHadamard,
    matmul: &mut MetalMatmul,
    buffers: &mut DeviceBuffers,
    encoder: &mut Encoder<Metal>,
) {
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
            mode: QuantizationMode::U8,
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
    matmul.gemm.encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder).expect("abf16w8 GEMM");
}

#[derive(Clone, Copy)]
enum Mode {
    LinearA8W8,
    LinearABf16W8,
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
    let hadamard =
        <MetalHadamard as HadamardTransformKernel>::new(&context, DataType::BF16, HadamardTransformOrder::Input)
            .expect("RHT kernel");

    let mut group = c.benchmark_group("Metal/Kernel/A8W8");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for &(m, k, n, group_size) in SHAPES {
        let label = format!("m{m}_k{k}_n{n}_gs{group_size}");
        let host = HostInputs::generate(m, k, n, group_size, 0xB8_00 ^ k as u64);
        let mut buffers = upload(&context, &host);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        for (name, mode) in [("linear_a8w8", Mode::LinearA8W8), ("linear_abf16w8", Mode::LinearABf16W8)] {
            group.bench_with_input(BenchmarkId::new(name, &label), &mode, |b, &mode| {
                b.iter_custom(|iterations| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iterations {
                        match mode {
                            Mode::LinearA8W8 => encode_a8w8(&prepare, &mut matmul, &mut buffers, &mut encoder),
                            Mode::LinearABf16W8 => encode_abf16_w8(&hadamard, &mut matmul, &mut buffers, &mut encoder),
                        }
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });
        }
    }
    group.finish();
}
