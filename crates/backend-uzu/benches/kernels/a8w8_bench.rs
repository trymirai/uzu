use std::time::Duration;

use backend_uzu::{
    backends::{
        common::{
            Allocation, AllocationType, Backend, Context, Encoder,
            gpu_types::{HadamardTransformOrder, QuantizationMode},
            kernel::{
                ActivationsPrepareKernel, HadamardTransformKernel, Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{Metal, MetalContext},
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

struct DeviceInputs {
    /// Unsigned packed codes for the BF16 quantized baseline.
    weights_u8: Allocation<Metal>,
    /// Midpoint-signed codes for the A8 MXU path.
    weights_signed: Allocation<Metal>,
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

fn device_inputs(
    context: &MetalContext,
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    seed: u64,
) -> DeviceInputs {
    assert!(bits == 4 || bits == 8);
    assert!(k.is_multiple_of(group_size as usize));
    let groups = k / group_size as usize;
    let mut rng = SmallRng::seed_from_u64(seed);

    let weights: Vec<u8> = (0..n * k * (bits as usize) / 8).map(|_| rng.random_range(0u32..256) as u8).collect();
    let midpoint_mask = if bits == 4 { 0x88u8 } else { 0x80u8 };
    let weights_signed: Vec<u8> = weights.iter().map(|code| code ^ midpoint_mask).collect();
    let weight_scales: Vec<bf16> = (0..n * groups).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.3f32))).collect();
    let activations: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0f32))).collect();
    let rht_factors: Vec<i32> = (0..k)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let alloc = |bytes: usize| context.create_allocation(bytes, AllocationType::Global).expect("allocation");
    let upload = |data: &[u8]| {
        let mut allocation = alloc(data.len());
        allocation.copyin(data);
        allocation
    };
    let upload_pod = |bytes: usize, copyin: &mut dyn FnMut(&mut Allocation<Metal>)| {
        let mut allocation = alloc(bytes);
        copyin(&mut allocation);
        allocation
    };
    DeviceInputs {
        weights_u8: upload(&weights),
        weights_signed: upload(&weights_signed),
        weight_scales: upload_pod(size_of_val(weight_scales.as_slice()), &mut |a| a.copyin(&weight_scales)),
        activations: upload_pod(size_of_val(activations.as_slice()), &mut |a| a.copyin(&activations)),
        rht_factors: upload_pod(size_of_val(rht_factors.as_slice()), &mut |a| a.copyin(&rht_factors)),
        a_working: alloc(m * k * size_of::<bf16>()),
        a_int8: alloc(m * k),
        a_scales: alloc(m * groups * size_of::<f32>()),
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

fn arguments<'a, 'b, 'd>(
    a: MatmulA<'a, Metal>,
    inputs: &'b DeviceInputs,
    signed_weights: bool,
    output: &'d mut Allocation<Metal>,
) -> MatmulArguments<'a, 'b, 'd, Metal, &'b Allocation<Metal>> {
    MatmulArguments {
        a,
        b: MatmulB::ScaleSymmetricDequant {
            b: if signed_weights {
                &inputs.weights_signed
            } else {
                &inputs.weights_u8
            },
            scales: &inputs.weight_scales,
            mode: inputs.mode,
            group_size: inputs.group_size,
        },
        b_leading_dimension: None,
        b_transpose: true,
        d: output,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m: inputs.m,
        n: inputs.n,
        k: inputs.k,
    }
}

#[uzu_bench]
fn bench_a8w8(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let prepare = <MetalPrepare as ActivationsPrepareKernel>::new(&context, DataType::BF16).expect("prepare kernel");
    let hadamard =
        <MetalHadamard as HadamardTransformKernel>::new(&context, DataType::BF16, HadamardTransformOrder::Input)
            .expect("hadamard kernel");

    let mut group = c.benchmark_group("Metal/Kernel/A8W");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(800));

    for &(m, k, n) in SHAPES {
        for bits in [8u32, 4u32] {
            let group_size = 32;
            let mut inputs = device_inputs(&context, m, k, n, group_size, bits, 0xA8_00 ^ (bits as u64) ^ k as u64);
            let mut output = context
                .create_allocation(m * n * size_of::<bf16>(), AllocationType::Global)
                .expect("output allocation");
            group.throughput(Throughput::Elements((m * k * n) as u64));
            let label = format!("m{m}_k{k}_n{n}_w{bits}_gs{group_size}");

            group.bench_with_input(BenchmarkId::new(format!("a8w{bits}"), &label), &bits, |bench, _| {
                bench.iter_custom(|iterations| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iterations {
                        prepare.encode(
                            &inputs.activations,
                            &mut inputs.a_int8,
                            &mut inputs.a_scales,
                            &inputs.rht_factors,
                            inputs.m,
                            inputs.k,
                            inputs.group_size,
                            &mut encoder,
                        );
                        let a = MatmulA::Int8Symmetric {
                            values: &inputs.a_int8,
                            scales: &inputs.a_scales,
                            group_size: inputs.group_size,
                        };
                        matmul.encode(arguments(a, &inputs, true, &mut output), &mut encoder).expect("a8 encode");
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });

            group.bench_with_input(BenchmarkId::new(format!("bf16w{bits}"), &label), &bits, |bench, _| {
                bench.iter_custom(|iterations| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iterations {
                        encoder.encode_copy(&inputs.activations, .., &mut inputs.a_working, ..);
                        hadamard.encode(&mut inputs.a_working, &inputs.rht_factors, inputs.k, inputs.m, &mut encoder);
                        let a = MatmulA::FullPrecision {
                            values: &inputs.a_working,
                            offset: 0,
                        };
                        matmul.encode(arguments(a, &inputs, false, &mut output), &mut encoder).expect("bf16 encode");
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });
        }
    }
    group.finish();
}
