use std::time::Duration;

use backend_uzu::{
    backends::{
        common::{
            Allocation, AllocationType, Backend, Context, Encoder,
            gpu_types::{ActivationPrepareOps, ActivationScaleStatistic, HadamardTransformOrder, QuantizationMode},
            kernel::{
                ActivationsPrepareKernel, HadamardTransformKernel, Kernels, group_stat,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
                quantize_symmetric_i8, symmetric_divisor,
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

const CORRECTNESS_SHAPES: &[(usize, usize, usize, u32)] =
    &[(128, 256, 128, 32), (64, 512, 64, 64), (128, 256, 256, 128), (96, 256, 100, 32)];
const PERF_SHAPES: &[(usize, usize, usize, u32)] = &[(256, 2048, 2048, 64), (128, 4096, 2048, 128)];
const RHT_BLOCK_SIZE: usize = 32;

fn hadamard(values: &mut [f32; RHT_BLOCK_SIZE]) {
    let mut stride = 1;
    while stride < RHT_BLOCK_SIZE {
        for lane in 0..RHT_BLOCK_SIZE {
            if lane & stride == 0 {
                let left = values[lane];
                let right = values[lane | stride];
                values[lane] = left + right;
                values[lane | stride] = left - right;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (RHT_BLOCK_SIZE as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

struct HostInputs {
    weights: Vec<u8>,
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
            weights: (0..n * k).map(|_| rng.random_range(0u32..256) as u8).collect(),
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

    fn prepared_activations(&self) -> Vec<f32> {
        let mut prepared = vec![0.0f32; self.m * self.k];
        for row in 0..self.m {
            for block_start in (0..self.k).step_by(RHT_BLOCK_SIZE) {
                let mut block = [0.0f32; RHT_BLOCK_SIZE];
                for lane in 0..RHT_BLOCK_SIZE {
                    let column = block_start + lane;
                    block[lane] = self.activations[row * self.k + column].to_f32() * self.rht_factors[column] as f32;
                }
                hadamard(&mut block);
                let start = row * self.k + block_start;
                prepared[start..start + RHT_BLOCK_SIZE].copy_from_slice(&block);
            }
        }
        prepared
    }

    fn reference(&self) -> Vec<f32> {
        let group_size = self.group_size as usize;
        let groups = self.k.div_ceil(group_size);
        let prepared = self.prepared_activations();
        let mut values = vec![0i8; self.m * self.k];
        let mut scales = vec![0.0f32; self.m * groups];

        for row in 0..self.m {
            for group in 0..groups {
                let start = group * group_size;
                let end = (start + group_size).min(self.k);
                let row_start = row * self.k;
                let divisor = symmetric_divisor(group_stat(
                    &prepared[row_start + start..row_start + end],
                    ActivationScaleStatistic::AbsMax,
                ));
                scales[row * groups + group] = divisor;
                for column in start..end {
                    values[row_start + column] = quantize_symmetric_i8(prepared[row_start + column], divisor);
                }
            }
        }

        let mut output = vec![0.0f32; self.m * self.n];
        for row in 0..self.m {
            for column in 0..self.n {
                for group in 0..groups {
                    let start = group * group_size;
                    let end = (start + group_size).min(self.k);
                    let dot = (start..end)
                        .map(|inner| {
                            values[row * self.k + inner] as i32 * (self.weights[column * self.k + inner] as i32 - 128)
                        })
                        .sum::<i32>();
                    output[row * self.n + column] += scales[row * groups + group]
                        * self.weight_scales[column * groups + group].to_f32()
                        * dot as f32;
                }
            }
        }
        output
    }
}

struct GpuBuffers {
    weights: Allocation<Metal>,
    weight_scales: Allocation<Metal>,
    activations: Allocation<Metal>,
    fp_activations: Allocation<Metal>,
    rht_factors: Allocation<Metal>,
    int8_activations: Allocation<Metal>,
    activation_scales: Allocation<Metal>,
    output: Allocation<Metal>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}

fn alloc_with<T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
    context: &MetalContext,
    data: &[T],
) -> Allocation<Metal> {
    let mut allocation =
        context.create_allocation(std::mem::size_of_val(data), AllocationType::Global).expect("allocation");
    allocation.copyin(data);
    allocation
}

fn upload(
    context: &MetalContext,
    host: &HostInputs,
) -> GpuBuffers {
    let activation_bytes = host.m * host.k * std::mem::size_of::<bf16>();
    let groups = host.k.div_ceil(host.group_size as usize);
    GpuBuffers {
        weights: alloc_with(context, &host.weights),
        weight_scales: alloc_with(context, &host.weight_scales),
        activations: alloc_with(context, &host.activations),
        fp_activations: context.create_allocation(activation_bytes, AllocationType::Global).expect("fp activations"),
        rht_factors: alloc_with(context, &host.rht_factors),
        int8_activations: context.create_allocation(host.m * host.k, AllocationType::Global).expect("int8 activations"),
        activation_scales: context
            .create_allocation(host.m * groups * std::mem::size_of::<f32>(), AllocationType::Global)
            .expect("activation scales"),
        output: context
            .create_allocation(host.m * host.n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("output"),
        m: host.m as u32,
        k: host.k as u32,
        n: host.n as u32,
        group_size: host.group_size,
    }
}

fn encode_int8(
    prepare: &MetalPrepare,
    matmul: &mut MetalMatmul,
    buffers: &mut GpuBuffers,
    encoder: &mut Encoder<Metal>,
) {
    prepare.encode(
        &buffers.activations,
        Some(&mut buffers.int8_activations),
        Some(&mut buffers.activation_scales),
        Some(&buffers.rht_factors),
        buffers.m,
        buffers.k,
        buffers.group_size,
        encoder,
    );
    let arguments: MatmulArguments<'_, '_, '_, Metal> = MatmulArguments {
        a: MatmulA::Int8Symmetric {
            values: &buffers.int8_activations,
            scales: &buffers.activation_scales,
            group_size: buffers.group_size,
        },
        b: MatmulB::ScaleSymmetricDequant {
            b: &buffers.weights,
            scales: &buffers.weight_scales,
            mode: QuantizationMode::I8,
            group_size: buffers.group_size,
        },
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.output,
        d_transform: MatmulDOps::none(),
        m: buffers.m,
        n: buffers.n,
        k: buffers.k,
    };
    matmul.gemm.encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder).expect("int8 GEMM");
}

fn encode_bf16(
    hadamard: &MetalHadamard,
    matmul: &mut MetalMatmul,
    buffers: &mut GpuBuffers,
    encoder: &mut Encoder<Metal>,
) {
    encoder.encode_copy(&buffers.activations, .., &mut buffers.fp_activations, ..);
    hadamard.encode(&mut buffers.fp_activations, &buffers.rht_factors, buffers.k, buffers.m, encoder);
    let arguments: MatmulArguments<'_, '_, '_, Metal> = MatmulArguments {
        a: MatmulA::FullPrecision {
            values: &buffers.fp_activations,
            offset: 0,
        },
        b: MatmulB::ScaleSymmetricDequant {
            b: &buffers.weights,
            scales: &buffers.weight_scales,
            mode: QuantizationMode::I8,
            group_size: buffers.group_size,
        },
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut buffers.output,
        d_transform: MatmulDOps::none(),
        m: buffers.m,
        n: buffers.n,
        k: buffers.k,
    };
    matmul.gemm.encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder).expect("bf16 GEMM");
}

fn check_correctness(
    context: &MetalContext,
    prepare: &MetalPrepare,
    matmul: &mut MetalMatmul,
) {
    for &(m, k, n, group_size) in CORRECTNESS_SHAPES {
        let host = HostInputs::generate(m, k, n, group_size, 0xA8_00 ^ k as u64);
        let expected = host.reference();
        let mut buffers = upload(context, &host);
        let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
        encode_int8(prepare, matmul, &mut buffers, &mut encoder);
        encoder.end_encoding().submit().wait_until_completed().expect("submit");
        let actual = buffers.output.copyout::<bf16>();

        for (index, (&expected, actual)) in expected.iter().zip(actual).enumerate() {
            let actual = actual.to_f32();
            let tolerance = 0.6f32.max(expected.abs() * 0.06);
            assert!(
                (expected - actual).abs() <= tolerance,
                "A8W8 mismatch at {index}: expected {expected}, got {actual}",
            );
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Int8,
    Bf16,
}

#[uzu_bench]
fn bench_a8w8(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let ops = ActivationPrepareOps::INPUT_RHT | ActivationPrepareOps::QUANTIZE;
    let prepare = <MetalPrepare as ActivationsPrepareKernel>::new(
        &context,
        DataType::BF16,
        ops,
        ActivationScaleStatistic::AbsMax,
    )
    .expect("prepare kernel");
    let hadamard =
        <MetalHadamard as HadamardTransformKernel>::new(&context, DataType::BF16, HadamardTransformOrder::Input)
            .expect("RHT kernel");

    check_correctness(&context, &prepare, &mut matmul);

    let mut group = c.benchmark_group("Metal/Kernel/A8W8");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    for &(m, k, n, group_size) in PERF_SHAPES {
        let label = format!("m{m}_k{k}_n{n}_gs{group_size}");
        let host = HostInputs::generate(m, k, n, group_size, 0xB8_00 ^ k as u64);
        let mut buffers = upload(&context, &host);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        for (name, mode) in [("int8", Mode::Int8), ("bf16", Mode::Bf16)] {
            group.bench_with_input(BenchmarkId::new(name, &label), &mode, |b, &mode| {
                b.iter_custom(|iterations| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iterations {
                        match mode {
                            Mode::Int8 => encode_int8(&prepare, &mut matmul, &mut buffers, &mut encoder),
                            Mode::Bf16 => encode_bf16(&hadamard, &mut matmul, &mut buffers, &mut encoder),
                        }
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });
        }
    }
    group.finish();
}
