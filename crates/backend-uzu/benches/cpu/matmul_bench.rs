use std::time::Duration;

use backend_uzu::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::QuantizationMethod,
            kernel::matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
        cpu::Cpu,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::common::matmul::{
    Case, QuantBuffers, QuantInput, Shape, deterministic_input, iter_encode_loop_named, quant_arguments,
};

const FULL_PRECISION_SHAPES: &[Shape] =
    &[Shape::new(1, 1024, 1024), Shape::new(4, 2048, 2048), Shape::new(16, 2048, 2048)];

const QUANTIZED_SHAPES: &[Shape] = &[Shape::new(1, 2048, 2048), Shape::new(4, 2048, 4096), Shape::new(4, 4096, 4096)];

fn configure_cpu_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

#[uzu_bench]
fn bench_cpu_full_precision_matmul(c: &mut Criterion) {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("CPU MatmulKernel");

    let group_path = "Cpu/Kernel/Matmul/FullPrecision";
    let mut group = c.benchmark_group(group_path);
    configure_cpu_group(&mut group);

    for shape in FULL_PRECISION_SHAPES {
        let Shape {
            m,
            k,
            n,
        } = *shape;
        let input = deterministic_input::<bf16>(Case::new(*shape));
        let a = context.create_array_from(&[m, k], &input.a).into_allocation();
        let b_matrix = context.create_array_from(&[n, k], &input.b);
        let mut d = context.create_array_uninitialized(&[m, n], bf16::data_type()).into_allocation();

        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_function(BenchmarkId::new("BF16", shape.to_string()), |b| {
            let benchmark_path = format!("{group_path}/{shape}");
            iter_encode_loop_named::<Cpu, _>(&context, b, &benchmark_path, |encoder| {
                matmul
                    .encode(
                        MatmulArguments {
                            a: &a,
                            a_offset: 0,
                            b: MatmulB::FullPrecision {
                                b: b_matrix.allocation(),
                            },
                            b_offset: 0,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut d,
                            d_transform: MatmulDOps::none(),
                            m: m as u32,
                            n: n as u32,
                            k: k as u32,
                        },
                        encoder,
                    )
                    .expect("CPU full precision matmul encode failed");
            });
        });
    }

    group.finish();
}

fn bench_cpu_quantized_matmul_variant(
    c: &mut Criterion,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("CPU MatmulKernel");

    let group_path = format!("Cpu/Kernel/Matmul/Quantized/{label}");
    let mut group = c.benchmark_group(group_path.clone());
    configure_cpu_group(&mut group);

    for shape in QUANTIZED_SHAPES {
        let Shape {
            m,
            k,
            n,
        } = *shape;
        let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<Cpu, bf16>::allocate(&context, &input);

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            let benchmark_path = format!("{group_path}/{shape}");
            iter_encode_loop_named::<Cpu, _>(&context, b, &benchmark_path, |encoder: &mut Encoder<Cpu>| {
                let args = quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("CPU quantized matmul encode failed");
            });
        });
    }

    group.finish();
}

#[uzu_bench]
fn bench_cpu_quantized_matmul(c: &mut Criterion) {
    bench_cpu_quantized_matmul_variant(c, "ScaleBias_BF16_gs64_4b", 64, 4, QuantizationMethod::ScaleBias);
    bench_cpu_quantized_matmul_variant(c, "ZP_BF16_gs64_4b", 64, 4, QuantizationMethod::ScaleZeroPoint);
    bench_cpu_quantized_matmul_variant(c, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
}
