use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, QuantizedMatmulQmvLloydMaxKernel, matmul::MatmulKernel},
        },
        metal::Metal,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;

use crate::{
    common::{
        matmul::{
            LloydMaxQuantBuffers, LloydMaxQuantInput, QuantBuffers, QuantInput, Shape, bench_quant_gemv_shapes,
            iter_encode_loop, lloyd_max_quant_arguments, quant_arguments,
        },
        type_short_name,
    },
    uzu_bench,
};

fn bench_gemv_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Gemv/{}", type_short_name::<B>(), label));

    for shape in bench_quant_gemv_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<B, T>::allocate(context, &input);
        let mut matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                let args = quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("encode failed");
            });
        });
    }
}

fn bench_lloyd_max_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Gemv/{}", type_short_name::<B>(), label));

    for shape in bench_quant_gemv_shapes(4) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = LloydMaxQuantInput::<T>::new(m, k, n, group_size);
        let mut buffers = LloydMaxQuantBuffers::<B, T>::allocate(context, &input);
        let mut matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                let args = lloyd_max_quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("encode failed");
            });
        });
    }
}

fn bench_lloyd_max_direct_metal<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &<Metal as Backend>::Context,
    label: &str,
    group_size: u32,
    fused: bool,
) {
    let mut group = c.benchmark_group(format!("Metal/Kernel/Gemv/{label}"));
    let shapes = [1usize, 2, 3, 4].map(|m| Shape::new(m, 4096, 4096));

    for shape in shapes {
        let input = LloydMaxQuantInput::<T>::new(shape.m, shape.k, shape.n, group_size);
        let mut buffers = LloydMaxQuantBuffers::<Metal, T>::allocate(context, &input);
        let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .expect("MatmulMetalKernel");

        group.throughput(Throughput::Elements((shape.m * shape.n * shape.k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            if fused {
                iter_encode_loop::<Metal, _>(context, b, |encoder| {
                    let arguments = lloyd_max_quant_arguments(&mut buffers, &input);
                    matmul.encode(arguments, encoder).expect("encode production Lloyd-Max path");
                });
            } else {
                let kernel =
                    <<<Metal as Backend>::Kernels as Kernels>::QuantizedMatmulQmvLloydMaxKernel as QuantizedMatmulQmvLloydMaxKernel>::new(
                        context,
                        T::data_type(),
                        group_size,
                        4,
                    )
                    .expect("QuantizedMatmulQmvLloydMaxMetalKernel");
                iter_encode_loop::<Metal, _>(context, b, |encoder| {
                    kernel.encode(
                        &buffers.w,
                        &buffers.scales,
                        &buffers.codebook,
                        &buffers.bias_indices,
                        &buffers.bias_codebook,
                        &buffers.x,
                        &mut buffers.y,
                        input.k,
                        input.n,
                        input.m,
                        encoder,
                    );
                });
            }
        });
    }
}

#[uzu_bench]
fn bench_gemv(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs32", 32, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs32", 32, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
        bench_gemv_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
        bench_lloyd_max_typed::<B, bf16>(c, &context, "LloydMax_BF16_gs64", 64);
    });

    let context = <Metal as Backend>::Context::new().unwrap();
    bench_lloyd_max_direct_metal::<bf16>(c, &context, "LloydMaxIndependent_BF16_gs64", 64, false);
    bench_lloyd_max_direct_metal::<bf16>(c, &context, "LloydMaxFused_BF16_gs64", 64, true);
}
