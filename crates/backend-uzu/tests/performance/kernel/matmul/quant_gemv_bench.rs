#![cfg(metal_backend)]

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;

use crate::{
    common::{
        matmul::{QuantBuffers, QuantInput, bench_quant_gemv_shapes, iter_encode_loop, quant_arguments},
        type_short_name,
    },
    uzu_bench,
};

fn bench_quant_gemv_typed<T: ArrayElement + Float>(
    context: &MetalContext,
    c: &mut Criterion,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/QuantGemv/{}", type_short_name::<Metal>(), label));

    for shape in bench_quant_gemv_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<Metal, T>::allocate(context, &input);
        let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<Metal, _>(context, b, |encoder| {
                matmul.quant_gemv.encode(encoder, quant_arguments(&mut buffers, &input)).expect("encode quant gemv");
            });
        });
    }
}

#[uzu_bench]
fn bench_quant_gemv(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    bench_quant_gemv_typed::<bf16>(&context, c, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
    bench_quant_gemv_typed::<bf16>(&context, c, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
    bench_quant_gemv_typed::<bf16>(&context, c, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
    bench_quant_gemv_typed::<bf16>(&context, c, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
}
