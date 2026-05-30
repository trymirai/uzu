use backend_uzu::{
    ArrayElement,
    backends::common::{
        Allocation, Backend, Context, Kernels, gpu_types::QuantizationMethod, kernel::QuantizedMatmulQmvFastKernel,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;

use crate::{
    common::{
        matmul::{QuantBuffers, QuantInput, bench_quant_gemv_shapes, iter_encode_loop},
        type_short_name,
    },
    uzu_bench,
};

fn bench_qmv_fast_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmvFast/{}", type_short_name::<B>(), label));

    for shape in bench_quant_gemv_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<B, T>::allocate(context, &input);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
            group_size,
            bits,
            quant_method,
            false,
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<B, _>(context, b, |encoder| {
                kernel.encode(
                    &buffers.w,
                    &buffers.scales,
                    buffers.zp.as_ref(),
                    buffers.bias.as_ref(),
                    &buffers.x,
                    &mut buffers.y,
                    None::<&Allocation<B>>,
                    k as u32,
                    n as u32,
                    m as u32,
                    encoder,
                );
            });
        });
    }
}

#[uzu_bench]
fn bench_qmv_fast(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs32", 32, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs32", 32, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
    });
}
