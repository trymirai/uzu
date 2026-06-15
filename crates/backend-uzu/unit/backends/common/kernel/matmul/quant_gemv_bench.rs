use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;
use proc_macros::uzu_bench;
use test_runner::for_each_backend;

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, Context,
        gpu_types::QuantizationMethod,
        kernel::{Kernels, matmul::MatmulKernel},
    },
    tests::{
        matmul::{QuantBuffers, QuantInput, bench_quant_gemv_shapes, iter_encode_loop_named, quant_arguments},
        util::type_short_name,
    },
};

fn bench_gemv_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let group_path = format!("{}/Kernel/Gemv/{}", type_short_name::<B>(), label);
    let mut group = c.benchmark_group(group_path.clone());

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
            let benchmark_path = format!("{group_path}/{shape}");
            iter_encode_loop_named::<B, _>(context, b, &benchmark_path, |encoder| {
                let args = quant_arguments(&mut buffers, &input);
                matmul.encode(args, encoder).expect("encode failed");
            });
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
    });
}
