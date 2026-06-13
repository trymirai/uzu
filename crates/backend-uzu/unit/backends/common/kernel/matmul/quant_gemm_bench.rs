#![cfg(metal_backend)]

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
    tests::{
        matmul::{QuantBuffers, QuantInput, bench_quant_gemm_shapes, iter_encode_loop, quant_arguments},
        util::type_short_name,
    },
};

fn bench_unified_quant_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let supports_mxu = context.supports_mxu();
    let paths: &[(&str, GemmDispatchPath)] = if supports_mxu {
        &[("Simdgroup", GemmDispatchPath::Simdgroup), ("Mxu", GemmDispatchPath::Mxu)]
    } else {
        &[("Simdgroup", GemmDispatchPath::Simdgroup)]
    };

    for shape in bench_quant_gemm_shapes(bits) {
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

        for &(path_label, path) in paths {
            let mut group = c.benchmark_group(format!(
                "{}/Kernel/UnifiedQuantizedGemm/{}/{}",
                type_short_name::<Metal>(),
                path_label,
                label
            ));
            group.throughput(Throughput::Elements((m * n * k) as u64));
            group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
                iter_encode_loop::<Metal, _>(context, b, |encoder| {
                    matmul
                        .gemm
                        .encode_dispatch_path(quant_arguments(&mut buffers, &input), path, encoder)
                        .expect("encode unified quant matmul");
                });
            });
            drop(group);
        }
    }
}

#[uzu_bench]
fn bench_unified_quantized_gemm(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    bench_unified_quant_typed::<bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
    bench_unified_quant_typed::<bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
    bench_unified_quant_typed::<bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
    bench_unified_quant_typed::<bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
}
