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
        metal::{Metal, MetalContext},
    },
    common::{
        cold_pool::ColdPool,
        matmul::{QuantBuffers, QuantInput, iter_encode_loop, quant_arguments, qwen3_layer_shapes},
        type_short_name,
    },
};

fn bench_qwen3_layers_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Qwen3Layers/{}", type_short_name::<Metal>(), label));
    for (layer, shape) in qwen3_layer_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers =
            ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, T>::allocate(context, &input));
        let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("{layer}_{shape}")), |b| {
            iter_encode_loop::<Metal, _>(context, b, |encoder| {
                matmul.encode(quant_arguments(buffers.next_mut(), &input), encoder).expect("encode qwen3 layer");
            });
        });
    }
}

#[uzu_bench]
fn bench_qwen3_layers(c: &mut Criterion) {
    let context = crate::common::shared_metal_context();
    bench_qwen3_layers_typed::<bf16>(c, &context, "ScaleBias_BF16_gs128_4b", 128, 4, QuantizationMethod::ScaleBias);
    bench_qwen3_layers_typed::<bf16>(c, &context, "ZP_BF16_gs128_4b", 128, 4, QuantizationMethod::ScaleZeroPoint);
}
