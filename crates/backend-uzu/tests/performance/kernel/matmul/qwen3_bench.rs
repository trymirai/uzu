#![cfg(metal_backend)]

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context,
            gpu_types::QuantizationMethod,
            kernel::{ManualKernels, matmul::MatmulKernel},
        },
        metal::{MatmulDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;

use crate::{
    common::{
        matmul::{QuantBuffers, QuantInput, iter_encode_loop, quant_arguments, qwen3_layer_shapes},
        type_short_name,
    },
    uzu_bench,
};

fn bench_qwen3_layers_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group =
        c.benchmark_group(format!("{}/Kernel/Qwen3Layers/{}", type_short_name::<Metal>(), label));
    for (layer, shape) in qwen3_layer_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<Metal, T>::allocate(context, &input);
        let mut matmul =
            <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type()).unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("{layer}_{shape}")), |b| {
            iter_encode_loop::<Metal, _>(context, b, |encoder| {
                matmul
                    .encode_dispatch_path(
                        quant_arguments(&mut buffers, &input),
                        encoder,
                        MatmulDispatchPath::Auto,
                    )
                    .expect("encode qwen3 layer");
            });
        });
    }
}

#[uzu_bench]
fn bench_qwen3_layers(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    bench_qwen3_layers_typed::<bf16>(c, &context, "ScaleBias_BF16_gs128_4b", 128, 4, QuantizationMethod::ScaleBias);
    bench_qwen3_layers_typed::<bf16>(c, &context, "ZP_BF16_gs128_4b", 128, 4, QuantizationMethod::ScaleZeroPoint);
}
