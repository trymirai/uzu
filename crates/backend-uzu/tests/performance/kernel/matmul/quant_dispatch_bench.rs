#![cfg(metal_backend)]

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{ManualKernels, matmul::MatmulKernel},
        },
        metal::{MatmulDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use itertools::iproduct;
use num_traits::Float;

use crate::{
    common::{
        matmul::{QuantBuffers, QuantInput, quant_arguments},
        type_short_name,
    },
    uzu_bench,
};

fn bench_unified_quant_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    // Mirror the shape/filter logic of `qmm_transposed_bench` so the criterion
    // benchmark IDs line up axis-by-axis for `--baseline` comparison.
    let block_size: usize = if bits == 4 { 512 } else { 256 };

    for (m, n, k) in iproduct!([4usize, 5, 6, 7, 8, 16, 32, 48, 64], [2048usize, 4096, 14336], [2048usize, 4096, 14336]) {
        if n % 32 != 0 || k % block_size != 0 {
            continue;
        }

        let input = QuantInput::<T>::random(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers = QuantBuffers::<Metal, T>::allocate(context, &input);
        let mut matmul =
            <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type()).unwrap();

        let mut group =
            c.benchmark_group(format!("{}/Kernel/UnifiedQuantizedGemm/{}", type_short_name::<Metal>(), label));
        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("M[{m}]N[{n}]K[{k}]")), |b| {
            b.iter_custom(|n_iters| {
                let mut encoder = Encoder::<Metal>::new(context).unwrap();
                for _ in 0..n_iters {
                    matmul
                        .encode_with_path(
                            quant_arguments(&mut buffers, &input),
                            &mut encoder,
                            MatmulDispatchPath::QuantGemm,
                        )
                        .expect("encode unified quant matmul");
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            })
        });
        drop(group);
    }
}

#[uzu_bench]
fn bench_unified_quantized_gemm(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    bench_unified_quant_typed::<bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
    bench_unified_quant_typed::<bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
    bench_unified_quant_typed::<bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
    bench_unified_quant_typed::<bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
    bench_unified_quant_typed::<f16>(c, &context, "ZP_F16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
}
