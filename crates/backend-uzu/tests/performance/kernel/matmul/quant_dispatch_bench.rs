#![cfg(metal_backend)]

use std::collections::HashSet;

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulKernel},
            },
        },
        metal::{MatmulDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use itertools::iproduct;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    common::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
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
    let mode = match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::I8,
        _ => panic!("Unsupported bits: {bits}"),
    };

    // Mirror the shape/filter logic of `qmm_transposed_bench` so the criterion
    // benchmark IDs line up axis-by-axis for `--baseline` comparison.
    let block_size: usize = if bits == 4 { 512 } else { 256 };

    for (m, n, k) in iproduct!([4usize, 5, 6, 7, 8, 16, 32, 48, 64], [2048usize, 4096, 14336], [2048usize, 4096, 14336]) {
        if n % 32 != 0 || k % block_size != 0 {
            continue;
        }

        let num_groups = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(42);

        let mut matmul =
            <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type()).unwrap();

        let w_buf = alloc_allocation_with_data::<Metal, u32>(
            context,
            &(0..n * k * bits as usize / 32).map(|_| rng.random_range(0..u32::MAX)).collect::<Vec<_>>(),
        );
        let scales_buf = alloc_allocation_with_data::<Metal, T>(
            context,
            &(0..n * num_groups).map(|_| T::from(rng.random_range(0.01f32..1.0f32)).unwrap()).collect::<Vec<_>>(),
        );
        let x_buf = alloc_allocation_with_data::<Metal, T>(
            context,
            &(0..m * k).map(|_| T::from(rng.random_range(-1.0f32..1.0f32)).unwrap()).collect::<Vec<_>>(),
        );
        let mut y_buf = alloc_allocation::<Metal, T>(context, m * n);

        let zp_stride = if bits == 4 { num_groups.div_ceil(2) } else { num_groups };
        let zp_or_bias = match quant_method {
            QuantizationMethod::ScaleZeroPoint => alloc_allocation_with_data::<Metal, u8>(
                context,
                &(0..n * zp_stride).map(|_| rng.random_range(0u8..u8::MAX)).collect::<Vec<_>>(),
            ),
            QuantizationMethod::ScaleBias => alloc_allocation_with_data::<Metal, u8>(
                context,
                // Reinterpret as raw bytes; matches the path used by the kernel
                // for `biases` (T elements), but the bench only measures kernel
                // throughput so the underlying byte pattern doesn't affect
                // timing.
                &vec![0u8; n * num_groups * std::mem::size_of::<T>()],
            ),
        };

        let mut group = c.benchmark_group(format!("{}/Kernel/UnifiedQuantizedGemm/{}", type_short_name::<Metal>(), label));
        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("M[{m}]N[{n}]K[{k}]")), |b| {
            b.iter_custom(|n_iters| {
                let mut encoder = Encoder::<Metal>::new(context).unwrap();
                for _ in 0..n_iters {
                    let b_variant = match quant_method {
                        QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                            b: &w_buf,
                            scales: &scales_buf,
                            zero_points: &zp_or_bias,
                            mode,
                            group_size,
                        },
                        QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
                            b: &w_buf,
                            scales: &scales_buf,
                            biases: &zp_or_bias,
                            mode,
                            group_size,
                        },
                    };
                    matmul
                        .encode_with_path(
                            MatmulArguments {
                                a: &x_buf,
                                a_offset: 0,
                                b: b_variant,
                                b_offset: 0,
                                b_leading_dimension: None,
                                b_transpose: true,
                                d: &mut y_buf,
                                d_transform: HashSet::new(),
                                m: m as u32,
                                n: n as u32,
                                k: k as u32,
                            },
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
