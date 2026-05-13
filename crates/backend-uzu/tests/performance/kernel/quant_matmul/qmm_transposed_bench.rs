use backend_uzu::{
    ArrayElement,
    backends::common::{
        Allocation, Backend, Context, Encoder, Kernels,
        gpu_types::QuantizationMethod,
        kernel::QuantizedMatmulQmmTransposedKernel,
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

fn bench_qmm_transposed_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let block_size: usize = if bits == 4 { 512 } else { 256 };

    for (m, n, k) in iproduct!([4usize, 5, 6, 7, 8, 16, 32, 48, 64], [2048usize, 4096, 14336], [2048usize, 4096, 14336]) {
        if n % 32 != 0 || k % block_size != 0 {
            continue;
        }

        let (bm, bk, bn, wm, wn) = if m < 48 {
            (8u32, 32u32, 32u32, 1u32, 1u32)
        } else {
            (32u32, 32u32, 32u32, 2u32, 2u32)
        };

        let num_groups = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(42);

        let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
            context,
            T::data_type(),
            group_size,
            bits,
            bm,
            bk,
            bn,
            wm,
            wn,
            quant_method,
            false,
            n % bn as usize == 0,
        )
        .unwrap();

        let w_buf = alloc_allocation_with_data::<B, u32>(
            context,
            &(0..n * k * bits as usize / 32).map(|_| rng.random_range(0..u32::MAX)).collect::<Vec<_>>(),
        );
        let scales_buf = alloc_allocation_with_data::<B, T>(
            context,
            &(0..n * num_groups).map(|_| T::from(rng.random_range(0.01f32..1.0f32)).unwrap()).collect::<Vec<_>>(),
        );
        let x_buf = alloc_allocation_with_data::<B, T>(
            context,
            &(0..m * k).map(|_| T::from(rng.random_range(-1.0f32..1.0f32)).unwrap()).collect::<Vec<_>>(),
        );
        let mut y_buf = alloc_allocation::<B, T>(context, m * n);

        let zp_stride = if bits == 4 {
            (num_groups + 1) / 2
        } else {
            num_groups
        };
        let zp_buf = (quant_method == QuantizationMethod::ScaleZeroPoint).then(|| {
            alloc_allocation_with_data::<B, u8>(
                context,
                &(0..n * zp_stride).map(|_| rng.random_range(0u8..u8::MAX)).collect::<Vec<_>>(),
            )
        });
        let bias_buf = (quant_method == QuantizationMethod::ScaleBias).then(|| {
            alloc_allocation_with_data::<B, T>(
                context,
                &(0..n * num_groups).map(|_| T::from(rng.random_range(-0.5f32..0.5f32)).unwrap()).collect::<Vec<_>>(),
            )
        });

        let mut group = c.benchmark_group(format!("{}/Kernel/QmmTransposed/{}", type_short_name::<B>(), label));
        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("M[{m}]N[{n}]K[{k}]")), |b| {
            b.iter_custom(|n_iters| {
                let mut encoder = Encoder::<B>::new(context).unwrap();
                for _ in 0..n_iters {
                    kernel.encode(
                        &w_buf,
                        &scales_buf,
                        zp_buf.as_ref(),
                        bias_buf.as_ref(),
                        &x_buf,
                        &mut y_buf,
                        None::<&Allocation<B>>,
                        k as u32,
                        n as u32,
                        m as u32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            })
        });
        drop(group);
    }
}

#[uzu_bench]
fn bench_qmm_transposed(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_qmm_transposed_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs64", 64, 4, QuantizationMethod::ScaleBias);
        bench_qmm_transposed_typed::<B, bf16>(c, &context, "ZP_BF16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmm_transposed_typed::<B, bf16>(c, &context, "ScaleBias_BF16_gs128", 128, 4, QuantizationMethod::ScaleBias);
        bench_qmm_transposed_typed::<B, bf16>(c, &context, "ZP_BF16_gs128", 128, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmm_transposed_typed::<B, f16>(c, &context, "ZP_F16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
    });
}
