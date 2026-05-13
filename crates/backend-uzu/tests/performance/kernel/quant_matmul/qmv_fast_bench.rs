use backend_uzu::{
    ArrayElement,
    backends::common::{
        Allocation, Backend, Context, Encoder, Kernels,
        gpu_types::QuantizationMethod,
        kernel::QuantizedMatmulQmvFastKernel,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    common::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        type_short_name,
    },
    uzu_bench,
};

fn gen_random<T: rand::distr::uniform::SampleUniform + PartialOrd + Copy, R: rand::Rng>(
    rng: &mut R,
    range: std::ops::Range<T>,
    len: usize,
) -> Box<[T]> {
    (0..len).map(|_| rng.random_range(range.clone())).collect()
}

fn bench_qmv_fast_typed<B: Backend, T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmvFast/{}", type_short_name::<B>(), label));
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };

    let nk_pairs: &[(usize, usize)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    for &(n, k) in nk_pairs {
        if n % 8 != 0 || k % block_size != 0 {
            continue;
        }
        for m in [1usize, 2, 4, 8] {
            let num_groups = k.div_ceil(group_size as usize);
            let mut rng = SmallRng::seed_from_u64(42);

            let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                context,
                T::data_type(),
                group_size,
                bits,
                quant_method,
                false,
            )
            .unwrap();

            let w_buf = alloc_allocation_with_data::<B, u32>(
                context,
                &gen_random::<u32, _>(&mut rng, 0..u32::MAX, n * k * bits as usize / 32),
            );
            let scales_buf = alloc_allocation_with_data::<B, T>(
                context,
                &gen_random::<f32, _>(&mut rng, 0.01..1.0, n * num_groups)
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect::<Vec<_>>(),
            );
            let x_buf = alloc_allocation_with_data::<B, T>(
                context,
                &gen_random::<f32, _>(&mut rng, -1.0..1.0, m * k)
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect::<Vec<_>>(),
            );
            let mut y_buf = alloc_allocation::<B, T>(context, m * n);

            let zp_buf = (quant_method == QuantizationMethod::ScaleZeroPoint).then(|| {
                let zp_stride = if bits == 4 {
                    (num_groups + 1) / 2
                } else {
                    num_groups
                };
                alloc_allocation_with_data::<B, u8>(context, &gen_random::<u8, _>(&mut rng, 0..u8::MAX, n * zp_stride))
            });
            let bias_buf = (quant_method == QuantizationMethod::ScaleBias).then(|| {
                alloc_allocation_with_data::<B, T>(
                    context,
                    &gen_random::<f32, _>(&mut rng, -0.5..0.5, n * num_groups)
                        .iter()
                        .map(|&v| T::from(v).unwrap())
                        .collect::<Vec<_>>(),
                )
            });

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
        }
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
        bench_qmv_fast_typed::<B, f16>(c, &context, "ZP_F16_gs64", 64, 4, QuantizationMethod::ScaleZeroPoint);
        bench_qmv_fast_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_8b", 64, 8, QuantizationMethod::ScaleZeroPoint);
    });
}
