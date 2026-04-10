use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use itertools::iproduct;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu::{
    ArrayElement,
    backends::common::{
        Backend, Context, Encoder,
        gpu_types::QuantizationMode,
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
            QuantizedMatmulType,
        },
    },
};

use crate::{
    common::{helpers::alloc_buffer_with_data, type_short_name},
    uzu_bench,
};

fn gen_random<T: rand::distr::uniform::SampleUniform + PartialOrd + Copy, R: rand::Rng>(
    rng: &mut R,
    range: std::ops::Range<T>,
    len: usize,
) -> Box<[T]> {
    (0..len).map(|_| rng.random_range(range.clone())).collect()
}

fn bench_qmm_small_typed<B: Backend, T: ArrayElement>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    use_zero_points: bool,
    use_mlx_quant: bool,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/QmmSmall/{}", type_short_name::<B>(), label));
    let block_size: usize = if bits == 4 { 512 } else { 256 };

    let mode = if bits == 4 { QuantizationMode::UINT4 } else { QuantizationMode::UINT8 };
    let quantization_type = if use_mlx_quant { QuantizedMatmulType::Mlx } else { QuantizedMatmulType::ZeroPoint };

    for (m, n, k) in iproduct!(1..=64usize, [2048usize, 4096, 14336], [2048usize, 4096, 8192, 14336]) {
        if n % 8 != 0 || k % block_size != 0 {
            continue;
        }

        let num_groups = k.div_ceil(group_size as usize);
        let mut rng = SmallRng::seed_from_u64(42);

        let configuration = QuantizedMatmulConfiguration {
            data_type: T::data_type(),
            group_size: group_size as usize,
            input_dim: k,
            output_dim: n,
            mode,
            quantization_type,
            use_hadamard: false,
        };
        let dispatcher = QuantizedMatmulKernelEncodable::<B>::new(context, configuration).unwrap();

        let w_buf = alloc_buffer_with_data::<B, u32>(
            context,
            &gen_random::<u32, _>(&mut rng, 0..u32::MAX, n * k * bits as usize / 32),
        );
        let scales_buf = alloc_buffer_with_data::<B, T>(
            context,
            &gen_random::<f32, _>(&mut rng, 0.01..1.0, n * num_groups)
                .iter()
                .map(|&v| T::from(v).unwrap())
                .collect::<Vec<_>>(),
        );
        let x_buf = alloc_buffer_with_data::<B, T>(
            context,
            &gen_random::<f32, _>(&mut rng, -1.0..1.0, m * k)
                .iter()
                .map(|&v| T::from(v).unwrap())
                .collect::<Vec<_>>(),
        );
        let mut y_buf = context.create_buffer(m * n * std::mem::size_of::<T>()).unwrap();

        let zp_or_bias_buf = if use_zero_points {
            let zp_stride = if bits == 4 { (num_groups + 1) / 2 } else { num_groups };
            alloc_buffer_with_data::<B, u8>(context, &gen_random::<u8, _>(&mut rng, 0..u8::MAX, n * zp_stride))
        } else {
            alloc_buffer_with_data::<B, T>(
                context,
                &gen_random::<f32, _>(&mut rng, -0.5..0.5, n * num_groups)
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect::<Vec<_>>(),
            )
        };

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("M[{m}]N[{n}]K[{k}]")), |b| {
            b.iter_custom(|n_iters| {
                let mut encoder = Encoder::<B>::new(context).unwrap();
                for _ in 0..n_iters {
                    let args = QuantizedMatmulArguments {
                        a_buffer: &x_buf,
                        a_offset: 0,
                        b_buffer: &w_buf,
                        scales_buffer: &scales_buf,
                        zero_points_or_biases_buffer: &zp_or_bias_buf,
                        output_buffer: &mut y_buf,
                        hadamard_factors: None,
                        batch_dim: m,
                    };
                    dispatcher.encode(&mut encoder, args).unwrap();
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            })
        });
    }
}

#[uzu_bench]
fn bench_qmm_small(c: &mut Criterion) {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        bench_qmm_small_typed::<B, bf16>(c, &context, "Mlx_BF16_gs128", 128, 4, false, true);
        bench_qmm_small_typed::<B, f16>(c, &context, "ZP_F16_gs64", 64, 4, true, false);
    });
}
