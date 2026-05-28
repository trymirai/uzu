use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::common::{Allocation, Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::GatedActMulKernel},
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use itertools::iproduct;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{common::type_short_name, uzu_bench};

fn random_buffer<B: Backend>(context: &B::Context, length: usize) -> Allocation<B> {
    let mut rng = SmallRng::seed_from_u64(1337);
    let mut data = vec![bf16::ZERO; length];
    for value in data.iter_mut() {
        *value = bf16::from_f32(rng.random_range(-2.0f32..2.0f32));
    }
    context.create_array_from(&[length], data.as_slice()).into_allocation()
}

// Benches both GatedActMul layouts: interleaved (MLP, gate+up halves of one buffer)
// at MLP intermediate/token shapes, and separate (PLE, dense gate * strided per-layer
// input) at per-layer-embedding shapes.
#[uzu_bench]
fn bench_gated_act_mul(c: &mut Criterion) {
    type T = bf16;
    let act_type = ActivationType::GELUApprox;

    for_each_non_cpu_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        let interleaved_kernel =
            <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(&context, T::data_type(), true, false).unwrap();
        let separate_kernel =
            <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(&context, T::data_type(), false, false)
                .unwrap();

        let mut group = c.benchmark_group(format!("{}/Kernel/GatedActMul", type_short_name::<B>()));

        for (m, h) in iproduct!([1usize, 128, 512], [8192usize, 16384]) {
            let fused_buffer = random_buffer::<B>(&context, m * 2 * h);
            let mut output_buffer = context.create_array_uninitialized(&[m * h], T::data_type()).into_allocation();

            group.throughput(Throughput::Elements((m * h) as u64));
            group.bench_function(BenchmarkId::new("Interleaved", format!("m{m}_h{h}")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();
                    for _ in 0..n_iters {
                        interleaved_kernel.encode(
                            &fused_buffer,
                            None::<&Allocation<B>>,
                            &mut output_buffer,
                            None::<&Allocation<B>>,
                            h as i32,
                            m as i32,
                            0,
                            0,
                            act_type,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });
        }

        let ple_dim = 256usize;
        let num_layers = 32usize;
        for m in [1usize, 128, 512] {
            let gate_buffer = random_buffer::<B>(&context, m * ple_dim);
            let value_buffer = random_buffer::<B>(&context, m * num_layers * ple_dim);
            let mut output_buffer = context.create_array_uninitialized(&[m * ple_dim], T::data_type()).into_allocation();

            group.throughput(Throughput::Elements((m * ple_dim) as u64));
            group.bench_function(BenchmarkId::new("Separate", format!("m{m}_ple{ple_dim}")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();
                    for _ in 0..n_iters {
                        separate_kernel.encode(
                            &gate_buffer,
                            Some(&value_buffer),
                            &mut output_buffer,
                            None::<&Allocation<B>>,
                            ple_dim as i32,
                            m as i32,
                            0,
                            (num_layers * ple_dim) as i32,
                            act_type,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });
        }
    });
}
