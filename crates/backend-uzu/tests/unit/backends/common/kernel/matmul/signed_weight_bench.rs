#![cfg(backend = "metal")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::gpu_types::QuantizationMethod,
        metal::{GemvDispatch, GemvSpecialization, Metal},
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        matmul::{QuantBuffers, QuantInput, iter_encode_loop_named, quant_arguments_full_precision_a},
        util::shared_metal_context,
    },
};

#[uzu_bench]
fn bench_signed_weight_gemv(c: &mut Criterion) {
    let context = shared_metal_context();
    let device_tier = context.device_tier();
    let (m, k, n, group_size) = (1usize, 4096usize, 4096usize, 64u32);

    for bits in [4u32, 8u32] {
        let group_path = format!("Metal/Kernel/SignedWeightGemv/w{bits}");
        let mut group = c.benchmark_group(&group_path);
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_millis(800));
        group.throughput(Throughput::Elements((m * k * n) as u64));

        for signed_codes in [false, true] {
            let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleZeroPoint, 42);
            let input = if signed_codes {
                input.with_signed_weight_codes()
            } else {
                input
            };
            let mut buffers =
                ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, bf16>::allocate(&context, &input));
            let mut gemv = GemvDispatch::new(DataType::BF16, DataType::BF16, DataType::BF16);
            let specialization = {
                let args = quant_arguments_full_precision_a(buffers.next_mut(), &input);
                GemvSpecialization::select(&args, DataType::BF16, DataType::BF16, DataType::BF16, device_tier)
                    .expect("signed-weight GEMV specialization")
            };
            let label = if signed_codes {
                "signed_codes"
            } else {
                "unsigned_codes"
            };
            let benchmark_path = format!("{group_path}/{label}");

            group.bench_function(BenchmarkId::from_parameter(label), |bench| {
                iter_encode_loop_named::<Metal, _>(&context, bench, &benchmark_path, |encoder| {
                    let args = quant_arguments_full_precision_a(buffers.next_mut(), &input);
                    gemv.encode(args, specialization, encoder).expect("signed-weight GEMV encode");
                });
            });
        }
        group.finish();
    }
}
