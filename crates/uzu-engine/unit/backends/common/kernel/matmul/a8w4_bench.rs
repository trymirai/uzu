// Only Metal reports the GPU execution time the benchmark loop measures.
#![cfg(backend = "metal")]

use criterion::{BenchmarkId, Criterion, Throughput};
use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend, Context,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{ActivationFormat, MatmulKernel},
            },
        },
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        matmul::{QuantInput, Shape, iter_encode_loop_named, production_quant_input, quant_arguments},
        util::type_short_name,
    },
};

#[uzu_bench]
fn bench_a8w4(c: &mut Criterion) {
    let context = block_on(<Metal as Backend>::Context::new()).unwrap();
    let new_matmul = || {
        <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        )
        .unwrap()
    };
    let (layout, mut matmul) = (new_matmul(), new_matmul());
    let group_path = format!("{}/Kernel/A8W4", type_short_name::<Metal>());
    let mut group = c.benchmark_group(group_path.clone());
    let target = [
        ("27b_qkvg", 5120, 14336),
        ("27b_o", 6144, 5120),
        ("27b_up", 5120, 34816),
        ("27b_down", 17408, 5120),
        ("27b_in", 5120, 16480),
    ];
    let draft =
        [("draft_qkvg", 5120, 6144), ("draft_o", 4096, 5120), ("draft_down", 17408, 5120), ("draft_up", 5120, 34816)];
    let shapes = target
        .into_iter()
        .flat_map(|(layer, k, n)| [8, 16, 32, 64].map(|m| (layer, Shape::new(m, k, n), 64)))
        .chain(draft.map(|(layer, k, n)| (layer, Shape::new(12, k, n), 32)));
    for (layer, shape, group_size) in shapes {
        let input =
            QuantInput::<bf16>::new(shape.m, shape.k, shape.n, group_size, 4, QuantizationMethod::ScaleZeroPoint, 42)
                .with_group_output();
        let (prepared, ActivationFormat::Int8(_), _) =
            production_quant_input::<Metal, bf16>(&context, &layout, input.clone())
        else {
            continue;
        };
        let mut buffers = ColdPool::new(prepared.weight_buffer_bytes(), || {
            production_quant_input::<Metal, bf16>(&context, &layout, input.clone()).2
        });
        group.throughput(Throughput::Bytes(prepared.weight_buffer_bytes() as u64));
        group.bench_function(BenchmarkId::new(layer, shape.to_string()), |b| {
            let benchmark_path = format!("{group_path}/{layer}/{shape}");
            iter_encode_loop_named::<Metal, _>(&context, b, &benchmark_path, |command_buffer| {
                matmul.encode(quant_arguments(buffers.next_mut(), &prepared), command_buffer).expect("encode");
            });
        });
    }
    group.finish();
}
