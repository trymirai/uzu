use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::Metal,
    },
    data_type::DataTypeRepr,
    tests::{
        cold_pool::ColdPool,
        matmul::{QuantInput, iter_encode_loop, production_quant_input, quant_arguments},
        util::type_short_name,
    },
};

/// Inkling 2 Mini attention projections at decode, MLX affine 4-bit with groups of 64: (label, k, n).
const PROJECTIONS: &[(&str, u32, u32)] = &[
    ("query", 2304, 6552),
    ("query_padded", 2304, 6560),
    ("key_value", 2304, 1024),
    ("out", 3072, 2304),
];

/// One decode GEMV per projection with production weight layouts; `thrpt` is the weight bandwidth.
#[uzu_bench]
fn bench_inkling_projections(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/InklingProjections", type_short_name::<Metal>()));
    for &(label, k, n) in PROJECTIONS {
        let new_matmul = || {
            <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                bf16::data_type(),
                bf16::data_type(),
                bf16::data_type(),
            )
            .unwrap()
        };
        let new_input = || QuantInput::<bf16>::new(1, k, n, 64, 4, QuantizationMethod::ScaleBias, 42);
        let preparer = new_matmul();
        let mut matmul = new_matmul();
        let (input, format, first) = production_quant_input::<Metal, bf16>(&context, &preparer, new_input());
        eprintln!("{label} {k}x{n}: layout {:?}, activations {:?}", first.weights_layout, format);
        let mut buffers = ColdPool::new(input.weight_buffer_bytes(), || {
            production_quant_input::<Metal, bf16>(&context, &preparer, new_input()).2
        });
        group.throughput(Throughput::Bytes(u64::from(n) * u64::from(k) * 9 / 16));
        group.bench_function(BenchmarkId::from_parameter(format!("{label}_{k}x{n}")), |b| {
            iter_encode_loop::<Metal, _>(&context, b, |command_buffer| {
                matmul.encode(quant_arguments(buffers.next_mut(), &input), command_buffer).expect("encode projection");
            });
        });
    }
    group.finish();
}
