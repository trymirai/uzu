#[cfg(metal_backend)]
use std::ops::{Deref, DerefMut};

#[cfg(metal_backend)]
use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{
                PerLayerEmbeddingCombineKernel, SoftCapKernel, TensorFinalizeKernel, TensorMulSliceKernel,
                ValueNormKernel,
            },
        },
        metal::{Metal, MetalContext},
    },
};
#[cfg(metal_backend)]
use criterion::{BenchmarkId, Criterion, Throughput};
#[cfg(metal_backend)]
use half::bf16;

#[cfg(metal_backend)]
use crate::{common::type_short_name, uzu_bench};

#[cfg(metal_backend)]
#[uzu_bench]
fn bench_model_extension_kernels(criterion: &mut Criterion) {
    let context = MetalContext::new().expect("Failed to create Metal context");
    let mut group = criterion.benchmark_group(format!("{}/Kernel/ModelExtension", type_short_name::<Metal>()));

    bench_value_norm(&context, &mut group);
    bench_per_layer_embedding_combine(&context, &mut group);
    bench_tensor_mul_slice(&context, &mut group);
    bench_tensor_finalize(&context, &mut group);
    bench_soft_cap(&context, &mut group);
    group.finish();
}

#[cfg(metal_backend)]
fn bench_value_norm(
    context: &MetalContext,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let batch_size = 128usize;
    let num_heads = 8usize;
    let num_groups = 1usize;
    let head_dim = 512usize;
    let row_stride = (num_heads + 2 * num_groups) * head_dim;
    let qkv = context.create_array_uninitialized(&[batch_size, row_stride], bf16::data_type(), "value_norm_qkv");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::ValueNormKernel::new(context, bf16::data_type())
        .expect("Failed to create ValueNormKernel");

    group.throughput(Throughput::Elements((batch_size * num_groups * head_dim) as u64));
    group.bench_function(BenchmarkId::new("ValueNorm", format!("Batch[{batch_size}]HeadDim[{head_dim}]")), |bencher| {
        let qkv_buffer = qkv.buffer();
        bencher.iter_custom(|iteration_count| {
            let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
            for _ in 0..iteration_count {
                kernel.encode(
                    qkv_buffer.borrow_mut().deref_mut(),
                    batch_size as u32,
                    num_heads as u32,
                    num_groups as u32,
                    head_dim as u32,
                    1e-6,
                    &mut encoder,
                );
            }
            encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
        });
    });
}

#[cfg(metal_backend)]
fn bench_per_layer_embedding_combine(
    context: &MetalContext,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let batch_size = 128usize;
    let num_layers = 35usize;
    let ple_dim = 256usize;
    let total_ple_dim = num_layers * ple_dim;
    let token_ple = context.create_array_uninitialized(&[batch_size, total_ple_dim], bf16::data_type(), "ple_token");
    let model_ple = context.create_array_uninitialized(&[batch_size, total_ple_dim], bf16::data_type(), "ple_model");
    let scales = context.create_array_uninitialized(&[ple_dim], bf16::data_type(), "ple_scales");
    let combined = context.create_array_uninitialized(&[batch_size, total_ple_dim], bf16::data_type(), "ple_combined");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::PerLayerEmbeddingCombineKernel::new(
        context,
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("Failed to create PerLayerEmbeddingCombineKernel");

    group.throughput(Throughput::Elements((batch_size * total_ple_dim) as u64));
    group.bench_function(
        BenchmarkId::new("PerLayerEmbeddingCombine", format!("Batch[{batch_size}]Dim[{total_ple_dim}]")),
        |bencher| {
            let token_ple_buffer = token_ple.buffer();
            let model_ple_buffer = model_ple.buffer();
            let scales_buffer = scales.buffer();
            let combined_buffer = combined.buffer();
            bencher.iter_custom(|iteration_count| {
                let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
                for _ in 0..iteration_count {
                    kernel.encode(
                        token_ple_buffer.borrow().deref(),
                        model_ple_buffer.borrow().deref(),
                        scales_buffer.borrow().deref(),
                        combined_buffer.borrow_mut().deref_mut(),
                        batch_size as u32,
                        num_layers as u32,
                        ple_dim as u32,
                        0.025515518,
                        0.70710677,
                        1e-6,
                        0.0,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            });
        },
    );
}

#[cfg(metal_backend)]
fn bench_tensor_mul_slice(
    context: &MetalContext,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let batch_size = 128usize;
    let num_layers = 35usize;
    let slice_dim = 256usize;
    let total_slice_dim = num_layers * slice_dim;
    let slice_index = 17usize;
    let values = context.create_array_uninitialized(&[batch_size, slice_dim], bf16::data_type(), "tensor_mul_values");
    let slice_source =
        context.create_array_uninitialized(&[batch_size, total_slice_dim], bf16::data_type(), "tensor_mul_slice");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::TensorMulSliceKernel::new(context, bf16::data_type())
        .expect("Failed to create TensorMulSliceKernel");

    group.throughput(Throughput::Elements((batch_size * slice_dim) as u64));
    group.bench_function(
        BenchmarkId::new("TensorMulSlice", format!("Batch[{batch_size}]SliceDim[{slice_dim}]")),
        |bencher| {
            let values_buffer = values.buffer();
            let slice_source_buffer = slice_source.buffer();
            bencher.iter_custom(|iteration_count| {
                let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
                for _ in 0..iteration_count {
                    kernel.encode(
                        values_buffer.borrow_mut().deref_mut(),
                        slice_source_buffer.borrow().deref(),
                        batch_size as u32,
                        total_slice_dim as u32,
                        slice_dim as u32,
                        slice_index as u32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            });
        },
    );
}

#[cfg(metal_backend)]
fn bench_tensor_finalize(
    context: &MetalContext,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let batch_size = 128usize;
    let model_dim = 1536usize;
    let length = batch_size * model_dim;
    let shortcut = context.create_array_uninitialized(&[batch_size, model_dim], bf16::data_type(), "finalize_shortcut");
    let main = context.create_array_uninitialized(&[batch_size, model_dim], bf16::data_type(), "finalize_main");
    let scalar = context.create_array_from(&[1], &[bf16::from_f32(0.5)], "finalize_scalar");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::TensorFinalizeKernel::new(context, bf16::data_type(), true)
        .expect("Failed to create TensorFinalizeKernel");

    group.throughput(Throughput::Elements(length as u64));
    group.bench_function(
        BenchmarkId::new("TensorFinalize", format!("Batch[{batch_size}]Dim[{model_dim}]")),
        |bencher| {
            let shortcut_buffer = shortcut.buffer();
            let main_buffer = main.buffer();
            let scalar_buffer = scalar.buffer();
            bencher.iter_custom(|iteration_count| {
                let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
                for _ in 0..iteration_count {
                    kernel.encode(
                        shortcut_buffer.borrow_mut().deref_mut(),
                        main_buffer.borrow_mut().deref_mut(),
                        Some(scalar_buffer.borrow().deref()),
                        length as u32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            });
        },
    );
}

#[cfg(metal_backend)]
fn bench_soft_cap(
    context: &MetalContext,
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let batch_size = 8usize;
    let vocabulary_size = 262_144usize;
    let logits =
        context.create_array_uninitialized(&[batch_size, vocabulary_size], bf16::data_type(), "soft_cap_logits");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::SoftCapKernel::new(context, bf16::data_type(), true)
        .expect("Failed to create SoftCapKernel");

    group.throughput(Throughput::Elements((batch_size * vocabulary_size) as u64));
    group.bench_function(
        BenchmarkId::new("SoftCap", format!("Batch[{batch_size}]Vocab[{vocabulary_size}]")),
        |bencher| {
            let logits_buffer = logits.buffer();
            bencher.iter_custom(|iteration_count| {
                let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
                for _ in 0..iteration_count {
                    kernel.encode(
                        None::<&<Metal as Backend>::Buffer>,
                        logits_buffer.borrow_mut().deref_mut(),
                        (batch_size * vocabulary_size) as u32,
                        30.0,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
            });
        },
    );
}
