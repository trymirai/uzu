use std::time::Duration;

use backend_uzu::engine::Engine;
use criterion::{BenchmarkId, Criterion};
use proc_macros::uzu_bench;
use test_runner::{for_each_non_cpu_backend, path::get_test_model_path};

#[uzu_bench]
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model loading");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    for_each_non_cpu_backend!(|B| {
        group.bench_function(BenchmarkId::from_parameter("test model path"), |b| {
            b.iter(|| {
                // TODO: Should this reuse engine not measure creation?
                let model_path = get_test_model_path();
                let engine = Engine::<B>::new().unwrap();
                // TODO: This should flush caches (maybe both cold/hot benchmarks?)
                engine.load_language_model(&model_path).unwrap();
            });
        });
    });
}
