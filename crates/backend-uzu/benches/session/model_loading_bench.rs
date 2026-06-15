use std::time::Duration;

use backend_uzu::session::{ChatSession, config::DecodingConfig};
use criterion::{BenchmarkId, Criterion};
use proc_macros::uzu_bench;
use test_runner::path::get_test_model_path;

#[uzu_bench]
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model loading");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.bench_function(BenchmarkId::from_parameter("test model path"), |b| {
        b.iter(|| {
            let model_path = get_test_model_path();
            let _ = ChatSession::new(model_path, DecodingConfig::default());
        });
    });
}
