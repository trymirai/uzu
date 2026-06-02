use std::time::Duration;

use backend_uzu::{
    backends::common::Backend,
    prelude::{DecodingConfig, Input, Output, RunConfig},
    session::ChatSession,
};
use criterion::{BenchmarkId, Criterion};

use crate::{
    common::{metrics::wait_gpu_cooldown, path::get_test_model_path},
    uzu_bench,
};

fn create_session<B: Backend>() -> ChatSession {
    let model_path = get_test_model_path();
    let decoding_config = DecodingConfig::default();
    ChatSession::new_with_backend::<B>(model_path, decoding_config).unwrap()
}

fn run_session(
    session: &mut ChatSession,
    input: Input,
) -> Duration {
    let mut run_config = RunConfig::default();
    run_config.tokens_limit = 64;
    let result = session.run_internal(input, run_config, None::<fn(Output) -> bool>).unwrap();
    Duration::from_secs_f64(result.stats.total_stats.duration)
}

#[uzu_bench]
fn bench_chat_session(c: &mut Criterion) {
    let message = "How are you?";
    for_each_non_cpu_backend!(|B| {
        wait_gpu_cooldown();

        let mut session = create_session::<B>();
        let mut group = c.benchmark_group("ChatSession run");
        group.sample_size(10);
        group.bench_function(BenchmarkId::from_parameter(message), |b| {
            b.iter_custom(|n_iter| {
                let mut total_duration = Duration::from_secs(0);
                for _ in 0..n_iter {
                    session.reset().unwrap();
                    let duration = run_session(&mut session, Input::Text(message.to_string()));
                    total_duration += duration;
                }
                total_duration
            });
        });
        group.finish();
    });
}
