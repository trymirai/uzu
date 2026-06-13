use std::time::Duration;

use backend_uzu::{
    backends::common::Backend,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        types::{Input, Output},
    },
    tests::metrics::wait_gpu_cooldown,
};
use criterion::{BenchmarkId, Criterion};
use proc_macros::uzu_bench;

use crate::common::path::get_test_model_path;

fn create_session<B: Backend>() -> ChatSession {
    let model_path = get_test_model_path();
    let decoding_config = DecodingConfig::default();
    ChatSession::new_with_backend::<B>(model_path, decoding_config).unwrap()
}

fn run_session(
    session: &mut ChatSession,
    input: Input,
    tokens_limit: u64,
) -> Duration {
    let run_config = RunConfig {
        tokens_limit,
        ..Default::default()
    };
    let result = session.run_internal(input, run_config, None::<fn(Output) -> bool>).unwrap();
    Duration::from_secs_f64(result.stats.total_stats.duration)
}

#[uzu_bench]
fn bench_chat_session(c: &mut Criterion) {
    let message = "How are you?";
    let tokens_limit = 64;

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
                    let duration = run_session(&mut session, Input::Text(message.to_string()), tokens_limit);
                    total_duration += duration;
                }
                total_duration
            });
        });
        group.finish();
    });
}
