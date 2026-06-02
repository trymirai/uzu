use std::time::Duration;

use backend_uzu::{
    backends::common::Backend,
    prelude::{DecodingConfig, Input, Output, RunConfig},
    session::ChatSession,
};
use criterion::{BatchSize, BenchmarkId, Criterion};

use crate::{common::path::get_test_model_path, uzu_bench};

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
    let mut iter = 0;
    for_each_non_cpu_backend!(|B| {
        let mut group = c.benchmark_group("ChatSession run");
        group.bench_function(BenchmarkId::from_parameter(message), |b| {
            b.iter_batched_ref(
                create_session::<B>,
                |session| {
                    let duration = run_session(session, Input::Text(message.to_string()));

                    println!("Iter {iter}: {duration:?}");
                    iter += 1;
                    duration
                },
                BatchSize::PerIteration,
            );
        });
        group.finish();
    });
}
