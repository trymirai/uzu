use std::time::{Duration, Instant};

use backend_uzu::{
    backends::common::Backend,
    prelude::{DecodingConfig, Output, RunConfig},
    session::ChatSession,
};
use criterion::{BatchSize, BenchmarkId, Criterion};

use crate::{common::path::get_test_model_path, uzu_bench};

fn create_session<B: Backend>() -> ChatSession {
    let model_path = get_test_model_path();
    let decoding_config = DecodingConfig::default();
    ChatSession::new_with_backend::<B>(model_path.clone(), decoding_config).unwrap()
}

fn run_session(
    session: &mut ChatSession,
    tokens: &Vec<u64>,
) -> Duration {
    let run_config = RunConfig::default();
    let start = Instant::now();
    let output = session.run_forward_pass(tokens, run_config, None::<fn(Output) -> bool>, start).unwrap();
    Duration::from_secs_f64(output.stats.total_stats.gpu_duration)
}

#[uzu_bench]
fn bench_forward_pass(c: &mut Criterion) {
    let tokens: Vec<u64> = vec![0; 128];
    for_each_non_cpu_backend!(|B| {
        let mut iter = 0;
        let mut group = c.benchmark_group("Forward pass");
        group.sample_size(10);
        group.bench_function(BenchmarkId::from_parameter(format!("{} zero tokens", tokens.len())), |b| {
            b.iter_batched_ref(
                || create_session::<B>(),
                |session| {
                    // TODO: maybe add waiting for GPU cooling

                    let duration = run_session(session, &tokens);

                    if iter == 0 || iter == 1 {
                        println!()
                    }
                    println!("Iteration {iter} completed in {:?}", duration);

                    iter += 1;
                    duration
                },
                BatchSize::PerIteration,
            )
        });
        group.finish()
    })
}
