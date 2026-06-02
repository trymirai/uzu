use std::{path::Path, time::Duration};

use backend_uzu::{
    _private::{LanguageModelConfig, LanguageModelGenerator},
    backends::common::Backend,
    prelude::DecodingConfig,
};
use criterion::{BatchSize, BenchmarkId, Criterion};
use monostate::alphabet::B;

use crate::{common::path::get_test_model_path, uzu_bench};

fn create_generator<B: Backend>(model_path: &Path) -> LanguageModelGenerator<B> {
    let decoding_config = DecodingConfig::default();
    let model_config = LanguageModelConfig::new(model_path).unwrap();
    LanguageModelGenerator::new(model_path, decoding_config, &model_config).unwrap()
}

fn run_generator<B: Backend>(generator: &mut LanguageModelGenerator<B>) -> Duration {
    // let task = Task {
    //
    // };
    // let result = generator.run_model()
}

#[uzu_bench]
fn bench_forward_pass(c: &mut Criterion) {
    let tokens: Vec<u64> = vec![0; 128];
    let model_path = get_test_model_path();
    for_each_non_cpu_backend!(|B| {
        let mut iter = 0;
        let mut group = c.benchmark_group("Forward pass");
        group.sample_size(10);
        group.bench_function(BenchmarkId::from_parameter(format!("{} zero tokens", tokens.len())), |b| {
            b.iter_batched_ref(create_generator::<B>, |generator| {}, BatchSize::PerIteration)
            // b.iter_batched_ref(
            //     || create_session::<B>(),
            //     |session| {
            //         // TODO: maybe add waiting for GPU cooling
            //
            //         let duration = run_session(session, &tokens);
            //
            //         if iter == 0 || iter == 1 {
            //             println!()
            //         }
            //         println!("Iteration {iter} completed in {:?}", duration);
            //
            //         iter += 1;
            //         duration
            //     },
            //     BatchSize::PerIteration,
            // )
        });
        group.finish()
    })
}
