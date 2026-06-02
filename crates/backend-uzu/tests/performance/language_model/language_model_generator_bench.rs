use std::time::Duration;

use backend_uzu::{
    _private::{LanguageModelConfig, LanguageModelGenerator, TrieCreationConfig, TrieNode},
    backends::common::Backend,
    prelude::{DecodingConfig, SamplingMethod},
};
use criterion::{BatchSize, BenchmarkId, Criterion};

use crate::{common::path::get_test_model_path, uzu_bench};

fn create_generator<B: Backend>() -> LanguageModelGenerator<B> {
    let model_path = get_test_model_path();
    let decoding_config = DecodingConfig::default();
    let model_config = LanguageModelConfig::new(&model_path).unwrap();
    LanguageModelGenerator::new(&model_path, decoding_config, &model_config).unwrap()
}

fn run_generator<B: Backend>(
    generator: &mut LanguageModelGenerator<B>,
    tokens: &[u64],
) -> Duration {
    let speculator = &generator.decoding_config.speculator_config.speculator;
    let suffix_length = generator.decoding_config.generate_suffix_length();
    let suffix_root = TrieNode::from_speculator(
        &generator.tokens,
        &generator.context.seed,
        None,
        speculator.as_ref(),
        &TrieCreationConfig::default(),
        suffix_length,
    );
    let flat_trie = suffix_root.linearize();

    generator.tokens = tokens.to_vec();
    let task = generator.get_generate_task(&flat_trie, false);
    let result = generator.run_model(task, SamplingMethod::Greedy).unwrap();
    result.gpu_run_time
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
                create_generator::<B>,
                |generator| {
                    let duration = run_generator(generator, &tokens);

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
