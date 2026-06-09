use std::time::Duration;

use backend_uzu::{
    _benchmarks::{LanguageModelConfig, LanguageModelGenerator, RunModelResult, TrieCreationConfig, TrieNode},
    backends::common::Backend,
    prelude::{DecodingConfig, SamplingMethod},
};
use criterion::{BenchmarkId, Criterion, Throughput};

use crate::{
    common::{metrics::wait_gpu_cooldown, path::get_test_model_path},
    uzu_bench,
};

fn create_generator<B: Backend>() -> LanguageModelGenerator<B> {
    let model_path = get_test_model_path();
    let decoding_config = DecodingConfig::default();
    let model_config = LanguageModelConfig::new(&model_path).unwrap();
    LanguageModelGenerator::new(&model_path, decoding_config, &model_config).unwrap()
}

fn run_generator<B: Backend>(
    generator: &mut LanguageModelGenerator<B>,
    tokens: &[u64],
) -> RunModelResult<B> {
    generator.tokens = tokens.to_vec();
    let speculator = &generator.decoding_config.speculator_config.speculator;
    let suffix_length = generator.decoding_config.generate_suffix_length();
    let suffix_root = TrieNode::from_speculator(
        &generator.tokens,
        &generator.context.seed,
        None,
        speculator.as_ref(),
        1024, // greedy, doesn't matter
        &TrieCreationConfig::default(),
        suffix_length,
    );
    let flat_trie = suffix_root.linearize();

    let task = generator.get_generate_task(&flat_trie, false);
    generator.run_model(task, SamplingMethod::Greedy).unwrap()
}

#[uzu_bench]
fn bench_forward_pass(c: &mut Criterion) {
    // Empty prompt first decode: the generate path requires one seed token.
    let tokens: Vec<u64> = vec![0];
    for_each_non_cpu_backend!(|B| {
        wait_gpu_cooldown();

        let mut group = c.benchmark_group("Forward pass");
        group.sample_size(10);
        group.throughput(Throughput::Elements(tokens.len() as u64));
        group.bench_function(BenchmarkId::from_parameter("empty prompt first decode"), |b| {
            b.iter_custom(|n_iters| {
                let mut total_duration = Duration::from_secs(0);
                for _ in 0..n_iters {
                    let mut generator = create_generator::<B>();
                    let result = run_generator(&mut generator, &tokens);
                    total_duration += result.gpu_run_time
                }
                total_duration
            });
        });
        group.finish()
    })
}
