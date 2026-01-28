#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::prelude::*;
use objc2::rc::autoreleasepool;
use uzu::backends::metal::allocator::{
    AllocatedBuffer, CachingAllocator, DirectAllocator, MetalAllocator,
    UzuAllocator,
};

const MB: usize = 1024 * 1024;
const GB: f64 = 1024.0 * 1024.0 * 1024.0;
const F16_SIZE: usize = 2;

#[derive(Clone, Copy)]
struct ModelConfig {
    name: &'static str,
    model_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
    weight_size_gb: f64,
}

impl ModelConfig {
    fn llama_3_2_1b() -> Self {
        Self {
            name: "Llama-3.2-1B",
            model_dim: 2048,
            num_heads: 32,
            num_groups: 8,
            head_dim: 64,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 16,
            weight_size_gb: 2.33,
        }
    }

    fn llama_3_2_3b() -> Self {
        Self {
            name: "Llama-3.2-3B",
            model_dim: 3072,
            num_heads: 24,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 28,
            weight_size_gb: 6.04,
        }
    }

    fn qwen3_4b() -> Self {
        Self {
            name: "Qwen3-4B",
            model_dim: 2560,
            num_heads: 32,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 9728,
            vocab_size: 151936,
            num_layers: 36,
            weight_size_gb: 7.51,
        }
    }

    fn llama_3_1_8b() -> Self {
        Self {
            name: "Llama-3.1-8B",
            model_dim: 4096,
            num_heads: 32,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 14336,
            vocab_size: 128256,
            num_layers: 32,
            weight_size_gb: 15.02,
        }
    }

    fn qwen_coder_14b() -> Self {
        Self {
            name: "Qwen-Coder-14B",
            model_dim: 5120,
            num_heads: 40,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 13824,
            vocab_size: 152064,
            num_layers: 48,
            weight_size_gb: 27.52,
        }
    }

    fn qwen_coder_32b() -> Self {
        Self {
            name: "Qwen-Coder-32B",
            model_dim: 5120,
            num_heads: 40,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 27648,
            vocab_size: 152064,
            num_layers: 64,
            weight_size_gb: 61.04,
        }
    }

    fn qkv_dim(&self) -> usize {
        (2 * self.num_groups + self.num_heads) * self.head_dim
    }
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    use_case: &'static str,
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
}

impl Scenario {
    fn chat_short() -> Self {
        Self {
            name: "Chat Short",
            use_case: "Quick Q&A",
            prompt_length: 64,
            generate_length: 128,
            prefill_step_size: 512,
        }
    }

    fn chat_medium() -> Self {
        Self {
            name: "Chat Medium",
            use_case: "Conversation",
            prompt_length: 256,
            generate_length: 512,
            prefill_step_size: 512,
        }
    }

    fn code_completion() -> Self {
        Self {
            name: "Code Completion",
            use_case: "IDE autocomplete",
            prompt_length: 512,
            generate_length: 256,
            prefill_step_size: 512,
        }
    }

    fn long_generation() -> Self {
        Self {
            name: "Long Generation",
            use_case: "Story/article",
            prompt_length: 256,
            generate_length: 2048,
            prefill_step_size: 512,
        }
    }

    fn max_context() -> Self {
        Self {
            name: "Max Context",
            use_case: "RAG/summarization",
            prompt_length: 4096,
            generate_length: 512,
            prefill_step_size: 512,
        }
    }
}

#[derive(Clone, Copy)]
struct AllocationStats {
    total_allocs: usize,
    total_alloc_time_ns: u128,
    total_free_time_ns: u128,
    total_bytes_requested: usize,
}

impl AllocationStats {
    fn new() -> Self {
        Self {
            total_allocs: 0,
            total_alloc_time_ns: 0,
            total_free_time_ns: 0,
            total_bytes_requested: 0,
        }
    }

    fn record_alloc(
        &mut self,
        size: usize,
        duration: Duration,
    ) {
        self.total_allocs += 1;
        self.total_alloc_time_ns += duration.as_nanos();
        self.total_bytes_requested += size;
    }

    fn record_free(
        &mut self,
        duration: Duration,
    ) {
        self.total_free_time_ns += duration.as_nanos();
    }
}

struct InferenceSimulation<'a, A: MetalAllocator> {
    allocator: &'a A,
    config: ModelConfig,
    stats: AllocationStats,
    current_prefix_len: usize,
}

impl<'a, A: MetalAllocator> InferenceSimulation<'a, A> {
    fn new(
        allocator: &'a A,
        config: ModelConfig,
    ) -> Self {
        Self {
            allocator,
            config,
            stats: AllocationStats::new(),
            current_prefix_len: 0,
        }
    }

    fn alloc_scratch(
        &mut self,
        size: usize,
    ) -> Result<AllocatedBuffer, String> {
        let start = Instant::now();
        let result = self.allocator.alloc_scratch(size);
        self.stats.record_alloc(size, start.elapsed());
        result.map_err(|e| format!("{:?}", e))
    }

    fn free(
        &mut self,
        buffer: AllocatedBuffer,
    ) {
        let start = Instant::now();
        self.allocator.free(buffer);
        self.stats.record_free(start.elapsed());
    }

    fn simulate_embedding_kernel(
        &mut self,
        batch_size: usize,
    ) -> Result<AllocatedBuffer, String> {
        let size = batch_size * self.config.model_dim * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_attention_kernel(
        &mut self,
        batch_size: usize,
    ) -> Result<Vec<AllocatedBuffer>, String> {
        let mut buffers = Vec::new();

        let qkv_size = batch_size * self.config.qkv_dim() * F16_SIZE;
        buffers.push(self.alloc_scratch(qkv_size)?);

        let bias_size =
            batch_size * (batch_size + self.current_prefix_len) * F16_SIZE;
        buffers.push(self.alloc_scratch(bias_size)?);

        let rotated_q_size = self.config.num_heads
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_q_size)?);

        let rotated_k_size = self.config.num_groups
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_k_size)?);

        let attn_out_size = batch_size
            * self.config.num_heads
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(attn_out_size)?);

        Ok(buffers)
    }

    fn simulate_mlp_kernel(
        &mut self,
        batch_size: usize,
    ) -> Result<Vec<AllocatedBuffer>, String> {
        let mut buffers = Vec::new();

        let up_size = batch_size * 2 * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(up_size)?);

        let hidden_size = batch_size * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(hidden_size)?);

        Ok(buffers)
    }

    fn simulate_logits_kernel(
        &mut self,
        batch_size: usize,
    ) -> Result<AllocatedBuffer, String> {
        let size = batch_size * self.config.vocab_size * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_forward_pass(
        &mut self,
        batch_size: usize,
    ) -> Result<(), String> {
        let main = self.simulate_embedding_kernel(batch_size)?;

        for _layer in 0..self.config.num_layers {
            let attn_buffers = self.simulate_attention_kernel(batch_size)?;
            for buf in attn_buffers {
                self.free(buf);
            }

            let mlp_buffers = self.simulate_mlp_kernel(batch_size)?;
            for buf in mlp_buffers {
                self.free(buf);
            }
        }

        let logits = self.simulate_logits_kernel(batch_size)?;
        self.free(logits);
        self.free(main);

        Ok(())
    }

    fn run_prefill(
        &mut self,
        prompt_length: usize,
        step_size: usize,
    ) -> Result<(), String> {
        let mut remaining = prompt_length;

        while remaining > 0 {
            let batch_size = remaining.min(step_size);
            self.simulate_forward_pass(batch_size)?;
            self.current_prefix_len += batch_size;
            remaining = remaining.saturating_sub(step_size);
        }

        Ok(())
    }

    fn run_generation(
        &mut self,
        num_tokens: usize,
    ) -> Result<(), String> {
        for _ in 0..num_tokens {
            self.simulate_forward_pass(1)?;
            self.current_prefix_len += 1;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    allocator_name: &'static str,
    total_alloc_time_ms: f64,
    total_free_time_ms: f64,
    total_allocations: usize,
    total_bytes_mb: f64,
    peak_memory_mb: f64,
    cache_memory_mb: f64,
}

fn run_simulation<A: MetalAllocator>(
    allocator: &A,
    allocator_name: &'static str,
    model: ModelConfig,
    scenario: Scenario,
) -> BenchmarkResult {
    allocator.reset_peak_memory();
    allocator.clear_cache();

    let mut sim = InferenceSimulation::new(allocator, model);

    sim.run_prefill(scenario.prompt_length, scenario.prefill_step_size)
        .expect("Prefill failed");

    sim.run_generation(scenario.generate_length).expect("Generation failed");

    BenchmarkResult {
        allocator_name,
        total_alloc_time_ms: sim.stats.total_alloc_time_ns as f64 / 1_000_000.0,
        total_free_time_ms: sim.stats.total_free_time_ns as f64 / 1_000_000.0,
        total_allocations: sim.stats.total_allocs,
        total_bytes_mb: sim.stats.total_bytes_requested as f64 / MB as f64,
        peak_memory_mb: allocator.peak_memory() as f64 / MB as f64,
        cache_memory_mb: allocator.cache_memory() as f64 / MB as f64,
    }
}

fn run_model_scenario_benchmark(
    model: ModelConfig,
    scenario: Scenario,
) {
    println!(
        "\n--- Model: {} ({:.2} GB) | Scenario: {} ({}) ---",
        model.name, model.weight_size_gb, scenario.name, scenario.use_case
    );
    println!(
        "    Config: {}p + {}g | {} layers | vocab {}",
        scenario.prompt_length,
        scenario.generate_length,
        model.num_layers,
        model.vocab_size
    );
    println!();

    let direct = DirectAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let caching = CachingAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let uzu = UzuAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );

    let direct_result =
        run_simulation(&direct, "DirectAllocator", model, scenario);
    let caching_result =
        run_simulation(&caching, "CachingAllocator", model, scenario);
    let uzu_result = run_simulation(&uzu, "UzuAllocator", model, scenario);

    let baseline_time =
        direct_result.total_alloc_time_ms + direct_result.total_free_time_ms;

    println!(
        "    {:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Allocator", "Alloc ms", "Total ms", "Peak MB", "Cache MB", "Speedup"
    );
    println!("    {}", "-".repeat(74));

    for result in [&direct_result, &caching_result, &uzu_result] {
        let total_time = result.total_alloc_time_ms + result.total_free_time_ms;
        let speedup = baseline_time / total_time;
        println!(
            "    {:<20} {:>10.2} {:>10.2} {:>10.1} {:>10.1} {:>9.1}x",
            result.allocator_name,
            result.total_alloc_time_ms,
            total_time,
            result.peak_memory_mb,
            result.cache_memory_mb,
            speedup,
        );
    }
}

fn run_tier_benchmarks(
    tier_name: &str,
    models: &[ModelConfig],
    scenarios: &[Scenario],
) {
    println!("\n{}", "=".repeat(80));
    println!("=== {} ===", tier_name);
    println!("{}", "=".repeat(80));

    for model in models {
        for scenario in scenarios {
            run_model_scenario_benchmark(*model, *scenario);
        }
    }
}

fn print_summary_table(all_results: &[(ModelConfig, Scenario, f64, f64, f64)]) {
    println!("\n{}", "=".repeat(100));
    println!("=== SUMMARY: UzuAllocator Speedup vs DirectAllocator ===");
    println!("{}", "=".repeat(100));
    println!();

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Model",
        "Chat Short",
        "Chat Medium",
        "Code Comp",
        "Long Gen",
        "Max Context"
    );
    println!("{}", "-".repeat(84));

    let models: Vec<&str> = all_results
        .iter()
        .map(|(m, _, _, _, _)| m.name)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for model_name in &models {
        let row: Vec<String> = [
            "Chat Short",
            "Chat Medium",
            "Code Completion",
            "Long Generation",
            "Max Context",
        ]
        .iter()
        .map(|scenario_name| {
            all_results
                .iter()
                .find(|(m, s, _, _, _)| {
                    m.name == *model_name && s.name == *scenario_name
                })
                .map(|(_, _, _, _, speedup)| format!("{:.1}x", speedup))
                .unwrap_or_else(|| "-".to_string())
        })
        .collect();

        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}",
            model_name, row[0], row[1], row[2], row[3], row[4]
        );
    }
}

pub fn main() {
    autoreleasepool(|_| {
        let device = <dyn MTLDevice>::system_default()
            .expect("No Metal device available");
        println!("=== Multi-Model Allocator Benchmark ===");
        println!("Device: {}", device.name());
        println!();

        let small_models = [ModelConfig::llama_3_2_1b()];

        let medium_models =
            [ModelConfig::llama_3_2_3b(), ModelConfig::qwen3_4b()];

        let large_models = [ModelConfig::llama_3_1_8b()];

        let xlarge_models =
            [ModelConfig::qwen_coder_14b(), ModelConfig::qwen_coder_32b()];

        let common_scenarios = [
            Scenario::chat_short(),
            Scenario::chat_medium(),
            Scenario::long_generation(),
        ];

        let code_scenarios = [Scenario::code_completion()];

        let large_scenarios = [
            Scenario::chat_short(),
            Scenario::chat_medium(),
            Scenario::max_context(),
        ];

        run_tier_benchmarks(
            "Tier 2: Small Models (1-3GB)",
            &small_models,
            &common_scenarios,
        );
        run_tier_benchmarks(
            "Tier 3: Medium Models (4-8GB)",
            &medium_models,
            &common_scenarios,
        );
        run_tier_benchmarks(
            "Tier 3: Medium Models - Code Scenarios",
            &medium_models,
            &code_scenarios,
        );
        run_tier_benchmarks(
            "Tier 4: Large Models (8-16GB)",
            &large_models,
            &large_scenarios,
        );
        run_tier_benchmarks(
            "Tier 5: Extra-Large Models (16GB+)",
            &xlarge_models,
            &[Scenario::chat_short(), Scenario::chat_medium()],
        );

        println!("\n{}", "=".repeat(80));
        println!("=== Benchmark Complete ===");
        println!("{}", "=".repeat(80));
    });
}
