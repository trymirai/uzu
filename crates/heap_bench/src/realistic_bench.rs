#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::prelude::*;
use objc2::rc::autoreleasepool;
use uzu::backends::metal::allocator::{
    AllocatedBuffer, BucketedAllocator, CachingAllocator, DirectAllocator, MetalAllocator,
    TieredAllocator, UzuAllocator,
};

const MB: usize = 1024 * 1024;
const KB: usize = 1024;
const F16_SIZE: usize = 2;

#[derive(Clone, Copy)]
struct ModelConfig {
    model_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
}

impl ModelConfig {
    fn llama_3_2_1b() -> Self {
        Self {
            model_dim: 2048,
            num_heads: 32,
            num_groups: 8,
            head_dim: 64,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 16,
        }
    }

    fn qkv_dim(&self) -> usize {
        (2 * self.num_groups + self.num_heads) * self.head_dim
    }
}

#[derive(Clone, Copy)]
struct AllocationStats {
    total_allocs: usize,
    total_frees: usize,
    total_alloc_time_ns: u128,
    total_free_time_ns: u128,
    total_bytes_requested: usize,
    cache_hits: usize,
}

impl AllocationStats {
    fn new() -> Self {
        Self {
            total_allocs: 0,
            total_frees: 0,
            total_alloc_time_ns: 0,
            total_free_time_ns: 0,
            total_bytes_requested: 0,
            cache_hits: 0,
        }
    }

    fn record_alloc(&mut self, size: usize, duration: Duration) {
        self.total_allocs += 1;
        self.total_alloc_time_ns += duration.as_nanos();
        self.total_bytes_requested += size;
    }

    fn record_free(&mut self, duration: Duration) {
        self.total_frees += 1;
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
    fn new(allocator: &'a A, config: ModelConfig) -> Self {
        Self {
            allocator,
            config,
            stats: AllocationStats::new(),
            current_prefix_len: 0,
        }
    }

    fn alloc_scratch(&mut self, size: usize) -> Result<AllocatedBuffer, String> {
        let start = Instant::now();
        let result = self.allocator.alloc_scratch(size);
        self.stats.record_alloc(size, start.elapsed());
        result.map_err(|e| format!("{:?}", e))
    }

    fn free(&mut self, buffer: AllocatedBuffer) {
        let start = Instant::now();
        self.allocator.free(buffer);
        self.stats.record_free(start.elapsed());
    }

    fn simulate_embedding_kernel(&mut self, batch_size: usize) -> Result<AllocatedBuffer, String> {
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

        let bias_size = batch_size * (batch_size + self.current_prefix_len) * F16_SIZE;
        buffers.push(self.alloc_scratch(bias_size)?);

        let rotated_q_size =
            self.config.num_heads * batch_size * self.config.head_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_q_size)?);

        let rotated_k_size =
            self.config.num_groups * batch_size * self.config.head_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_k_size)?);

        let attn_out_size =
            batch_size * self.config.num_heads * self.config.head_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(attn_out_size)?);

        Ok(buffers)
    }

    fn simulate_mlp_kernel(&mut self, batch_size: usize) -> Result<Vec<AllocatedBuffer>, String> {
        let mut buffers = Vec::new();

        let up_size = batch_size * 2 * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(up_size)?);

        let hidden_size = batch_size * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(hidden_size)?);

        Ok(buffers)
    }

    fn simulate_logits_kernel(&mut self, batch_size: usize) -> Result<AllocatedBuffer, String> {
        let size = batch_size * self.config.vocab_size * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_forward_pass(&mut self, batch_size: usize) -> Result<(), String> {
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

    fn run_prefill(&mut self, prompt_length: usize, step_size: usize) -> Result<(), String> {
        let mut remaining = prompt_length;

        while remaining > 0 {
            let batch_size = remaining.min(step_size);
            self.simulate_forward_pass(batch_size)?;
            self.current_prefix_len += batch_size;
            remaining = remaining.saturating_sub(step_size);
        }

        Ok(())
    }

    fn run_generation(&mut self, num_tokens: usize) -> Result<(), String> {
        for _ in 0..num_tokens {
            self.simulate_forward_pass(1)?;
            self.current_prefix_len += 1;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    allocator_name: String,
    scenario: String,
    total_alloc_time_ms: f64,
    total_free_time_ms: f64,
    total_allocations: usize,
    total_bytes_mb: f64,
    peak_memory_mb: f64,
    cache_memory_mb: f64,
    final_prefix_len: usize,
}

fn run_simulation<A: MetalAllocator>(
    allocator: &A,
    allocator_name: &str,
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
) -> BenchmarkResult {
    allocator.reset_peak_memory();
    allocator.clear_cache();

    let config = ModelConfig::llama_3_2_1b();
    let mut sim = InferenceSimulation::new(allocator, config);

    sim.run_prefill(prompt_length, prefill_step_size)
        .expect("Prefill failed");

    sim.run_generation(generate_length)
        .expect("Generation failed");

    BenchmarkResult {
        allocator_name: allocator_name.to_string(),
        scenario: format!("{}p_{}g", prompt_length, generate_length),
        total_alloc_time_ms: sim.stats.total_alloc_time_ns as f64 / 1_000_000.0,
        total_free_time_ms: sim.stats.total_free_time_ns as f64 / 1_000_000.0,
        total_allocations: sim.stats.total_allocs,
        total_bytes_mb: sim.stats.total_bytes_requested as f64 / MB as f64,
        peak_memory_mb: allocator.peak_memory() as f64 / MB as f64,
        cache_memory_mb: allocator.cache_memory() as f64 / MB as f64,
        final_prefix_len: sim.current_prefix_len,
    }
}

fn print_result(result: &BenchmarkResult) {
    println!("  {} ({}):", result.allocator_name, result.scenario);
    println!(
        "    Alloc time:     {:>8.3} ms ({} allocations)",
        result.total_alloc_time_ms, result.total_allocations
    );
    println!("    Free time:      {:>8.3} ms", result.total_free_time_ms);
    println!(
        "    Total time:     {:>8.3} ms",
        result.total_alloc_time_ms + result.total_free_time_ms
    );
    println!("    Bytes requested:{:>8.2} MB", result.total_bytes_mb);
    println!("    Peak memory:    {:>8.2} MB", result.peak_memory_mb);
    println!("    Cache memory:   {:>8.2} MB", result.cache_memory_mb);
    println!("    Final prefix:   {:>8} tokens", result.final_prefix_len);
}

fn run_benchmark_suite(
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
    name: &str,
) {
    println!("=== {} ===", name);
    println!();
    println!(
        "Configuration: {} prompt, {} generate, {} prefill step",
        prompt_length, generate_length, prefill_step_size
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
    let bucketed = BucketedAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let tiered = TieredAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );

    println!("--- Per-Layer Simulation Results ---");
    println!();

    let direct_result = run_simulation(
        &direct,
        "DirectAllocator",
        prompt_length,
        generate_length,
        prefill_step_size,
    );
    print_result(&direct_result);
    println!();

    let caching_result = run_simulation(
        &caching,
        "CachingAllocator",
        prompt_length,
        generate_length,
        prefill_step_size,
    );
    print_result(&caching_result);
    println!();

    let uzu_result = run_simulation(
        &uzu,
        "UzuAllocator",
        prompt_length,
        generate_length,
        prefill_step_size,
    );
    print_result(&uzu_result);
    println!();

    let bucketed_result = run_simulation(
        &bucketed,
        "BucketedAllocator",
        prompt_length,
        generate_length,
        prefill_step_size,
    );
    print_result(&bucketed_result);
    println!();

    let tiered_result = run_simulation(
        &tiered,
        "TieredAllocator",
        prompt_length,
        generate_length,
        prefill_step_size,
    );
    print_result(&tiered_result);
    println!();

    println!("--- Summary ---");
    println!();
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10}",
        "Allocator", "Alloc ms", "Total ms", "Peak MB", "Cache MB"
    );
    println!("{}", "-".repeat(64));

    for result in [
        &direct_result,
        &caching_result,
        &uzu_result,
        &bucketed_result,
        &tiered_result,
    ] {
        println!(
            "{:<20} {:>10.3} {:>10.3} {:>10.2} {:>10.2}",
            result.allocator_name,
            result.total_alloc_time_ms,
            result.total_alloc_time_ms + result.total_free_time_ms,
            result.peak_memory_mb,
            result.cache_memory_mb,
        );
    }
    println!();
}

fn print_allocation_timeline() {
    let config = ModelConfig::llama_3_2_1b();

    println!("=== Allocation Timeline (Llama-3.2-1B) ===");
    println!();
    println!("Model Config:");
    println!("  model_dim: {}", config.model_dim);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_groups: {}", config.num_groups);
    println!("  head_dim: {}", config.head_dim);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  num_layers: {}", config.num_layers);
    println!();

    println!("Per-iteration buffer sizes (batch=1):");
    println!();

    let batch = 1;
    let prefix_start = 256;
    let prefix_end = 384;

    println!("  Fixed-size buffers:");
    println!(
        "    main/shortcut:    {:>8} bytes",
        batch * config.model_dim * F16_SIZE
    );
    println!(
        "    qkv:              {:>8} bytes",
        batch * config.qkv_dim() * F16_SIZE
    );
    println!(
        "    rotated_queries:  {:>8} bytes",
        config.num_heads * batch * config.head_dim * F16_SIZE
    );
    println!(
        "    rotated_keys:     {:>8} bytes",
        config.num_groups * batch * config.head_dim * F16_SIZE
    );
    println!(
        "    attn_output:      {:>8} bytes",
        batch * config.num_heads * config.head_dim * F16_SIZE
    );
    println!(
        "    mlp_up:           {:>8} bytes",
        batch * 2 * config.hidden_dim * F16_SIZE
    );
    println!(
        "    mlp_hidden:       {:>8} bytes",
        batch * config.hidden_dim * F16_SIZE
    );
    println!(
        "    logits:           {:>8} bytes ({:.2} KB)",
        batch * config.vocab_size * F16_SIZE,
        (batch * config.vocab_size * F16_SIZE) as f64 / KB as f64
    );
    println!();

    println!("  Growing buffers (attention_bias):");
    println!("    prefix={}:  {} bytes", prefix_start, batch * (batch + prefix_start) * F16_SIZE);
    println!("    prefix={}:  {} bytes", prefix_end, batch * (batch + prefix_end) * F16_SIZE);
    println!(
        "    Growth per token: {} bytes",
        batch * F16_SIZE
    );
    println!();

    println!("  Per-layer allocation count: 7 buffers");
    println!("  Total per iteration: {} buffers", 7 * config.num_layers + 2);
    println!();

    let buffers_per_iter = 7 * config.num_layers + 2;
    let prefill_iters = 1;
    let generate_iters = 128;
    let total_allocs = buffers_per_iter * (prefill_iters + generate_iters);

    println!("  For 256p + 128g scenario:");
    println!("    Prefill iterations: {}", prefill_iters);
    println!("    Generate iterations: {}", generate_iters);
    println!("    Total allocations: {}", total_allocs);
    println!();
}

fn run_size_drift_analysis<A: MetalAllocator>(
    allocator: &A,
    allocator_name: &str,
    prompt_length: usize,
    generate_length: usize,
) {
    allocator.reset_peak_memory();
    allocator.clear_cache();

    let config = ModelConfig::llama_3_2_1b();
    let mut sim = InferenceSimulation::new(allocator, config);

    sim.run_prefill(prompt_length, 512).expect("Prefill failed");

    let mut checkpoints = Vec::new();
    let check_interval = generate_length / 10;

    for i in 0..generate_length {
        sim.simulate_forward_pass(1).expect("Forward pass failed");
        sim.current_prefix_len += 1;

        if i > 0 && i % check_interval == 0 {
            checkpoints.push((
                sim.current_prefix_len,
                allocator.cache_memory(),
                allocator.active_memory(),
            ));
        }
    }

    println!("  {} size drift over {} tokens:", allocator_name, generate_length);
    println!(
        "    {:>10} {:>12} {:>12}",
        "Prefix", "Cache KB", "Active KB"
    );

    for (prefix, cache, active) in checkpoints {
        println!(
            "    {:>10} {:>12.1} {:>12.1}",
            prefix,
            cache as f64 / KB as f64,
            active as f64 / KB as f64,
        );
    }

    println!();
}

fn run_size_drift_test() {
    println!("=== Size Drift Analysis (256p + 2048g) ===");
    println!();
    println!("This test measures cache growth as context length increases.");
    println!("Lower cache = better memory efficiency (less fragmentation).");
    println!();

    let uzu = UzuAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let bucketed = BucketedAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let tiered = TieredAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let caching = CachingAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );

    run_size_drift_analysis(&caching, "CachingAllocator", 256, 2048);
    run_size_drift_analysis(&uzu, "UzuAllocator", 256, 2048);
    run_size_drift_analysis(&bucketed, "BucketedAllocator", 256, 2048);
    run_size_drift_analysis(&tiered, "TieredAllocator", 256, 2048);
}

pub fn main() {
    autoreleasepool(|_| {
        let device = <dyn MTLDevice>::system_default().expect("No Metal device available");
        println!("Device: {}", device.name());
        println!();

        print_allocation_timeline();
        println!("{}", "=".repeat(80));
        println!();

        run_benchmark_suite(256, 128, 512, "Short Generation (256p + 128g)");
        println!("{}", "=".repeat(80));
        println!();

        run_benchmark_suite(256, 512, 512, "Medium Generation (256p + 512g)");
        println!("{}", "=".repeat(80));
        println!();

        run_benchmark_suite(256, 2048, 512, "Long Generation (256p + 2048g)");
        println!("{}", "=".repeat(80));
        println!();

        run_size_drift_test();
    });
}
