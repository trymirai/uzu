#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::prelude::*;
use objc2::rc::autoreleasepool;
use uzu::backends::metal::allocator::{
    AllocatedBuffer, ArenaAllocator, CachingAllocator, DirectAllocator, MetalAllocator,
    UzuAllocator,
};

const MB: usize = 1024 * 1024;
const KB: usize = 1024;

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
}

#[derive(Clone)]
struct ScratchBufferSizes {
    token_ids: usize,
    token_positions: usize,
    logits: usize,
    main: usize,
    shortcut: usize,
    qkv: usize,
    attention_output: usize,
    rotated_queries: usize,
    rotated_keys: usize,
    extracted_values: usize,
    mlp_fused_up: usize,
    mlp_hidden: usize,
    attention_bias: usize,
}

impl ScratchBufferSizes {
    fn for_batch_size(batch_size: usize, prefix_len: usize, config: &ModelConfig) -> Self {
        let f16_size = 2;
        let i32_size = 4;
        let u64_size = 8;

        let qkv_dim = (2 * config.num_groups + config.num_heads) * config.head_dim;

        Self {
            token_ids: batch_size * u64_size,
            token_positions: batch_size * i32_size,
            logits: batch_size * config.vocab_size * f16_size,
            main: batch_size * config.model_dim * f16_size,
            shortcut: batch_size * config.model_dim * f16_size,
            qkv: batch_size * qkv_dim * f16_size,
            attention_output: batch_size * config.num_heads * config.head_dim * f16_size,
            rotated_queries: config.num_heads * batch_size * config.head_dim * f16_size,
            rotated_keys: config.num_groups * batch_size * config.head_dim * f16_size,
            extracted_values: config.num_groups * batch_size * config.head_dim * f16_size,
            mlp_fused_up: batch_size * 2 * config.hidden_dim * f16_size,
            mlp_hidden: batch_size * config.hidden_dim * f16_size,
            attention_bias: batch_size * (batch_size + prefix_len) * f16_size,
        }
    }

    fn total_size(&self) -> usize {
        self.token_ids
            + self.token_positions
            + self.logits
            + self.main
            + self.shortcut
            + self.qkv
            + self.attention_output
            + self.rotated_queries
            + self.rotated_keys
            + self.extracted_values
            + self.mlp_fused_up
            + self.mlp_hidden
            + self.attention_bias
    }

    fn buffer_sizes(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("token_ids", self.token_ids),
            ("token_positions", self.token_positions),
            ("logits", self.logits),
            ("main", self.main),
            ("shortcut", self.shortcut),
            ("qkv", self.qkv),
            ("attention_output", self.attention_output),
            ("rotated_queries", self.rotated_queries),
            ("rotated_keys", self.rotated_keys),
            ("extracted_values", self.extracted_values),
            ("mlp_fused_up", self.mlp_fused_up),
            ("mlp_hidden", self.mlp_hidden),
            ("attention_bias", self.attention_bias),
        ]
    }
}

struct InferenceWorkload {
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
    config: ModelConfig,
}

impl InferenceWorkload {
    fn typical() -> Self {
        Self {
            prompt_length: 256,
            generate_length: 128,
            prefill_step_size: 512,
            config: ModelConfig::llama_3_2_1b(),
        }
    }

    fn long_context() -> Self {
        Self {
            prompt_length: 4096,
            generate_length: 256,
            prefill_step_size: 512,
            config: ModelConfig::llama_3_2_1b(),
        }
    }

    fn prefill_steps(&self) -> Vec<(usize, usize)> {
        let mut steps = Vec::new();
        let mut remaining = self.prompt_length;
        let mut prefix_len = 0;

        while remaining > 0 {
            let batch_size = remaining.min(self.prefill_step_size);
            steps.push((batch_size, prefix_len));
            prefix_len += batch_size;
            remaining = remaining.saturating_sub(self.prefill_step_size);
        }

        steps
    }

    fn generate_steps(&self) -> Vec<(usize, usize)> {
        (0..self.generate_length)
            .map(|i| (1, self.prompt_length + i))
            .collect()
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    allocator_name: String,
    scenario: String,
    total_alloc_time: Duration,
    total_free_time: Duration,
    peak_memory: usize,
    cache_memory: usize,
    total_allocations: usize,
    total_bytes_allocated: usize,
}

fn allocate_scratch_buffers<A: MetalAllocator>(
    allocator: &A,
    sizes: &ScratchBufferSizes,
) -> (Vec<AllocatedBuffer>, Duration) {
    let start = Instant::now();
    let buffers: Vec<_> = sizes
        .buffer_sizes()
        .iter()
        .filter_map(|(_, size)| {
            if *size > 0 {
                allocator.alloc_scratch(*size).ok()
            } else {
                None
            }
        })
        .collect();
    let duration = start.elapsed();
    (buffers, duration)
}

fn free_buffers<A: MetalAllocator>(allocator: &A, buffers: Vec<AllocatedBuffer>) -> Duration {
    let start = Instant::now();
    for buffer in buffers {
        allocator.free(buffer);
    }
    start.elapsed()
}

fn benchmark_max_size_preallocation<A: MetalAllocator>(
    allocator: &A,
    allocator_name: &str,
    workload: &InferenceWorkload,
) -> BenchmarkResult {
    let max_batch_size = workload.prefill_step_size;
    let max_prefix_len = workload.prompt_length + workload.generate_length;

    let max_sizes = ScratchBufferSizes::for_batch_size(max_batch_size, max_prefix_len, &workload.config);

    allocator.reset_peak_memory();

    let (buffers, alloc_time) = allocate_scratch_buffers(allocator, &max_sizes);
    let total_allocations = buffers.len();
    let total_bytes = max_sizes.total_size();

    let prefill_steps = workload.prefill_steps().len();
    let generate_steps = workload.generate_steps().len();
    let _total_iterations = prefill_steps + generate_steps;

    let free_time = free_buffers(allocator, buffers);

    BenchmarkResult {
        allocator_name: allocator_name.to_string(),
        scenario: "max_size_preallocation".to_string(),
        total_alloc_time: alloc_time,
        total_free_time: free_time,
        peak_memory: allocator.peak_memory(),
        cache_memory: allocator.cache_memory(),
        total_allocations,
        total_bytes_allocated: total_bytes,
    }
}

fn benchmark_on_demand<A: MetalAllocator>(
    allocator: &A,
    allocator_name: &str,
    workload: &InferenceWorkload,
) -> BenchmarkResult {
    allocator.reset_peak_memory();
    allocator.clear_cache();

    let mut total_alloc_time = Duration::ZERO;
    let mut total_free_time = Duration::ZERO;
    let mut total_allocations = 0;
    let mut total_bytes = 0;

    for (batch_size, prefix_len) in workload.prefill_steps() {
        let sizes = ScratchBufferSizes::for_batch_size(batch_size, prefix_len, &workload.config);
        let (buffers, alloc_time) = allocate_scratch_buffers(allocator, &sizes);

        total_alloc_time += alloc_time;
        total_allocations += buffers.len();
        total_bytes += sizes.total_size();

        let free_time = free_buffers(allocator, buffers);
        total_free_time += free_time;
    }

    for (batch_size, prefix_len) in workload.generate_steps() {
        let sizes = ScratchBufferSizes::for_batch_size(batch_size, prefix_len, &workload.config);
        let (buffers, alloc_time) = allocate_scratch_buffers(allocator, &sizes);

        total_alloc_time += alloc_time;
        total_allocations += buffers.len();
        total_bytes += sizes.total_size();

        let free_time = free_buffers(allocator, buffers);
        total_free_time += free_time;
    }

    BenchmarkResult {
        allocator_name: allocator_name.to_string(),
        scenario: "on_demand".to_string(),
        total_alloc_time,
        total_free_time,
        peak_memory: allocator.peak_memory(),
        cache_memory: allocator.cache_memory(),
        total_allocations,
        total_bytes_allocated: total_bytes,
    }
}

fn benchmark_arena_on_demand(
    allocator: &ArenaAllocator,
    workload: &InferenceWorkload,
) -> BenchmarkResult {
    let max_batch_size = workload.prefill_step_size;
    let max_prefix_len = workload.prompt_length + workload.generate_length;
    let max_sizes = ScratchBufferSizes::for_batch_size(max_batch_size, max_prefix_len, &workload.config);

    allocator
        .init_arena(max_sizes.total_size())
        .expect("Failed to init arena");
    allocator.reset_peak_memory();

    let mut total_alloc_time = Duration::ZERO;
    let mut total_free_time = Duration::ZERO;
    let mut total_allocations = 0;
    let mut total_bytes = 0;

    for (batch_size, prefix_len) in workload.prefill_steps() {
        let sizes = ScratchBufferSizes::for_batch_size(batch_size, prefix_len, &workload.config);
        let (buffers, alloc_time) = allocate_scratch_buffers(allocator, &sizes);

        total_alloc_time += alloc_time;
        total_allocations += buffers.len();
        total_bytes += sizes.total_size();

        let free_time = free_buffers(allocator, buffers);
        total_free_time += free_time;

        allocator.reset_arena();
    }

    for (batch_size, prefix_len) in workload.generate_steps() {
        let sizes = ScratchBufferSizes::for_batch_size(batch_size, prefix_len, &workload.config);
        let (buffers, alloc_time) = allocate_scratch_buffers(allocator, &sizes);

        total_alloc_time += alloc_time;
        total_allocations += buffers.len();
        total_bytes += sizes.total_size();

        let free_time = free_buffers(allocator, buffers);
        total_free_time += free_time;

        allocator.reset_arena();
    }

    let (used, total) = allocator.arena_usage();

    BenchmarkResult {
        allocator_name: "ArenaAllocator".to_string(),
        scenario: "on_demand".to_string(),
        total_alloc_time,
        total_free_time,
        peak_memory: allocator.peak_memory(),
        cache_memory: total - used,
        total_allocations,
        total_bytes_allocated: total_bytes,
    }
}

fn print_workload_info(workload: &InferenceWorkload) {
    println!("Workload Configuration:");
    println!("  Prompt length: {} tokens", workload.prompt_length);
    println!("  Generate length: {} tokens", workload.generate_length);
    println!("  Prefill step size: {} tokens", workload.prefill_step_size);
    println!("  Prefill steps: {}", workload.prefill_steps().len());
    println!("  Generate steps: {}", workload.generate_steps().len());
    println!();

    let max_batch = workload.prefill_step_size;
    let max_prefix = workload.prompt_length + workload.generate_length;
    let max_sizes = ScratchBufferSizes::for_batch_size(max_batch, max_prefix, &workload.config);

    println!("Buffer Sizes at Max Capacity (batch={}, prefix={}):", max_batch, max_prefix);
    for (name, size) in max_sizes.buffer_sizes() {
        if size > MB {
            println!("  {:<20}: {:>8.2} MB", name, size as f64 / MB as f64);
        } else if size > KB {
            println!("  {:<20}: {:>8.2} KB", name, size as f64 / KB as f64);
        } else {
            println!("  {:<20}: {:>8} B", name, size);
        }
    }
    println!("  {:<20}: {:>8.2} MB", "TOTAL", max_sizes.total_size() as f64 / MB as f64);
    println!();

    let gen_sizes = ScratchBufferSizes::for_batch_size(1, max_prefix, &workload.config);
    println!("Buffer Sizes During Generation (batch=1, prefix={}):", max_prefix);
    for (name, size) in gen_sizes.buffer_sizes() {
        if size > MB {
            println!("  {:<20}: {:>8.2} MB", name, size as f64 / MB as f64);
        } else if size > KB {
            println!("  {:<20}: {:>8.2} KB", name, size as f64 / KB as f64);
        } else {
            println!("  {:<20}: {:>8} B", name, size);
        }
    }
    println!("  {:<20}: {:>8.2} MB", "TOTAL", gen_sizes.total_size() as f64 / MB as f64);
    println!();

    let savings = max_sizes.total_size() - gen_sizes.total_size();
    let savings_pct = savings as f64 / max_sizes.total_size() as f64 * 100.0;
    println!(
        "Memory savings during generation: {:.2} MB ({:.1}%)",
        savings as f64 / MB as f64,
        savings_pct
    );
    println!();
}

fn print_result(result: &BenchmarkResult) {
    println!(
        "  {} ({}):",
        result.allocator_name, result.scenario
    );
    println!(
        "    Alloc time:    {:>10.3} ms ({} allocations)",
        result.total_alloc_time.as_secs_f64() * 1000.0,
        result.total_allocations
    );
    println!(
        "    Free time:     {:>10.3} ms",
        result.total_free_time.as_secs_f64() * 1000.0
    );
    println!(
        "    Total time:    {:>10.3} ms",
        (result.total_alloc_time + result.total_free_time).as_secs_f64() * 1000.0
    );
    println!(
        "    Peak memory:   {:>10.2} MB",
        result.peak_memory as f64 / MB as f64
    );
    println!(
        "    Cache memory:  {:>10.2} MB",
        result.cache_memory as f64 / MB as f64
    );
    println!(
        "    Bytes alloc'd: {:>10.2} MB",
        result.total_bytes_allocated as f64 / MB as f64
    );
}

fn run_benchmark_suite(workload: &InferenceWorkload, name: &str) {
    println!("=== {} ===", name);
    println!();
    print_workload_info(workload);

    let direct = DirectAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let caching = CachingAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let uzu = UzuAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );
    let arena = ArenaAllocator::new(
        <dyn MTLDevice>::system_default().expect("No Metal device"),
    );

    println!("--- Scenario A: Max-size Pre-allocation ---");
    println!();
    let direct_max = benchmark_max_size_preallocation(&direct, "DirectAllocator", workload);
    print_result(&direct_max);
    println!();
    let caching_max = benchmark_max_size_preallocation(&caching, "CachingAllocator", workload);
    print_result(&caching_max);
    println!();
    let uzu_max = benchmark_max_size_preallocation(&uzu, "UzuAllocator", workload);
    print_result(&uzu_max);
    println!();

    println!("--- Scenario B: On-demand Allocation ---");
    println!();
    let direct_demand = benchmark_on_demand(&direct, "DirectAllocator", workload);
    print_result(&direct_demand);
    println!();
    let caching_demand = benchmark_on_demand(&caching, "CachingAllocator", workload);
    print_result(&caching_demand);
    println!();
    let uzu_demand = benchmark_on_demand(&uzu, "UzuAllocator", workload);
    print_result(&uzu_demand);
    println!();

    println!("--- Scenario C: Arena Allocator (heap suballocation) ---");
    println!();
    let arena_demand = benchmark_arena_on_demand(&arena, workload);
    print_result(&arena_demand);
    println!();

    println!("--- Summary Comparison ---");
    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Allocator", "Max Alloc", "Max Total", "OnDemand", "OD Total"
    );
    println!("{}", "-".repeat(72));

    for (max_res, od_res) in [
        (&direct_max, &direct_demand),
        (&caching_max, &caching_demand),
        (&uzu_max, &uzu_demand),
        (&uzu_max, &arena_demand), // Arena doesn't have max-size, use uzu for comparison
    ] {
        let max_total = max_res.total_alloc_time + max_res.total_free_time;
        let od_total = od_res.total_alloc_time + od_res.total_free_time;
        println!(
            "{:<20} {:>9.3} ms {:>9.3} ms {:>9.3} ms {:>9.3} ms",
            od_res.allocator_name,
            max_res.total_alloc_time.as_secs_f64() * 1000.0,
            max_total.as_secs_f64() * 1000.0,
            od_res.total_alloc_time.as_secs_f64() * 1000.0,
            od_total.as_secs_f64() * 1000.0,
        );
    }
    println!();

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Allocator", "Max Peak", "Max Cache", "OD Peak", "OD Cache"
    );
    println!("{}", "-".repeat(72));

    for (max_res, od_res) in [
        (&direct_max, &direct_demand),
        (&caching_max, &caching_demand),
        (&uzu_max, &uzu_demand),
        (&uzu_max, &arena_demand),
    ] {
        println!(
            "{:<20} {:>9.2} MB {:>9.2} MB {:>9.2} MB {:>9.2} MB",
            od_res.allocator_name,
            max_res.peak_memory as f64 / MB as f64,
            max_res.cache_memory as f64 / MB as f64,
            od_res.peak_memory as f64 / MB as f64,
            od_res.cache_memory as f64 / MB as f64,
        );
    }
    println!();
}

fn main() {
    autoreleasepool(|_| {
        let device = <dyn MTLDevice>::system_default().expect("No Metal device available");
        println!("Device: {}", device.name());
        println!();

        run_benchmark_suite(&InferenceWorkload::typical(), "Typical Workload (256 prompt, 128 generate)");
        println!("{}", "=".repeat(80));
        println!();
        run_benchmark_suite(&InferenceWorkload::long_context(), "Long Context Workload (4096 prompt, 256 generate)");
    });
}
