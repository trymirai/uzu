#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::backends::{
    common::{Allocator, BufferLifetime},
    metal::{Metal, new_allocator},
};

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

#[derive(Clone, Copy, Default)]
struct AllocationStats {
    total_allocs: usize,
    total_frees: usize,
    total_alloc_time_ns: u128,
    total_free_time_ns: u128,
    total_bytes_requested: usize,
}

impl AllocationStats {
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
        self.total_frees += 1;
        self.total_free_time_ns += duration.as_nanos();
    }

    fn avg_alloc_time_us(&self) -> f64 {
        if self.total_allocs == 0 {
            0.0
        } else {
            self.total_alloc_time_ns as f64 / self.total_allocs as f64 / 1000.0
        }
    }

    fn avg_free_time_us(&self) -> f64 {
        if self.total_frees == 0 {
            0.0
        } else {
            self.total_free_time_ns as f64 / self.total_frees as f64 / 1000.0
        }
    }
}

struct InferenceSimulation<'a> {
    allocator: &'a Allocator<Metal>,
    config: ModelConfig,
    stats: AllocationStats,
    current_prefix_len: usize,
}

impl<'a> InferenceSimulation<'a> {
    fn new(
        allocator: &'a Allocator<Metal>,
        config: ModelConfig,
    ) -> Self {
        Self {
            allocator,
            config,
            stats: AllocationStats::default(),
            current_prefix_len: 0,
        }
    }

    fn alloc_scratch(
        &mut self,
        size: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let start = Instant::now();
        let result = self.allocator.alloc(BufferLifetime::Scratch, size);
        self.stats.record_alloc(size, start.elapsed());
        result.expect("Allocation failed")
    }

    fn free(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) {
        let start = Instant::now();
        self.allocator.free(buffer);
        self.stats.record_free(start.elapsed());
    }

    fn simulate_embedding_kernel(
        &mut self,
        batch_size: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = batch_size * self.config.model_dim * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_attention_kernel(
        &mut self,
        batch_size: usize,
    ) -> Vec<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let mut buffers = Vec::new();

        let qkv_size = batch_size * self.config.qkv_dim() * F16_SIZE;
        buffers.push(self.alloc_scratch(qkv_size));

        let bias_size =
            batch_size * (batch_size + self.current_prefix_len) * F16_SIZE;
        buffers.push(self.alloc_scratch(bias_size));

        let rotated_q_size = self.config.num_heads
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_q_size));

        let rotated_k_size = self.config.num_groups
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_k_size));

        let attn_out_size = batch_size
            * self.config.num_heads
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(attn_out_size));

        buffers
    }

    fn simulate_mlp_kernel(
        &mut self,
        batch_size: usize,
    ) -> Vec<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let mut buffers = Vec::new();

        let up_size = batch_size * 2 * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(up_size));

        let hidden_size = batch_size * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(hidden_size));

        buffers
    }

    fn simulate_logits_kernel(
        &mut self,
        batch_size: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = batch_size * self.config.vocab_size * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_forward_pass(
        &mut self,
        batch_size: usize,
    ) {
        let main = self.simulate_embedding_kernel(batch_size);

        for _layer in 0..self.config.num_layers {
            let attn_buffers = self.simulate_attention_kernel(batch_size);
            for buf in attn_buffers {
                self.free(buf);
            }

            let mlp_buffers = self.simulate_mlp_kernel(batch_size);
            for buf in mlp_buffers {
                self.free(buf);
            }
        }

        let logits = self.simulate_logits_kernel(batch_size);
        self.free(logits);
        self.free(main);
    }

    fn run_prefill(
        &mut self,
        prompt_length: usize,
        step_size: usize,
    ) {
        let mut remaining = prompt_length;

        while remaining > 0 {
            let batch_size = remaining.min(step_size);
            self.simulate_forward_pass(batch_size);
            self.current_prefix_len += batch_size;
            remaining = remaining.saturating_sub(step_size);
        }
    }

    fn run_generation(
        &mut self,
        num_tokens: usize,
    ) {
        for _ in 0..num_tokens {
            self.simulate_forward_pass(1);
            self.current_prefix_len += 1;
        }
    }
}

struct SimulationResult {
    stats: AllocationStats,
    peak_memory_bytes: usize,
    cache_memory_bytes: usize,
    final_prefix_len: usize,
}

fn run_simulation(
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
) -> SimulationResult {
    let device = <dyn MTLDevice>::system_default().expect("No Metal device");
    let allocator =
        new_allocator(device, MTLResourceOptions::STORAGE_MODE_SHARED);

    let config = ModelConfig::llama_3_2_1b();
    let mut sim = InferenceSimulation::new(&allocator, config);

    sim.run_prefill(prompt_length, prefill_step_size);
    sim.run_generation(generate_length);

    SimulationResult {
        stats: sim.stats,
        peak_memory_bytes: allocator.peak_memory(),
        cache_memory_bytes: allocator.cache_memory(),
        final_prefix_len: sim.current_prefix_len,
    }
}

const KB: f64 = 1024.0;
const MB: f64 = 1024.0 * 1024.0;

#[test]
fn test_allocator_short_generation() {
    let result = run_simulation(256, 128, 512);

    println!("=== Short Generation (256p + 128g) ===");
    println!("  Allocations:      {}", result.stats.total_allocs);
    println!("  Frees:            {}", result.stats.total_frees);
    println!("  Avg alloc time:   {:.2} µs", result.stats.avg_alloc_time_us());
    println!("  Avg free time:    {:.2} µs", result.stats.avg_free_time_us());
    println!(
        "  Bytes requested:  {:.2} MB",
        result.stats.total_bytes_requested as f64 / MB
    );
    println!(
        "  Peak memory:      {:.2} MB",
        result.peak_memory_bytes as f64 / MB
    );
    println!(
        "  Cache memory:     {:.2} KB",
        result.cache_memory_bytes as f64 / KB
    );
    println!("  Final prefix:     {}", result.final_prefix_len);

    assert_eq!(result.stats.total_allocs, result.stats.total_frees);
    assert_eq!(result.final_prefix_len, 256 + 128);
    assert!(
        result.peak_memory_bytes < 100 * 1024 * 1024,
        "Peak memory exceeded 100MB"
    );
    assert!(
        result.stats.avg_alloc_time_us() < 100.0,
        "Avg alloc time exceeded 100µs"
    );
}

#[test]
fn test_allocator_long_generation() {
    let result = run_simulation(256, 2048, 512);

    println!("=== Long Generation (256p + 2048g) ===");
    println!("  Allocations:      {}", result.stats.total_allocs);
    println!("  Frees:            {}", result.stats.total_frees);
    println!("  Avg alloc time:   {:.2} µs", result.stats.avg_alloc_time_us());
    println!("  Avg free time:    {:.2} µs", result.stats.avg_free_time_us());
    println!(
        "  Bytes requested:  {:.2} MB",
        result.stats.total_bytes_requested as f64 / MB
    );
    println!(
        "  Peak memory:      {:.2} MB",
        result.peak_memory_bytes as f64 / MB
    );
    println!(
        "  Cache memory:     {:.2} KB",
        result.cache_memory_bytes as f64 / KB
    );
    println!("  Final prefix:     {}", result.final_prefix_len);

    assert_eq!(result.stats.total_allocs, result.stats.total_frees);
    assert_eq!(result.final_prefix_len, 256 + 2048);
    assert!(
        result.peak_memory_bytes < 100 * 1024 * 1024,
        "Peak memory exceeded 100MB"
    );
    assert!(
        result.stats.avg_alloc_time_us() < 100.0,
        "Avg alloc time exceeded 100µs"
    );
}

#[test]
fn test_allocator_cache_reuse() {
    let device = <dyn MTLDevice>::system_default().expect("No Metal device");
    let allocator =
        new_allocator(device, MTLResourceOptions::STORAGE_MODE_SHARED);

    let size = 4096;

    let buf1 = allocator.alloc(BufferLifetime::Scratch, size).unwrap();
    let buf1_id = uzu::backends::common::Buffer::id(&buf1);
    allocator.free(buf1);

    let buf2 = allocator.alloc(BufferLifetime::Scratch, size).unwrap();
    let buf2_id = uzu::backends::common::Buffer::id(&buf2);
    allocator.free(buf2);

    assert_eq!(buf1_id, buf2_id, "Buffer should be reused from cache");

    println!("=== Cache Reuse Test ===");
    println!("  Buffer 1 ID: {}", buf1_id);
    println!("  Buffer 2 ID: {}", buf2_id);
    println!("  Cache hit confirmed");
}

#[test]
fn test_allocator_permanent_not_cached() {
    let device = <dyn MTLDevice>::system_default().expect("No Metal device");
    let allocator =
        new_allocator(device, MTLResourceOptions::STORAGE_MODE_SHARED);

    let size = 4096;

    let cache_before = allocator.cache_memory();

    let buf1 = allocator.alloc(BufferLifetime::Permanent, size).unwrap();
    allocator.free(buf1);

    let buf2 = allocator.alloc(BufferLifetime::Permanent, size).unwrap();
    allocator.free(buf2);

    let cache_after = allocator.cache_memory();

    assert_eq!(
        cache_before, cache_after,
        "Cache should not grow for permanent buffers"
    );

    println!("=== Permanent Buffer Test ===");
    println!("  Cache before: {} bytes", cache_before);
    println!("  Cache after:  {} bytes", cache_after);
    println!("  No caching confirmed");
}

#[test]
fn test_allocator_size_tolerance() {
    let device = <dyn MTLDevice>::system_default().expect("No Metal device");
    let allocator =
        new_allocator(device, MTLResourceOptions::STORAGE_MODE_SHARED);

    let large_size = 1024 * 1024;
    let slightly_smaller = large_size - 1000;

    let buf1 = allocator.alloc(BufferLifetime::Scratch, large_size).unwrap();
    let buf1_id = uzu::backends::common::Buffer::id(&buf1);
    allocator.free(buf1);

    let buf2 =
        allocator.alloc(BufferLifetime::Scratch, slightly_smaller).unwrap();
    let buf2_id = uzu::backends::common::Buffer::id(&buf2);
    allocator.free(buf2);

    assert_eq!(
        buf1_id, buf2_id,
        "Slightly smaller request should reuse larger buffer"
    );

    println!("=== Size Tolerance Test ===");
    println!("  Large buffer ID: {}", buf1_id);
    println!("  Smaller request ID: {}", buf2_id);
    println!("  Size tolerance reuse confirmed");
}
