#![cfg(target_os = "macos")]

mod allocation_stats;
mod allocator_trait;
mod caching_allocator;
mod heap_allocator;
mod inference_simulation;
mod model_config;
mod simulation_result;

use allocation_stats::AllocationStats;
use allocator_trait::AllocatorTrait;
use caching_allocator::CachingAllocator;
use heap_allocator::MTLHeapAllocator;
use inference_simulation::InferenceSimulation;
use model_config::ModelConfig;
use simulation_result::{SimulationResult, run_simulation};

const KB: f64 = 1024.0;
const MB: f64 = 1024.0 * 1024.0;

#[test]
fn test_allocator_comparison() {
    let prompt_length = 256;
    let generate_length = 2048;
    let prefill_step_size = 512;

    println!("\n=== Allocator Performance Comparison ===");
    println!(
        "Simulation: {}p + {}g tokens, step size {}",
        prompt_length, generate_length, prefill_step_size
    );
    println!();

    // Run with caching allocator
    let caching_allocator = CachingAllocator::new();
    let caching_result = run_simulation(
        &caching_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    // Run with heap allocator (no caching)
    let heap_allocator = MTLHeapAllocator::new();
    let heap_result = run_simulation(
        &heap_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    // Print comparison table
    println!("{:<25} {:>15} {:>15}", "Metric", "Caching", "Heap (no cache)");
    println!("{:-<55}", "");

    println!(
        "{:<25} {:>15} {:>15}",
        "Allocations",
        caching_result.stats.total_allocs,
        heap_result.stats.total_allocs
    );

    println!(
        "{:<25} {:>15} {:>15}",
        "Frees",
        caching_result.stats.total_frees,
        heap_result.stats.total_frees
    );

    println!(
        "{:<25} {:>14.2}µs {:>14.2}µs",
        "Avg alloc time",
        caching_result.stats.avg_alloc_time_us(),
        heap_result.stats.avg_alloc_time_us()
    );

    println!(
        "{:<25} {:>14.2}µs {:>14.2}µs",
        "Avg free time",
        caching_result.stats.avg_free_time_us(),
        heap_result.stats.avg_free_time_us()
    );

    println!(
        "{:<25} {:>13.2} MB {:>13.2} MB",
        "Bytes requested",
        caching_result.stats.total_bytes_requested as f64 / MB,
        heap_result.stats.total_bytes_requested as f64 / MB
    );

    println!(
        "{:<25} {:>13.2} MB {:>13.2} MB",
        "Peak memory",
        caching_result.peak_memory_bytes as f64 / MB,
        heap_result.peak_memory_bytes as f64 / MB
    );

    println!(
        "{:<25} {:>13.2} KB {:>13.2} KB",
        "Cache memory",
        caching_result.cache_memory_bytes as f64 / KB,
        heap_result.cache_memory_bytes as f64 / KB
    );

    println!();

    // Calculate speedup
    let alloc_speedup = heap_result.stats.avg_alloc_time_us()
        / caching_result.stats.avg_alloc_time_us();
    let free_speedup = heap_result.stats.avg_free_time_us()
        / caching_result.stats.avg_free_time_us();

    println!(
        "Caching allocator is {:.1}x faster for alloc, {:.1}x faster for free",
        alloc_speedup, free_speedup
    );

    // Assertions
    assert_eq!(
        caching_result.stats.total_allocs, caching_result.stats.total_frees,
        "Caching allocator: allocs should equal frees"
    );
    assert_eq!(
        heap_result.stats.total_allocs, heap_result.stats.total_frees,
        "Heap allocator: allocs should equal frees"
    );
    assert_eq!(
        caching_result.final_prefix_len,
        prompt_length + generate_length
    );
    assert_eq!(heap_result.final_prefix_len, prompt_length + generate_length);
}
