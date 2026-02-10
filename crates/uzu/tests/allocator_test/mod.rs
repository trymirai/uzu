#![cfg(target_os = "macos")]

mod allocation_stats;
mod allocator_trait;
mod caching_allocator;
mod device_allocator;
mod heap_allocator;
mod inference_simulation;
mod mlx_allocator;
mod model_config;
mod simulation_result;

use caching_allocator::CachingAllocator;
use device_allocator::DeviceAllocator;
use heap_allocator::MTLHeapAllocator;
use mlx_allocator::MlxAllocator;
use simulation_result::{run_long_simulation, run_simulation};
use uzu::backends::common::Context;

const KB: f64 = 1024.0;
const MB: f64 = 1024.0 * 1024.0;
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

fn print_three_way_comparison(
    name: &str,
    uzu_result: &simulation_result::SimulationResult,
    mlx_result: &simulation_result::SimulationResult,
    heap_result: &simulation_result::SimulationResult,
) {
    println!("\n=== {} ===", name);
    println!(
        "{:<25} {:>15} {:>15} {:>15}",
        "Metric", "UZU Caching", "MLX-style LRU", "Heap (baseline)"
    );
    println!("{:-<72}", "");

    println!(
        "{:<25} {:>15} {:>15} {:>15}",
        "Allocations",
        uzu_result.stats.total_allocs,
        mlx_result.stats.total_allocs,
        heap_result.stats.total_allocs
    );

    println!(
        "{:<25} {:>15} {:>15} {:>15}",
        "Frees",
        uzu_result.stats.total_frees,
        mlx_result.stats.total_frees,
        heap_result.stats.total_frees
    );

    println!(
        "{:<25} {:>14.2}µs {:>14.2}µs {:>14.2}µs",
        "Avg alloc time",
        uzu_result.stats.avg_alloc_time_us(),
        mlx_result.stats.avg_alloc_time_us(),
        heap_result.stats.avg_alloc_time_us()
    );

    println!(
        "{:<25} {:>14.2}µs {:>14.2}µs {:>14.2}µs",
        "Avg free time",
        uzu_result.stats.avg_free_time_us(),
        mlx_result.stats.avg_free_time_us(),
        heap_result.stats.avg_free_time_us()
    );

    println!(
        "{:<25} {:>13.2} MB {:>13.2} MB {:>13.2} MB",
        "Bytes requested",
        uzu_result.stats.total_bytes_requested as f64 / MB,
        mlx_result.stats.total_bytes_requested as f64 / MB,
        heap_result.stats.total_bytes_requested as f64 / MB
    );

    println!(
        "{:<25} {:>13.2} MB {:>13.2} MB {:>13.2} MB",
        "Peak memory",
        uzu_result.peak_memory_bytes as f64 / MB,
        mlx_result.peak_memory_bytes as f64 / MB,
        heap_result.peak_memory_bytes as f64 / MB
    );

    println!(
        "{:<25} {:>13.2} KB {:>13.2} KB {:>13.2} KB",
        "Final cache",
        uzu_result.cache_memory_bytes as f64 / KB,
        mlx_result.cache_memory_bytes as f64 / KB,
        heap_result.cache_memory_bytes as f64 / KB
    );

    println!();

    let uzu_speedup = heap_result.stats.avg_alloc_time_us()
        / uzu_result.stats.avg_alloc_time_us().max(0.001);
    let mlx_speedup = heap_result.stats.avg_alloc_time_us()
        / mlx_result.stats.avg_alloc_time_us().max(0.001);

    println!(
        "Alloc speedup vs heap: UZU {:.1}x, MLX {:.1}x",
        uzu_speedup, mlx_speedup
    );
}

fn print_memory_over_time(
    name: &str,
    snapshots: &[inference_simulation::MemorySnapshot],
    sample_count: usize,
) {
    if snapshots.is_empty() {
        return;
    }

    println!("\n--- {} Memory Over Time (sampled) ---", name);
    println!("{:>8} {:>12} {:>12} {:>12}", "Step", "Active", "Cache", "Peak");

    let step = (snapshots.len() / sample_count).max(1);
    for (i, snapshot) in snapshots.iter().enumerate() {
        if i % step == 0 || i == snapshots.len() - 1 {
            println!(
                "{:>8} {:>10.2}MB {:>10.2}MB {:>10.2}MB",
                snapshot.step,
                snapshot.active_memory as f64 / MB,
                snapshot.cache_memory as f64 / MB,
                snapshot.peak_memory as f64 / MB
            );
        }
    }
}

#[test]
fn test_allocator_comparison() {
    let prompt_length = 256;
    let generate_length = 2048;
    let prefill_step_size = 512;

    println!("\n========================================");
    println!("  Short Conversation Test");
    println!("  {}p + {}g tokens", prompt_length, generate_length);
    println!("========================================");

    let uzu_allocator = CachingAllocator::new();
    let uzu_result = run_simulation(
        &uzu_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    let mlx_allocator = MlxAllocator::new(512 * 1024 * 1024, 256 * 1024 * 1024);
    let mlx_result = run_simulation(
        &mlx_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    let heap_allocator = MTLHeapAllocator::new(512 * 1024 * 1024);
    let heap_result = run_simulation(
        &heap_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    print_three_way_comparison(
        "Short Conversation Results",
        &uzu_result,
        &mlx_result,
        &heap_result,
    );

    assert_eq!(
        uzu_result.stats.total_allocs, uzu_result.stats.total_frees,
        "UZU allocator: allocs should equal frees"
    );
    assert_eq!(
        mlx_result.stats.total_allocs, mlx_result.stats.total_frees,
        "MLX allocator: allocs should equal frees"
    );
    assert_eq!(
        heap_result.stats.total_allocs, heap_result.stats.total_frees,
        "Heap allocator: allocs should equal frees"
    );
}

#[test]
fn test_long_conversation_comparison() {
    let num_conversations = 10;
    let avg_prompt_length = 512;
    let avg_generate_length = 1024;
    let prefill_step_size = 512;
    let snapshot_interval = 100;

    println!("\n========================================");
    println!("  Long Conversation Test");
    println!(
        "  {} conversations, ~{}p + ~{}g tokens each",
        num_conversations, avg_prompt_length, avg_generate_length
    );
    println!("========================================");

    let gc_limit = 256 * 1024 * 1024;
    let max_pool = 128 * 1024 * 1024;

    let uzu_allocator = CachingAllocator::new();
    uzu_allocator.context.allocator().set_gc_limit(gc_limit);
    uzu_allocator.context.allocator().set_max_pool_size(max_pool);

    let uzu_result = run_long_simulation(
        &uzu_allocator,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    );

    let mlx_allocator = MlxAllocator::new(gc_limit, max_pool);
    let mlx_result = run_long_simulation(
        &mlx_allocator,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    );

    let heap_allocator = MTLHeapAllocator::new(1024 * 1024 * 1024);
    let heap_result = run_long_simulation(
        &heap_allocator,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    );

    print_three_way_comparison(
        "Long Conversation Results",
        &uzu_result,
        &mlx_result,
        &heap_result,
    );

    print_memory_over_time("UZU Caching", &uzu_result.snapshots, 15);
    print_memory_over_time("MLX-style LRU", &mlx_result.snapshots, 15);
    print_memory_over_time("Heap (baseline)", &heap_result.snapshots, 15);

    println!("\n--- Memory Efficiency Summary ---");
    println!(
        "UZU peak/requested ratio: {:.2}%",
        (uzu_result.peak_memory_bytes as f64
            / uzu_result.stats.total_bytes_requested as f64)
            * 100.0
    );
    println!(
        "MLX peak/requested ratio: {:.2}%",
        (mlx_result.peak_memory_bytes as f64
            / mlx_result.stats.total_bytes_requested as f64)
            * 100.0
    );
    println!(
        "Heap peak/requested ratio: {:.2}%",
        (heap_result.peak_memory_bytes as f64
            / heap_result.stats.total_bytes_requested as f64)
            * 100.0
    );

    assert_eq!(
        uzu_result.stats.total_allocs, uzu_result.stats.total_frees,
        "UZU allocator: allocs should equal frees"
    );
    assert_eq!(
        mlx_result.stats.total_allocs, mlx_result.stats.total_frees,
        "MLX allocator: allocs should equal frees"
    );
}
