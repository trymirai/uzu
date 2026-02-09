#![cfg(target_os = "macos")]

mod allocation_stats;
mod allocator_trait;
mod caching_allocator;
mod device_allocator;
mod heap_allocator;
mod inference_simulation;
mod model_config;
mod simulation_result;

use caching_allocator::CachingAllocator;
use device_allocator::DeviceAllocator;
use heap_allocator::MTLHeapAllocator;
use model_config::ModelConfig;
use simulation_result::{
    run_long_simulation, run_simulation, run_simulation_with_config,
};
use uzu::backends::common::Context;

const KB: f64 = 1024.0;
const MB: f64 = 1024.0 * 1024.0;

fn print_comparison(
    name: &str,
    uzu_result: &simulation_result::SimulationResult,
    heap_result: &simulation_result::SimulationResult,
    device_result: &simulation_result::SimulationResult,
) {
    println!("\n=== {} ===", name);
    println!(
        "{:<20} {:>14} {:>14} {:>14}",
        "Metric", "UZU Caching", "Heap", "Device"
    );
    println!("{:-<62}", "");

    println!(
        "{:<20} {:>14} {:>14} {:>14}",
        "Allocations",
        uzu_result.stats.total_allocs,
        heap_result.stats.total_allocs,
        device_result.stats.total_allocs
    );

    println!(
        "{:<20} {:>14} {:>14} {:>14}",
        "Frees",
        uzu_result.stats.total_frees,
        heap_result.stats.total_frees,
        device_result.stats.total_frees
    );

    println!(
        "{:<20} {:>13.2}µs {:>13.2}µs {:>13.2}µs",
        "Avg alloc time",
        uzu_result.stats.avg_alloc_time_us(),
        heap_result.stats.avg_alloc_time_us(),
        device_result.stats.avg_alloc_time_us()
    );

    println!(
        "{:<20} {:>13.2}µs {:>13.2}µs {:>13.2}µs",
        "Avg free time",
        uzu_result.stats.avg_free_time_us(),
        heap_result.stats.avg_free_time_us(),
        device_result.stats.avg_free_time_us()
    );

    println!(
        "{:<20} {:>12.2} MB {:>12.2} MB {:>12.2} MB",
        "Bytes requested",
        uzu_result.stats.total_bytes_requested as f64 / MB,
        heap_result.stats.total_bytes_requested as f64 / MB,
        device_result.stats.total_bytes_requested as f64 / MB
    );

    println!(
        "{:<20} {:>12.2} MB {:>12.2} MB {:>12.2} MB",
        "Peak memory",
        uzu_result.peak_memory_bytes as f64 / MB,
        heap_result.peak_memory_bytes as f64 / MB,
        device_result.peak_memory_bytes as f64 / MB
    );

    println!(
        "{:<20} {:>12.2} KB {:>12.2} KB {:>12.2} KB",
        "Final cache",
        uzu_result.cache_memory_bytes as f64 / KB,
        heap_result.cache_memory_bytes as f64 / KB,
        device_result.cache_memory_bytes as f64 / KB
    );

    println!();

    let device_time = device_result.stats.avg_alloc_time_us().max(0.001);
    let uzu_speedup =
        device_time / uzu_result.stats.avg_alloc_time_us().max(0.001);
    let heap_speedup =
        device_time / heap_result.stats.avg_alloc_time_us().max(0.001);

    println!(
        "Alloc speedup vs device: UZU {:.1}x, Heap {:.1}x",
        uzu_speedup, heap_speedup
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

    let heap_allocator = MTLHeapAllocator::new(512 * 1024 * 1024);
    let heap_result = run_simulation(
        &heap_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    let device_allocator = DeviceAllocator::new();
    let device_result = run_simulation(
        &device_allocator,
        prompt_length,
        generate_length,
        prefill_step_size,
    );

    print_comparison(
        "Short Conversation Results",
        &uzu_result,
        &heap_result,
        &device_result,
    );

    assert_eq!(
        uzu_result.stats.total_allocs, uzu_result.stats.total_frees,
        "UZU allocator: allocs should equal frees"
    );
    assert_eq!(
        heap_result.stats.total_allocs, heap_result.stats.total_frees,
        "Heap allocator: allocs should equal frees"
    );
    assert_eq!(
        device_result.stats.total_allocs, device_result.stats.total_frees,
        "Device allocator: allocs should equal frees"
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

    let heap_allocator = MTLHeapAllocator::new(1024 * 1024 * 1024);
    let heap_result = run_long_simulation(
        &heap_allocator,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    );

    let device_allocator = DeviceAllocator::new();
    let device_result = run_long_simulation(
        &device_allocator,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    );

    print_comparison(
        "Long Conversation Results",
        &uzu_result,
        &heap_result,
        &device_result,
    );

    print_memory_over_time("UZU Caching", &uzu_result.snapshots, 15);
    print_memory_over_time("Heap", &heap_result.snapshots, 15);
    print_memory_over_time("Device (baseline)", &device_result.snapshots, 15);

    println!("\n--- Memory Efficiency Summary ---");
    println!(
        "UZU peak/requested ratio: {:.2}%",
        (uzu_result.peak_memory_bytes as f64
            / uzu_result.stats.total_bytes_requested as f64)
            * 100.0
    );
    println!(
        "Heap peak/requested ratio: {:.2}%",
        (heap_result.peak_memory_bytes as f64
            / heap_result.stats.total_bytes_requested as f64)
            * 100.0
    );
    println!(
        "Device peak/requested ratio: {:.2}%",
        (device_result.peak_memory_bytes as f64
            / device_result.stats.total_bytes_requested as f64)
            * 100.0
    );

    assert_eq!(
        uzu_result.stats.total_allocs, uzu_result.stats.total_frees,
        "UZU allocator: allocs should equal frees"
    );
    assert_eq!(
        device_result.stats.total_allocs, device_result.stats.total_frees,
        "Device allocator: allocs should equal frees"
    );
}

#[test]
fn test_multi_model_comparison() {
    let prompt_length = 256;
    let generate_length = 512;
    let prefill_step_size = 256;

    println!("\n========================================");
    println!("  Multi-Model Comparison Test");
    println!("  {}p + {}g tokens per model", prompt_length, generate_length);
    println!("========================================");

    println!(
        "\n{:<15} {:>12} {:>12} {:>12}",
        "Model", "UZU (µs)", "Heap (µs)", "Device (µs)"
    );
    println!("{:-<55}", "");

    for config in ModelConfig::all_models() {
        let uzu_allocator = CachingAllocator::new();
        let uzu_result = run_simulation_with_config(
            &uzu_allocator,
            config,
            prompt_length,
            generate_length,
            prefill_step_size,
        );

        let heap_allocator = MTLHeapAllocator::new(512 * 1024 * 1024);
        let heap_result = run_simulation_with_config(
            &heap_allocator,
            config,
            prompt_length,
            generate_length,
            prefill_step_size,
        );

        let device_allocator = DeviceAllocator::new();
        let device_result = run_simulation_with_config(
            &device_allocator,
            config,
            prompt_length,
            generate_length,
            prefill_step_size,
        );

        println!(
            "{:<15} {:>12.2} {:>12.2} {:>12.2}",
            config.name,
            uzu_result.stats.avg_alloc_time_us(),
            heap_result.stats.avg_alloc_time_us(),
            device_result.stats.avg_alloc_time_us()
        );

        assert_eq!(
            uzu_result.stats.total_allocs, uzu_result.stats.total_frees,
            "{}: UZU allocs should equal frees",
            config.name
        );
    }

    println!();
    println!("Summary: Lower allocation time is better.");
}
