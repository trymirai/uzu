#![cfg(target_os = "macos")]

use std::time::Instant;

use mach2::{
    kern_return::KERN_SUCCESS,
    message::mach_msg_type_number_t,
    task::task_info,
    task_info::{TASK_BASIC_INFO, task_basic_info},
    traps::mach_task_self,
};
use metal::{MTLHeapDescriptor, prelude::*};
use objc2::{rc::autoreleasepool, runtime::ProtocolObject};

const MB: usize = 1024 * 1024;
const GB: usize = 1024 * MB;

const BUFFER_SIZES: &[usize] = &[
    128 * MB,
    256 * MB,
    512 * MB,
    1 * GB,
    2 * GB,
    4 * GB,
    6 * GB,
    8 * GB,
    10 * GB,
];

const HEAP_SIZE: usize = 12 * GB;

#[derive(Debug, Clone, Copy)]
struct BufferBenchmark {
    size: usize,
    alloc_time_ns: u128,
    drop_time_ns: u128,
}

#[derive(Debug, Clone, Copy)]
struct MemorySnapshot {
    resident_size: u64,
    virtual_size: u64,
}

impl MemorySnapshot {
    fn delta(&self, previous: &Self) -> MemoryDelta {
        MemoryDelta {
            resident_delta: self.resident_size as i64 - previous.resident_size as i64,
            virtual_delta: self.virtual_size as i64 - previous.virtual_size as i64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MemoryDelta {
    resident_delta: i64,
    virtual_delta: i64,
}

fn get_memory_snapshot() -> MemorySnapshot {
    unsafe {
        let mut info = task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            user_time: std::mem::zeroed(),
            system_time: std::mem::zeroed(),
            policy: 0,
            suspend_count: 0,
        };

        let mut count: mach_msg_type_number_t =
            (std::mem::size_of::<task_basic_info>() / std::mem::size_of::<u32>()) as u32;

        let result = task_info(
            mach_task_self(),
            TASK_BASIC_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );

        if result != KERN_SUCCESS {
            panic!("Failed to get task info: {}", result);
        }

        MemorySnapshot {
            resident_size: info.resident_size as u64,
            virtual_size: info.virtual_size as u64,
        }
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_memory_bytes(bytes: u64) -> String {
    if bytes >= GB as u64 {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB as u64 {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_memory_delta(delta: i64) -> String {
    let abs_delta = delta.abs() as u64;
    let sign = if delta >= 0 { "+" } else { "-" };

    if abs_delta >= GB as u64 {
        format!("{}{:.2} GB", sign, abs_delta as f64 / GB as f64)
    } else if abs_delta >= MB as u64 {
        format!("{}{:.2} MB", sign, abs_delta as f64 / MB as f64)
    } else if abs_delta >= 1024 {
        format!("{}{:.2} KB", sign, abs_delta as f64 / 1024.0)
    } else {
        format!("{}{} B", sign, abs_delta)
    }
}

fn benchmark_heap_allocation(device: &ProtocolObject<dyn MTLDevice>) -> Box<[BufferBenchmark]> {
    let descriptor = MTLHeapDescriptor::new();
    descriptor.set_size(HEAP_SIZE);
    descriptor.set_storage_mode(MTLStorageMode::Shared);

    let heap = device
        .new_heap_with_descriptor(&descriptor)
        .expect("Failed to create heap");

    BUFFER_SIZES
        .iter()
        .map(|&size| {
            let (buffer, alloc_time_ns) = autoreleasepool(|_| {
                let start = Instant::now();
                let buffer = heap
                    .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
                    .expect(&format!("Failed to allocate {} buffer", format_bytes(size)));
                let alloc_time_ns = start.elapsed().as_nanos();
                (buffer, alloc_time_ns)
            });

            let drop_time_ns = autoreleasepool(|_| {
                let start = Instant::now();
                drop(buffer);
                start.elapsed().as_nanos()
            });

            BufferBenchmark {
                size,
                alloc_time_ns,
                drop_time_ns,
            }
        })
        .collect()
}

fn benchmark_device_allocation(device: &ProtocolObject<dyn MTLDevice>) -> Box<[BufferBenchmark]> {
    BUFFER_SIZES
        .iter()
        .map(|&size| {
            let (buffer, alloc_time_ns) = autoreleasepool(|_| {
                let start = Instant::now();
                let buffer = device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED);
                let alloc_time_ns = start.elapsed().as_nanos();
                (buffer, alloc_time_ns)
            });

            let drop_time_ns = autoreleasepool(|_| {
                let start = Instant::now();
                drop(buffer);
                start.elapsed().as_nanos()
            });

            BufferBenchmark {
                size,
                alloc_time_ns,
                drop_time_ns,
            }
        })
        .collect()
}

fn calculate_average(benchmarks: &[BufferBenchmark], extract: fn(&BufferBenchmark) -> u128) -> f64 {
    let sum: u128 = benchmarks.iter().map(extract).sum();
    sum as f64 / benchmarks.len() as f64
}

fn calculate_percentage_diff(heap_value: f64, device_value: f64) -> f64 {
    ((device_value - heap_value) / heap_value) * 100.0
}

fn main() {
    autoreleasepool(|_| {
        let device =
            <dyn MTLDevice>::system_default().expect("No Metal device available");
        println!("Device: {}", device.name());
        println!();

        // Track memory before anything
        println!("=== Memory Tracking (macOS Native task_info) ===");
        println!();

        let mem_before = get_memory_snapshot();
        println!("Initial memory usage:");
        println!(
            "  Resident: {}",
            format_memory_bytes(mem_before.resident_size)
        );
        println!(
            "  Virtual:  {}",
            format_memory_bytes(mem_before.virtual_size)
        );
        println!();

        // Create heap and check memory
        let descriptor = MTLHeapDescriptor::new();
        descriptor.set_size(HEAP_SIZE);
        descriptor.set_storage_mode(MTLStorageMode::Shared);

        let heap = device
            .new_heap_with_descriptor(&descriptor)
            .expect("Failed to create heap");

        let mem_after_heap = get_memory_snapshot();
        let heap_delta = mem_after_heap.delta(&mem_before);

        println!("After creating empty {} heap:", format_bytes(HEAP_SIZE));
        println!(
            "  Resident: {} ({})",
            format_memory_bytes(mem_after_heap.resident_size),
            format_memory_delta(heap_delta.resident_delta)
        );
        println!(
            "  Virtual:  {} ({})",
            format_memory_bytes(mem_after_heap.virtual_size),
            format_memory_delta(heap_delta.virtual_delta)
        );
        println!();

        // Allocate buffers from heap and track memory
        let buffer = heap
            .new_buffer(4 * GB, MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("Failed to allocate 4GB buffer from heap");

        let mem_with_buffer = get_memory_snapshot();
        let buffer_delta = mem_with_buffer.delta(&mem_after_heap);

        println!("After allocating 4 GB buffer from heap:");
        println!(
            "  Resident: {} ({})",
            format_memory_bytes(mem_with_buffer.resident_size),
            format_memory_delta(buffer_delta.resident_delta)
        );
        println!(
            "  Virtual:  {} ({})",
            format_memory_bytes(mem_with_buffer.virtual_size),
            format_memory_delta(buffer_delta.virtual_delta)
        );
        println!();

        drop(buffer);

        let mem_after_drop = get_memory_snapshot();
        let drop_delta = mem_after_drop.delta(&mem_with_buffer);

        println!("After dropping buffer:");
        println!(
            "  Resident: {} ({})",
            format_memory_bytes(mem_after_drop.resident_size),
            format_memory_delta(drop_delta.resident_delta)
        );
        println!(
            "  Virtual:  {} ({})",
            format_memory_bytes(mem_after_drop.virtual_size),
            format_memory_delta(drop_delta.virtual_delta)
        );
        println!();

        // Now compare device allocation
        let device_buffer =
            device.new_buffer(4 * GB, MTLResourceOptions::STORAGE_MODE_SHARED);

        let mem_with_device_buffer = get_memory_snapshot();
        let device_buffer_delta = mem_with_device_buffer.delta(&mem_after_drop);

        println!("After allocating 4 GB buffer from device:");
        println!(
            "  Resident: {} ({})",
            format_memory_bytes(mem_with_device_buffer.resident_size),
            format_memory_delta(device_buffer_delta.resident_delta)
        );
        println!(
            "  Virtual:  {} ({})",
            format_memory_bytes(mem_with_device_buffer.virtual_size),
            format_memory_delta(device_buffer_delta.virtual_delta)
        );
        println!();

        drop(device_buffer);
        drop(heap);

        let mem_final = get_memory_snapshot();
        let final_delta = mem_final.delta(&mem_before);

        println!("Final memory usage (after dropping everything):");
        println!(
            "  Resident: {} ({})",
            format_memory_bytes(mem_final.resident_size),
            format_memory_delta(final_delta.resident_delta)
        );
        println!(
            "  Virtual:  {} ({})",
            format_memory_bytes(mem_final.virtual_size),
            format_memory_delta(final_delta.virtual_delta)
        );
        println!();
        println!("{}", "=".repeat(85));
        println!();

        println!("Running heap allocation benchmarks...");
        let heap_results = benchmark_heap_allocation(&device);

        println!("Running device allocation benchmarks...");
        let device_results = benchmark_device_allocation(&device);

        println!();
        println!("=== Heap vs Device Allocation Comparison ===");
        println!();
        println!(
            "{:>10} | {:>15} | {:>15} | {:>15} | {:>15}",
            "Size", "Heap Alloc", "Device Alloc", "Heap Drop", "Device Drop"
        );
        println!("{}", "-".repeat(85));

        for (heap, device) in heap_results.iter().zip(device_results.iter()) {
            println!(
                "{:>10} | {:>12} ns | {:>12} ns | {:>12} ns | {:>12} ns",
                format_bytes(heap.size),
                heap.alloc_time_ns,
                device.alloc_time_ns,
                heap.drop_time_ns,
                device.drop_time_ns
            );
        }

        println!();
        println!("=== Summary Statistics ===");
        println!();

        let heap_avg_alloc = calculate_average(&heap_results, |b| b.alloc_time_ns);
        let device_avg_alloc = calculate_average(&device_results, |b| b.alloc_time_ns);
        let heap_avg_drop = calculate_average(&heap_results, |b| b.drop_time_ns);
        let device_avg_drop = calculate_average(&device_results, |b| b.drop_time_ns);

        let alloc_diff = calculate_percentage_diff(heap_avg_alloc, device_avg_alloc);
        let drop_diff = calculate_percentage_diff(heap_avg_drop, device_avg_drop);

        println!("Average Allocation Time:");
        println!("  Heap:   {:.0} ns", heap_avg_alloc);
        println!("  Device: {:.0} ns", device_avg_alloc);
        println!(
            "  Difference: {:.2}% {}",
            alloc_diff.abs(),
            if alloc_diff > 0.0 {
                "(device slower)"
            } else {
                "(device faster)"
            }
        );
        println!();

        println!("Average Deallocation Time:");
        println!("  Heap:   {:.0} ns", heap_avg_drop);
        println!("  Device: {:.0} ns", device_avg_drop);
        println!(
            "  Difference: {:.2}% {}",
            drop_diff.abs(),
            if drop_diff > 0.0 {
                "(device slower)"
            } else {
                "(device faster)"
            }
        );
    });
}
